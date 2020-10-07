# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
import torch.distributed as dist
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class BaseAgent(object):
    """ Base class for an VLN agent to generate and save trajectories. """

    def __init__(self, dataloader, results_path):
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.results_path = results_path
        self.results = {}
        self.losses = []  # For learning agents

    def write_results(self):
        output = [{"inst_idx": k, "trajectory": v} for k, v in self.results.items()]
        with open(self.results_path, "w") as f:
            json.dump(output, f)

    def rollout(self):
        """ Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  """
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def reset_dataloader(self):
        self.data_iter = iter(self.dataloader)
        self.dataloader.batch = None

    def test(self):
        self.reset_dataloader()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        # print "Testing %s" % self.__class__.__name__
        looped = False
        with torch.no_grad():
            while True:
                for traj in self.rollout(train=False):
                    if traj["inst_idx"] in self.results:
                        looped = True
                    else:
                        self.results[traj["inst_idx"]] = traj["path"]
                if looped:
                    break

    def test_only_next_action(self):
        self.reset_dataloader()
        self.results = {}
        self.losses = []
        self.correct = []

        looped = False
        while True:
            trajs, loss, correct = self.rollout(return_for_next_action=True)
            self.losses.append(loss)
            self.correct.append(correct)
            for traj in trajs:
                if traj["inst_idx"] in self.results:
                    looped = True
                else:
                    self.results[traj["inst_idx"]] = []
            if looped:
                break

        print("Avg Loss: %0.4f" % np.mean(self.losses))
        print("Avg Score: %0.4f" % np.mean(self.correct))


class OscarAgent(BaseAgent):
    """ An agent based on Oscar model. """

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = [
        "left",
        "right",
        "up",
        "down",
        "forward",
        "<end>",
        "<start>",
        "<ignore>",
    ]
    # fmt: off
    env_actions = [
      (0, -1,  0),  # left
      (0,  1,  0),  # right
      (0,  0,  1),  # up
      (0,  0, -1),  # down
      (1,  0,  0),  # forward
      (0,  0,  0),  # <end>
      (0,  0,  0),  # <start>
      (0,  0,  0)   # <ignore>
    ]
    # fmt: on
    feedback_options = ["teacher", "argmax", "sample"]

    def __init__(
        self,
        args,
        dataloader,
        results_path,
        model,
        episode_len=20,
        use_oscar_settings=False,
        add_region_labels=True,
    ):
        super(OscarAgent, self).__init__(dataloader, results_path)
        self.args = args
        self.model = model

        self.add_region_labels = add_region_labels

        self.episode_len = episode_len
        self.losses = []

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.model_actions.index("<ignore>")
        )

        self.use_oscar_settings = use_oscar_settings

    @staticmethod
    def n_inputs():
        return len(OscarAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(OscarAgent.model_actions) - 2  # Model doesn't output start or ignore

    def _teacher_action(self, obs, ended, device):
        """ Extract teacher actions into variable. """
        a = torch.LongTensor(len(obs))
        for i, ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix, heading_chg, elevation_chg = ob["teacher"]
            if heading_chg > 0:
                a[i] = self.model_actions.index("right")
            elif heading_chg < 0:
                a[i] = self.model_actions.index("left")
            elif elevation_chg > 0:
                a[i] = self.model_actions.index("up")
            elif elevation_chg < 0:
                a[i] = self.model_actions.index("down")
            elif ix > 0:
                a[i] = self.model_actions.index("forward")
            elif ended[i]:
                a[i] = self.model_actions.index("<ignore>")
            else:
                a[i] = self.model_actions.index("<end>")
        return Variable(a, requires_grad=False).to(device)

    def _get_batch(self, reset_epoch=False):
        if reset_epoch == True:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        else:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)
        self.dataloader.batch = batch
        return batch

    def _verify_batch_size(self, batch):
        batch_size = self.dataloader.batch_size

        if len(batch) != batch_size:
            # print("Batch length not equal to batch size, padding towards the end!")
            remaining_no = batch_size - len(batch)
            extra_batch = self._get_batch()
            new_batch = batch + extra_batch[: batch_size - len(batch)]
            assert batch_size == len(new_batch)
            self.dataloader.batch = new_batch
            return new_batch
        return batch

    def _retrieve_region_labels_tokens(
        self, scan_id, viewpoint_viewpointIdx, MAX_REGION_LABELS_LENGTH
    ):
        region_labels = []
        for viewpoint_id, view_idx in viewpoint_viewpointIdx:
            long_id = f"{scan_id}_{viewpoint_id}_{view_idx}"
            region_label = self.dataloader.region_labels[long_id]
            region_labels.extend(region_label)
        # print(region_labels)
        region_labels = set(region_labels)
        region_labels = " ".join(region_labels)
        token_region_labels = self.dataloader.tokenizer.tokenize(region_labels)
        token_region_labels = token_region_labels[-MAX_REGION_LABELS_LENGTH:]
        return token_region_labels

    def _retrieve_img_features(
        self, scan_id, viewpoint_viewpointIdx, MAX_IMG_FEATURES_LENGTH
    ):
        img_features = []
        for viewpoint_id, view_idx in viewpoint_viewpointIdx:
            long_id = f"{scan_id}_{viewpoint_id}_{view_idx}"
            feature = self.dataloader.features["features"][long_id]
            img_features.append(feature)
        if len(img_features) == 0:
            import pdb

            pdb.set_trace()
        img_features = np.concatenate(img_features, axis=0)
        img_features = img_features[-MAX_IMG_FEATURES_LENGTH:]
        return img_features

    def _retrieve_action_labels_tokens(self, actions, MAX_ACTION_LABELS_LENGTH):
        action_dict = {
            0: "left",
            1: "right",
            2: "up",
            3: "down",
            4: "forward",
            5: "stop",
        }
        actions_str = ""
        for action in actions:
            actions_str += f"{action_dict[action]} "
        actions_tokens = self.dataloader.tokenizer.tokenize(actions_str)
        actions_tokens = actions_tokens[-MAX_ACTION_LABELS_LENGTH:]
        return actions_tokens

    def rollout(self, return_for_next_action=False, train=True):
        if self.use_oscar_settings:
            pad_token = 0

            cls_token_segment_id = 1
            pad_token_segment_id = 0
            sep_token_segment_id = 0

            tar_token_segment_id = 0
            ques_token_segment_id = 0
            ans_token_segment_id = 0
            region_label_token_segment_id = 1
            # action_token_segment_id = 5

            # MAX_DIALOG_LEN = 512  # including [QUES]s and [ANS]s
            MAX_DIALOG_LEN = 114  # including [QUES]s and [ANS]s
            # MAX_ACTION_LEN = 128 - 1  # -1 for [SEP] after Action Stream
            MAX_REGION_LABEL_LEN = 10 - 1  # -1 for [SEP] after Region Stream
            MAX_TARGET_LENGTH = 4 - 2  # [CLS], [SEP] after QA and before Action
            MAX_IMG_LENGTH = 10
            # TODO: ^^ add them as args ^^

            # TOTAL 128, 10

        else:
            pad_token = 0
            cls_token_segment_id = 0
            pad_token_segment_id = 0
            sep_token_segment_id = 0
            tar_token_segment_id = 1
            ques_token_segment_id = 2
            ans_token_segment_id = 3
            region_label_token_segment_id = 4
            action_token_segment_id = 5

            MAX_DIALOG_LEN = 256  # including [QUES]s and [ANS]s
            MAX_ACTION_LEN = 128 - 1  # -1 for [SEP] after Action Stream
            MAX_REGION_LABEL_LEN = 64 - 8
            MAX_TARGET_LENGTH = 8 - 3  # [CLS], [TAR], [SEP] after QA and before Action
            MAX_IMG_LENGTH = 128
            # TODO: ^^ add them as args ^^

            # TOTAL 896

        batch = self._get_batch()
        batch = self._verify_batch_size(batch)

        scan_ids = [item["scan"] for item in batch]

        obs = np.array(self.dataloader.reset())

        batch_size = len(obs)

        # Record starting point
        traj = [
            {
                "inst_idx": ob["inst_idx"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in obs
        ]

        current_viewpoint_viewpointIdx = [
            (item["viewpoint"], item["viewIndex"]) for item in obs
        ]

        # Initial action
        a_t = Variable(
            torch.ones(batch_size).long() * self.model_actions.index("<start>"),
            requires_grad=False,
        ).to(self.args.device)

        ended = np.array([False] * batch_size)

        # Do a sequence rollout and calculate the loss
        self.loss = torch.zeros(1).to(self.args.device)
        if self.args.detach_loss:
            self.non_avg_loss = torch.zeros(1).to(self.args.device)
        env_action = [None] * batch_size

        for t in range(self.episode_len):

            batch_input_ids = []
            batch_input_masks = []
            batch_segment_ids = []
            batch_img_features = []

            for i in range(batch_size):
                tokens = []
                segment_ids = []
                target_dialog_tokens = batch[i]["target_dialog_tokens"]
                target_dialog_segment_ids = batch[i]["target_dialog_segment_ids"]

                tokens += target_dialog_tokens
                segment_ids += target_dialog_segment_ids
                if self.add_region_labels:
                    region_labels_tokens = self._retrieve_region_labels_tokens(
                        scan_ids[i],
                        [current_viewpoint_viewpointIdx[i]],
                        MAX_REGION_LABEL_LEN,
                    )

                    tokens += region_labels_tokens
                    segment_ids += [region_label_token_segment_id] * len(
                        region_labels_tokens
                    )

                    tokens += [self.dataloader.tokenizer.sep_token]
                    segment_ids += [sep_token_segment_id]

                input_ids = self.dataloader.tokenizer.convert_tokens_to_ids(tokens)
                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                # Zero-pad up to the sequence length.
                padding_length = self.args.max_seq_length - len(input_ids)

                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                input_ids = input_ids[: self.args.max_seq_length]
                input_mask = input_mask[: self.args.max_seq_length]
                segment_ids = segment_ids[: self.args.max_seq_length]

                assert len(input_ids) == self.args.max_seq_length
                assert len(input_mask) == self.args.max_seq_length
                assert len(segment_ids) == self.args.max_seq_length

                # img_features: (-1, self.args.img_feature_dim)
                img_features = self._retrieve_img_features(
                    scan_ids[i], [current_viewpoint_viewpointIdx[i]], MAX_IMG_LENGTH
                )
                img_features = torch.from_numpy(img_features)

                if img_features.shape[0] > self.args.max_img_seq_length:
                    img_features = img_features[
                        -self.args.max_img_seq_length :,
                    ]
                    if self.args.max_img_seq_length > 0:
                        input_mask = input_mask + [1] * img_features.shape[0]
                        # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
                else:
                    if self.args.max_img_seq_length > 0:
                        input_mask = input_mask + [1] * img_features.shape[0]
                        # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                    padding_matrix = torch.zeros(
                        (
                            self.args.max_img_seq_length - img_features.shape[0],
                            img_features.shape[1],
                        ),
                        dtype=img_features.dtype,
                    )
                    img_features = torch.cat((img_features, padding_matrix), dim=0)
                    if self.args.max_img_seq_length > 0:
                        input_mask = input_mask + [0] * padding_matrix.shape[0]
                        # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

                input_mask = input_mask + [1]

                batch_input_ids.append(input_ids)
                batch_input_masks.append(input_mask)
                batch_segment_ids.append(segment_ids)
                batch_img_features.append(img_features.unsqueeze(0))

            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(
                self.args.device
            )
            batch_input_masks = torch.tensor(batch_input_masks, dtype=torch.long).to(
                self.args.device
            )
            batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long).to(
                self.args.device
            )
            batch_img_features = torch.cat(batch_img_features, dim=0).to(
                self.args.device
            )

            inputs = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_input_masks,
                "token_type_ids": batch_segment_ids,
                "img_feats": None
                if self.args.img_feature_dim == -1
                else batch_img_features,
                "action_feats": a_t,
            }

            outputs = self.model(**inputs)

            # loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            logits = outputs[0]

            # Mask outputs where agent can't move forward
            for i, ob in enumerate(obs):
                if len(ob["navigableLocations"]) <= 1:
                    logits[i, self.model_actions.index("forward")] = -float("inf")

            # Supervised training
            target = self._teacher_action(obs, ended, self.args.device)

            if return_for_next_action:
                loss = self.criterion(logits, target).item()
                correct = torch.sum(logits.argmax(1) == target).item()
                correct = float(correct) / self.dataloader.batch_size
                return (
                    traj,
                    loss,
                    correct,
                )

            current_loss = self.criterion(logits, target)
            if self.args.detach_loss:
                self.non_avg_loss += current_loss
            else:
                self.loss += current_loss

            # Determine next model inputs
            if self.feedback == "teacher":
                a_t = target  # teacher forcing
            elif self.feedback == "argmax":
                _, a_t = logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == "sample":
                probs = F.softmax(logits, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()  # sampling an action from model
            else:
                sys.exit("Invalid feedback option")

            # Updated "ended" list and make environment action
            for idx in range(batch_size):
                action_idx = a_t[idx].item()
                if action_idx == self.model_actions.index("<end>"):
                    ended[idx] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.dataloader.step(env_action))

            # Save trajectory output
            current_viewpoint_viewpointIdx = [
                (item["viewpoint"], item["viewIndex"]) for item in obs
            ]
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]["path"].append(
                        (ob["viewpoint"], ob["heading"], ob["elevation"])
                    )
            if self.args.detach_loss and train and self.episode_len >= 30:
                if (
                    t % self.args.detach_loss_at == self.args.detach_loss_at - 1
                    or t + 1 == self.episode_len
                    or ended.all()
                ):
                    # if (t%trunc)==(trunc-1) or t+1 == self.args.timesteps or ended.all():
                    # avg_loss = self.loss / 10.0
                    # avg_loss = self.loss
                    if self.args.n_gpu > 1:
                        pass  # already reduced
                    elif self.args.local_rank != -1:
                        self.non_avg_loss /= dist.get_world_size()
                        dist.all_reduce(self.non_avg_loss, op=dist.ReduceOp.SUM)
                        self.non_avg_loss /= self.args.detach_loss_at

                        self.loss += self.non_avg_loss
                        self.loss.backward()

                        self.loss.detach_()
                        self.non_avg_loss = torch.zeros(1).to(self.args.device)

            # Early exit if all ended
            if ended.all():
                break
        if self.args.detach_loss:
            self.loss = self.loss / (self.episode_len // self.args.detach_loss_at)
        else:
            self.loss = self.loss / self.episode_len

        self.losses.append(self.loss.item())
        return traj

    # TODO: fix test fn
    def test(self, use_dropout=False, feedback="argmax", allow_cheat=False):
        """ Evaluate once on each instruction in the current environment """
        if not allow_cheat:  # permitted for purpose of calculating validation loss only
            assert feedback in [
                "argmax",
                "sample",
            ]  # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.model.train()
        else:
            self.model.eval()
        super(OscarAgent, self).test()

    # TODO: fix test fn
    def test_only_next_action(self, use_dropout=False):
        """ Evaluate once on each instruction in the current environment """
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(OscarAgent, self).test_only_next_action()

    def train(self, optimizer, scheduler, n_iters, feedback="teacher"):
        """ Train for a given number of iterations """
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.model.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            optimizer.zero_grad()
            self.model.zero_grad()

            self.rollout()

            if not self.args.detach_loss:
                if self.args.local_rank != -1:
                    self.loss /= dist.get_world_size()
                    dist.all_reduce(self.loss, op=dist.ReduceOp.SUM)
                self.loss.backward()

            optimizer.step()
            scheduler.step()

    # def save(self, encoder_path, decoder_path):
    #     """ Snapshot models """
    #     torch.save(self.encoder.state_dict(), encoder_path)
    #     torch.save(self.decoder.state_dict(), decoder_path)

    # def load(self, encoder_path, decoder_path):
    #     """ Loads parameters (but not training state) """
    #     print("%s %s" % (encoder_path, decoder_path))
    #     self.encoder.load_state_dict(torch.load(encoder_path))
    #     self.decoder.load_state_dict(torch.load(decoder_path))
