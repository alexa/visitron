# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os
import sys
import numpy as np
import random
import time
import utils

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
    env_actions = {
      "left":     (0, -1,  0),  # left
      "right":    (0,  1,  0),  # right
      "up":       (0,  0,  1),  # up
      "down":     (0,  0, -1),  # down
      "forward":  (1,  0,  0),  # forward
      "<end>":    (0,  0,  0),  # <end>
      "<start>":  (0,  0,  0),  # <start>
      "<ignore>": (0,  0,  0)   # <ignore>
    }
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

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid)

        self.use_oscar_settings = use_oscar_settings

    @staticmethod
    def n_inputs():
        return len(OscarAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(OscarAgent.model_actions) - 2  # Model doesn't output start or ignore

    def _teacher_action(self, obs, ended, device):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob["candidate"]):
                    if candidate["viewpointId"] == ob["teacher"]:  # Next view point
                        a[i] = k
                        break
                else:  # Stop here
                    assert (
                        ob["teacher"] == ob["viewpoint"]
                    )  # The teacher action should be "STAY HERE"
                    a[i] = len(ob["candidate"])
        return torch.from_numpy(a).to(device)

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

    def _preprocess_region_label_tokens(self, region_labels, MAX_REGION_LABELS_LENGTH):
        region_labels = set(region_labels)
        region_labels = " ".join(region_labels)
        token_region_labels = self.dataloader.tokenizer.tokenize(region_labels)
        token_region_labels = token_region_labels[-MAX_REGION_LABELS_LENGTH:]
        return token_region_labels

    def _preprocess_img_features(self, features, MAX_IMG_FEATURES_LENGTH):
        features = features[-MAX_IMG_FEATURES_LENGTH:]
        return features

    def _get_candidate_features(self, obs):
        candidate_leng = [len(ob["candidate"]) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros(
            (
                len(obs),
                max(candidate_leng),
                2048
                # + self.args.angle_feat_size,  # TODO: add this to args
            ),
            dtype=np.float32,
        )
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob["candidate"]):
                candidate_feat[i, j, :] = c["feature"]  # Image feat
        return torch.from_numpy(candidate_feat).to(self.args.device), candidate_leng

    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        def take_action(idx, name):
            if type(name) is int:  # Go to the next view
                self.dataloader.env.makeActionsatIndex((name, 0, 0), idx)
            else:  # Adjust
                self.dataloader.env.makeActionsatIndex(self.env_actions[name], idx)
            state = self.dataloader.env.getStates()[idx][1]
            if traj is not None:
                traj[idx]["path"].append(
                    (state.location.viewpointId, state.heading, state.elevation)
                )

        for idx in range(len(obs)):
            action = a_t[idx]
            if action != -1:  # -1 is the <stop> action
                select_candidate = obs[idx]["candidate"][action]
                src_point = obs[idx]["viewIndex"]
                trg_point = select_candidate["pointId"]
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:  # Tune up
                    take_action(idx, "up")
                    src_level += 1
                while src_level > trg_level:  # Tune down
                    take_action(idx, "down")
                    src_level -= 1
                while (
                    self.dataloader.env.getStates()[idx][1].viewIndex != trg_point
                ):  # Turn right until the target
                    take_action(idx, "right")
                assert (
                    select_candidate["viewpointId"]
                    == self.dataloader.env.getStates()[idx][1]
                    .navigableLocations[select_candidate["idx"]]
                    .viewpointId
                )
                take_action(idx, select_candidate["idx"])

    def get_input_action(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob["heading"], ob["elevation"])
        input_a_t = torch.from_numpy(input_a_t).to(self.args.device)
        return input_a_t

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
            MAX_ACTION_LEN = 128 - 1  # -1 for [SEP] after Action Stream
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

        # For test result submission
        visited = [set() for _ in obs]

        ended = np.array([False] * batch_size)

        # Do a sequence rollout and calculate the loss
        self.loss = torch.zeros(1).to(self.args.device)
        if self.args.detach_loss:
            self.non_avg_loss = torch.zeros(1).to(self.args.device)

        for t in range(self.episode_len):

            batch_input_a_t = self.get_input_action(obs)

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
                    region_labels_tokens = self._preprocess_region_label_tokens(
                        obs[i]["region_labels"], MAX_REGION_LABEL_LEN
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
                img_features = self._preprocess_img_features(
                    obs[i]["feature"], MAX_IMG_LENGTH
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

            batch_candidate_feats, candidate_lengths = self._get_candidate_features(obs)

            inputs = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_input_masks,
                "token_type_ids": batch_segment_ids,
                "img_feats": None
                if self.args.img_feature_dim == -1
                else batch_img_features,
                "candidate_feats": batch_candidate_feats,
                "action_feats": batch_input_a_t,
            }

            outputs = self.model(**inputs)

            # loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            logits = outputs[0]

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_lengths, self.args.device)
            if self.args.submit:  # Avoiding cyclic path
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob["viewpoint"])
                    for c_id, c in enumerate(ob["candidate"]):
                        if c["viewpointId"] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logits.masked_fill_(candidate_mask, -float("inf"))

            # Supervised training
            target = self._teacher_action(obs, ended, self.args.device)

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

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_lengths[i] - 1)
                    or next_id == self.args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = np.array(self.dataloader._get_obs())

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

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
