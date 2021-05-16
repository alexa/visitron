# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.optim import Adam

import model
import utils


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


class Agent(BaseAgent):
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
        tokenizer,
        dataloader,
        results_path,
        vocab=None,
        bert=None,
        word_embedding=None,
        episode_len=20,
    ):
        super(Agent, self).__init__(dataloader, results_path)
        self.args = args
        self.tokenizer = tokenizer

        self.pad_token_id = 0
        # Models

        self.encoder = model.OscarEncoder(
            args=args,
            bert=bert,
            hidden_size=args.encoder_hidden_size,
            decoder_hidden_size=args.rnn_dim,
            dropout_ratio=args.dropout,
            bidirectional=args.bidir,
        ).to(args.device)

        self.decoder = model.AttnDecoderLSTM(
            args.angle_feat_size,
            args.aemb,
            args.rnn_dim,
            args.dropout,
            feature_size=self.args.lstm_img_feature_dim + args.angle_feat_size,
        ).to(args.device)
        self.models = (self.encoder, self.decoder)

        # Optimizers
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr=args.learning_rate)
        self.decoder_optimizer = Adam(self.decoder.parameters(), lr=args.learning_rate)
        self.optimizers = (
            self.encoder_optimizer,
            self.decoder_optimizer,
        )

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid)

        self.episode_len = episode_len
        self.losses = []

    @staticmethod
    def n_inputs():
        return len(Agent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Agent.model_actions) - 2  # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        """Extract instructions from a list of observations and sort by descending
        sequence length (to enable PyTorch packing)."""

        seq_tensor = np.array([ob["target_dialog_tokens_id"] for ob in obs])
        # if not self.args.no_pretrained_model:
        segment_ids = np.array([ob["target_dialog_segment_ids"] for ob in obs])

        seq_lengths = np.argmax(seq_tensor == self.pad_token_id, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        # if not self.args.no_pretrained_model:
        segment_ids = torch.from_numpy(segment_ids)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        # if not self.args.no_pretrained_model:
        sorted_segment_ids = segment_ids[perm_idx]
        mask = (sorted_tensor == self.pad_token_id)[
            :, : seq_lengths[0]
        ]  # seq_lengths[0] is the Maximum length

        return (
            Variable(sorted_tensor, requires_grad=False).long().to(self.args.device),
            Variable(sorted_segment_ids, requires_grad=False)
            .long()
            .to(self.args.device),
            mask.byte().to(self.args.device),
            list(seq_lengths),
            list(perm_idx),
        )

    def _feature_variable(self, obs):
        """ Extract precomputed features into variable. """
        features = np.empty(
            (
                len(obs),
                self.args.views,
                self.args.lstm_img_feature_dim + self.args.angle_feat_size,
            ),
            dtype=np.float32,
        )
        for i, ob in enumerate(obs):
            features[i, :, :] = ob["feature"]  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).to(
            self.args.device
        )

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob["candidate"]) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros(
            (
                len(obs),
                max(candidate_leng),
                self.args.lstm_img_feature_dim + self.args.angle_feat_size,
            ),
            dtype=np.float32,
        )
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob["candidate"]):
                candidate_feat[i, j, :] = c["feature"]  # Image feat
        return torch.from_numpy(candidate_feat).to(self.args.device), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob["heading"], ob["elevation"])
        input_a_t = torch.from_numpy(input_a_t).to(self.args.device)

        f_t = self._feature_variable(obs)  # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
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
        return torch.from_numpy(a).to(self.args.device)

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

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        def take_action(i, idx, name):
            if type(name) is int:  # Go to the next view
                self.dataloader.env.makeActionsatIndex((name, 0, 0), idx)
            else:  # Adjust
                self.dataloader.env.makeActionsatIndex(self.env_actions[name], idx)
            state = self.dataloader.env.getStates()[idx][1]
            if traj is not None:
                traj[i]["path"].append(
                    (state.location.viewpointId, state.heading, state.elevation)
                )

        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                select_candidate = perm_obs[i]["candidate"][action]
                src_point = perm_obs[i]["viewIndex"]
                trg_point = select_candidate["pointId"]
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:  # Tune up
                    take_action(i, idx, "up")
                    src_level += 1
                while src_level > trg_level:  # Tune down
                    take_action(i, idx, "down")
                    src_level -= 1
                while (
                    self.dataloader.env.getStates()[idx][1].viewIndex != trg_point
                ):  # Turn right until the target
                    take_action(i, idx, "right")
                assert (
                    select_candidate["viewpointId"]
                    == self.dataloader.env.getStates()[idx][1]
                    .navigableLocations[select_candidate["idx"]]
                    .viewpointId
                )
                take_action(i, idx, select_candidate["idx"])

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob["heading"], ob["elevation"])
        input_a_t = torch.from_numpy(input_a_t).to(self.args.device)

        f_t = self._feature_variable(obs)  # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def rollout(self, train=True):

        batch = self._get_batch()
        batch = self._verify_batch_size(batch)

        scan_ids = [item["scan"] for item in batch]

        obs = np.array(self.dataloader.reset())

        batch_size = len(obs)

        # Reorder the language input for the encoder (do not ruin the original code)
        # if self.args.no_pretrained_model:
        #     seq, _, seq_mask, seq_lengths, perm_idx = self._sort_batch(batch)
        # else:
        seq, segment_ids, seq_mask, seq_lengths, perm_idx = self._sort_batch(batch)

        seq_lengths = torch.tensor(seq_lengths)
        perm_obs = obs[perm_idx]

        # if self.args.no_pretrained_model:
        #     ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        # else:
        ctx, h_t, c_t = self.encoder(
            inputs=seq,
            lengths=seq_lengths,
            mask=seq_mask,
            token_type_ids=segment_ids,
        )
        ctx_mask = seq_mask

        # Record starting point
        traj = [
            {
                "inst_idx": ob["inst_idx"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in perm_obs
        ]

        # For test result submission
        visited = [set() for _ in obs]

        ended = np.array([False] * batch_size)

        # Do a sequence rollout and calculate the loss
        self.loss = torch.zeros(1).to(self.args.device)
        if self.args.detach_loss:
            self.non_avg_loss = torch.zeros(1).to(self.args.device)

        h1 = h_t

        for t in range(self.episode_len):
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(
                perm_obs
            )
            h_t, c_t, logit, h1 = self.decoder(
                input_a_t,
                f_t,
                candidate_feat,
                h_t,
                h1,
                c_t,
                ctx,
                ctx_mask,
            )

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng, self.args.device)
            if self.args.submit:  # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob["viewpoint"])
                    for c_id, c in enumerate(ob["candidate"]):
                        if c["viewpointId"] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask, -float("inf"))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)

            current_loss = self.criterion(logit, target)
            if self.args.detach_loss:
                self.non_avg_loss += current_loss
            else:
                self.loss += current_loss

            # Determine next model inputs
            if self.feedback == "teacher":
                a_t = target  # teacher forcing
            elif self.feedback == "argmax":
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == "sample":
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()  # sampling an action from model
            else:
                sys.exit("Invalid feedback option")

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == self.args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.dataloader._get_obs())
            perm_obs = obs[perm_idx]

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

    def test(self, use_dropout=False, feedback="argmax", allow_cheat=False):
        """ Evaluate once on each instruction in the current environment """
        if not allow_cheat:  # permitted for purpose of calculating validation loss only
            assert feedback in [
                "argmax",
                "sample",
            ]  # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Agent, self).test()

    def train(self, n_iters, feedback="teacher"):
        """ Train for a given number of iterations """
        assert feedback in self.feedback_options

        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()

        self.losses = []
        for iter in range(1, n_iters + 1):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            self.rollout()

            if not self.args.detach_loss:
                if self.args.local_rank not in [-2, -1]:
                    self.loss /= dist.get_world_size()
                    dist.all_reduce(self.loss, op=dist.ReduceOp.SUM)
                self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.0)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.0)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        """ Snapshot models """
        encoder_weights = (
            self.encoder.state_dict().module
            if hasattr(self.encoder.state_dict(), "module")
            else self.encoder.state_dict()
        )
        decoder_weights = (
            self.decoder.state_dict().module
            if hasattr(self.decoder.state_dict(), "module")
            else self.decoder.state_dict()
        )
        torch.save(encoder_weights, encoder_path)
        torch.save(decoder_weights, decoder_path)

    def load(self, encoder_path, decoder_path):
        """ Loads parameters (but not training state) """
        print("%s %s" % (encoder_path, decoder_path))
        encoder_weights = torch.load(encoder_path)
        decoder_weights = torch.load(decoder_path)

        encoder_has_module = all(
            [True for key, value in encoder_weights.items() if "module" in key.lower()]
        )
        decoder_has_module = all(
            [True for key, value in decoder_weights.items() if "module" in key.lower()]
        )
        if encoder_has_module:
            new_encoder_weights = OrderedDict()
            for k, v in encoder_weights.items():
                name = k[7:]  # remove `module.`
                new_encoder_weights[name] = v
        else:
            new_encoder_weights = encoder_weights

        if decoder_has_module:
            new_decoder_weights = OrderedDict()
            for k, v in decoder_weights.items():
                name = k[7:]  # remove `module.`
                new_decoder_weights[name] = v
        else:
            new_decoder_weights = decoder_weights

        self.encoder.load_state_dict(new_encoder_weights)
        self.decoder.load_state_dict(new_decoder_weights)
        # self.decoder.load_state_dict(decoder_weights)
        # self.encoder.load_state_dict(encoder_weights)
