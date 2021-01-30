""" Agents: stop/random/shortest/seq2seq  """

import copy
import json
import os
import random

# from torch import optim
import sys
from collections import OrderedDict

import numpy as np
import torch

# import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.autograd import Variable
from utils import copy_dialog_history, next_decoder_input, padding_idx
from utils_data import get_encoding_for_oscar

# from env import R2RBatch


# import sys
# import time


class BaseAgent(object):
    """ Base class for an R2R agent to generate and save trajectories. """

    def __init__(self, dataloader, results_path, turn_based=False):
        self.dataloader = dataloader
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.generated_dialog = {}
        self.turn_based = turn_based  # Whether to activate turn-based strategy

        self.losses = []  # For learning agents

    def write_results(self):
        the_dir, _ = os.path.split(self.results_path)
        if not os.path.isdir(the_dir):
            os.makedirs(the_dir)
        if len(self.generated_dialog) > 0:
            output = [
                {
                    "inst_idx": k,
                    "trajectory": v,
                    "num_instrs": int(len(self.generated_dialog[k]) / 2),
                }
                for k, v in self.results.items()
            ]
        else:
            output = [
                {"inst_idx": k, "trajectory": v, "num_instrs": 0}
                for k, v in self.results.items()
            ]
        with open(self.results_path, "w") as f:
            json.dump(output, f)

    def get_generated_dialog(self):
        return self.generated_dialog

    def rollout(self):
        """ Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  """
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self):
        self.dataloader.reset_epoch()
        self.losses = []
        self.results = {}
        self.generated_dialog = {}
        # We rely on env showing the entire batch before repeating anything
        # print 'Testing %s' % self.__class__.__name__
        looped = False
        while True:
            if self.turn_based:
                trajs, gen_dialogs = self.rollout(train=False)
                for i in range(len(trajs)):
                    traj = trajs[i]
                    gen_dialog = gen_dialogs[i]
                    if traj["inst_idx"] in self.results:
                        looped = True
                    else:
                        self.results[traj["inst_idx"]] = traj["path"]
                        self.generated_dialog[gen_dialog["inst_idx"]] = gen_dialog[
                            "generated_dialog"
                        ]
            else:
                for traj in self.rollout(train=False):
                    if traj["inst_idx"] in self.results:
                        looped = True
                    else:
                        self.results[traj["inst_idx"]] = traj["path"]
            if looped:
                break
            # break
        # for traj in self.rollout(train=False):
        #     if traj['inst_idx'] in self.results:
        #         looped = True
        #     else:
        #         self.results[traj['inst_idx']] = traj['path']


class Agent(BaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """

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
    env_actions = [
        (0, -1, 0),  # left
        (0, 1, 0),  # right
        (0, 0, 1),  # up
        (0, 0, -1),  # down
        (1, 0, 0),  # forward
        (0, 0, 0),  # <end>
        (0, 0, 0),  # <start>
        (0, 0, 0),  # <ignore>
    ]

    feedback_options = [
        "teacher",
        "argmax",
        "sample",
        "topk",
        "nucleus",
        "temperature",
        "penalty",
        "nucleus_with_penalty",
    ]

    def __init__(
        self,
        args,
        dataloader,
        results_path,
        encoder,
        encoder_optimizer,
        decoder,
        decoder_optimizer,
        train_episode_len=80,
        eval_episode_len=80,
        path_type="planner_path",
        turn_based=False,
        temperature=0.6,
        current_q_a_only=False,
        critic=None,
        critic_optimizer=None,
        gamma=0.9,
        use_rl=False,
        agent_rl=False,
        random_start=False,
        J=0,
        steps_to_next_q=5,
        train_branching=1,
        action_probs_branching=False,
        oscar_tokenizer=None,
    ):
        super(Agent, self).__init__(dataloader, results_path, turn_based)

        self.args = args

        self.oscar_tokenizer = oscar_tokenizer

        self.encoder = encoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder = decoder
        self.decoder_optimizer = decoder_optimizer
        self.current_q_a_only = current_q_a_only
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=args.ignoreid, reduction="none"
        )

        self.speaker = None
        self.temperature = temperature

        self.random_start = random_start
        self.J = J  # Jitter factor, 0 for no jitter
        self.steps_to_next_q = steps_to_next_q

        # self.dataloader_actions_instructions = [
        #     self.dataloader.tok.word_to_index("left"),  # left
        #     self.dataloader.tok.word_to_index("right"),  # right
        #     self.dataloader.tok.word_to_index("up"),  # up
        #     self.dataloader.tok.word_to_index("down"),  # down
        #     self.dataloader.tok.word_to_index("forward"),  # forward
        #     self.dataloader.tok.word_to_index("stop"),  # <end>
        #     self.dataloader.tok.word_to_index("start"),  # <start>
        #     self.dataloader.tok.word_to_index("<PAD>"),  # <ignore>
        # ]

        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

        self.use_rl = use_rl
        self.agent_rl = agent_rl

        self.train_episode_len = train_episode_len
        self.eval_episode_len = eval_episode_len

        # self.train_branching = train_branching
        # self.action_probs_branching = action_probs_branching

    @staticmethod
    def n_inputs():
        return len(Agent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Agent.model_actions) - 2  # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        """Extract instructions from a list of observations and sort by descending
        sequence length (to enable PyTorch packing)."""

        seq_tensor = np.array([ob["instr_encoding"] for ob in obs])
        segment_ids = np.array([ob["segment_ids"] for ob in obs])

        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        segment_ids = torch.from_numpy(segment_ids)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)

        sorted_tensor = seq_tensor[perm_idx]
        sorted_segment_ids = segment_ids[perm_idx]

        mask = (sorted_tensor == padding_idx)[:, : seq_lengths[0]]

        return (
            Variable(sorted_tensor, requires_grad=False).long().to(self.args.device),
            Variable(sorted_segment_ids, requires_grad=False)
            .long()
            .to(self.args.device),
            mask.byte().to(self.args.device),
            Variable(seq_lengths, requires_grad=False).long().to(self.args.device),
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

    def rollout(
        self,
        reset=True,
        extract_loss=False,
        train=True,
        speaker_branching=3,
        extract_distance=False,
        start_t=0,
        prev_all_env_action=None,
        prev_a_t=None,
        prev_traj=None,
        prev_loss=None,
        help_already_given=False,
        prev_ended=None,
        prev_obs=None,
        prev_dialog=None,
        used_perm_idx=None,
        prev_last_dist=None,
        prev_rewards=None,
        prev_hidden_states=None,
        prev_policy_log_probs=None,
        prev_masks=None,
        prev_entropys=None,
        prev_ml_loss=None,
        train_rl=True,
    ):

        # if start_t ==0:
        #     print start_t

        # Get obs at current position
        (
            obs,
            seq,
            seq_mask,
            seq_lengths,
            perm_idx,
            ctx,
            h_t,
            c_t,
            follower_distance,
            rl_loss,
        ) = (None, None, None, None, None, None, None, None, None, None)
        if not reset:
            if prev_all_env_action is not None:
                self.dataloader.reset(next_minibatch=False)
                for env_act in prev_all_env_action:
                    self.dataloader.step(env_act)
            if prev_obs is not None:
                obs = np.array(copy_dialog_history(prev_obs))
                perm_obs = obs
                perm_idx = copy.deepcopy(used_perm_idx)
            else:
                obs = np.array(self.dataloader._get_obs())
                # Reorder the language input for the encoder
                new_obs = get_encoding_for_oscar(self.oscar_tokenizer, obs)
                seq, segment_ids, seq_mask, seq_lengths, perm_idx = self._sort_batch(
                    new_obs
                )
                perm_obs = obs[perm_idx]
                # Forward through encoder, giving initial hidden state and memory cell for decoder
                ctx, h_t, c_t = self.encoder(
                    inputs=seq,
                    lengths=seq_lengths,
                    mask=seq_mask,
                    token_type_ids=segment_ids,
                )
        else:
            obs = np.array(self.dataloader.reset())
            new_obs = get_encoding_for_oscar(self.oscar_tokenizer, obs)
            seq, segment_ids, seq_mask, seq_lengths, perm_idx = self._sort_batch(
                new_obs
            )
            perm_obs = obs[perm_idx]
            ctx, h_t, c_t = self.encoder(
                inputs=seq,
                lengths=seq_lengths,
                mask=seq_mask,
                token_type_ids=segment_ids,
            )
        batch_size = len(obs)

        # traj = [{
        #     'inst_idx': perm_obs[i]['inst_idx'],
        #     'path': [(perm_obs[i]['viewpoint'], perm_obs[i]['heading'], perm_obs[i]['elevation'])]
        # } for i in range(batch_size)]

        traj = []
        if prev_traj is None:
            # Record starting point
            for ob in obs:
                traj.append(
                    {
                        "inst_idx": ob["inst_idx"],
                        "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
                    }
                )
        else:
            traj = copy.deepcopy(prev_traj)

        # For test result submission
        visited = [set() for _ in perm_obs]

        gen_dialog = []
        if prev_dialog is None:
            for ob in obs:
                gen_dialog.append({"inst_idx": ob["inst_idx"], "generated_dialog": {}})
        else:
            gen_dialog = copy.deepcopy(prev_dialog)
            # print gen_dialog

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        if prev_last_dist is None:
            for i, ob in enumerate(
                obs
            ):  # The init distance from the view point to the target
                last_dist[i] = ob["distance"]

        # Init the logs
        rewards = [] if prev_rewards is None else copy.deepcopy(prev_rewards)
        hidden_states = (
            []
            if prev_hidden_states is None
            else [x.clone() for x in prev_hidden_states]
        )
        policy_log_probs = (
            []
            if prev_policy_log_probs is None
            else [x.clone() for x in prev_policy_log_probs]
        )
        masks = [] if prev_masks is None else copy.deepcopy(prev_masks)
        entropys = [] if prev_entropys is None else [x.clone() for x in prev_entropys]
        # ml_loss = 0. if prev_ml_loss is None else copy.deepcopy(prev_ml_loss)

        # Initial action
        input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
        if prev_a_t is not None:
            input_a_t = prev_a_t.clone()
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env
        if prev_ended is not None:
            ended = np.copy(prev_ended)

        # Do a sequence rollout and calculate the loss
        loss = torch.zeros(len(obs)).to(self.args.device)
        if prev_loss is not None:
            loss = prev_loss.clone()
        all_env_action, env_action = [], [None] * batch_size
        # agent_loss, new_traj = None, None
        episode_len = self.train_episode_len if train else self.eval_episode_len
        for t in range(start_t, episode_len):
            if self.turn_based:
                help_needed = False
                resort_needed = False
                help_requesters = set([])
                if not help_already_given:
                    for i, ob in enumerate(perm_obs):
                        item = self.dataloader.batch[perm_idx[i]]
                        time_step = t - self.J
                        if train and time_step in item["request_locations"]:
                            help_needed = True
                            nav_ins, ora_ins, end_index = item["request_locations"][
                                time_step
                            ]
                            # real = [m['message'] for m in item['dialog_history'][:end_index][-2:]]
                            item["nav_instructions"] = nav_ins
                            item[
                                "nav_instr_encoding"
                            ] = self.dataloader.dataset.tok.encode_sentence(
                                item["nav_instructions"], seps="<NAV>"
                            )
                            item["ora_instructions"] = ora_ins
                            item[
                                "ora_instr_encoding"
                            ] = self.dataloader.dataset.tok.encode_sentence(
                                item["ora_instructions"], seps="<ORA>"
                            )
                            help_requesters.add(i)
                        elif not train and time_step % self.steps_to_next_q == 0:
                            help_needed = True
                            help_requesters.add(i)
                else:
                    resort_needed = True
                    help_needed = True

                if help_needed:
                    # print("Episode %d" % t)
                    # nav_ora_generated = []
                    # if self.speaker is not None and self.feedback == 'argmax':
                    self.speaker.dataloader = self.dataloader
                    # print("Generating QA at %d" % t)
                    if not help_already_given:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        all_act = (
                            all_env_action
                            if prev_all_env_action is None
                            else prev_all_env_action + all_env_action
                        )
                        if train:
                            # if speaker_branching >1:
                            #     print("Train forked at %d" % t)
                            (
                                follower_distance,
                                loss,
                                rl_loss,
                                traj,
                                gen_dialog,
                            ) = self.speaker.train(
                                1,
                                do_reset=False,
                                train_nav=True,
                                return_predict=True,
                                k=speaker_branching,
                                current_t=t,
                                prev_act=all_act,
                                prev_a_t=None,
                                prev_traj=traj,
                                prev_loss=loss,
                                prev_ended=prev_ended,
                                prev_obs=perm_obs,
                                prev_dialog=gen_dialog,
                                perm_idx=perm_idx,
                                help_requesters=help_requesters,
                                prev_last_dist=last_dist,
                                prev_rewards=rewards,
                                prev_hidden_states=hidden_states,
                                prev_policy_log_probs=policy_log_probs,
                                prev_masks=masks,
                                prev_entropys=entropys,
                                prev_ml_loss=loss,
                                train_rl=train_rl,
                            )
                        else:
                            # if speaker_branching >1:
                            #     print("Eval forked at %d" % t)
                            (
                                follower_distance,
                                _,
                                loss,
                                rl_loss,
                                traj,
                                gen_dialog,
                            ) = self.speaker.teacher_forcing(
                                train=False,
                                for_nav=True,
                                eval=False,
                                k=speaker_branching,
                                current_t=t,
                                prev_act=all_act,
                                prev_a_t=None,
                                prev_traj=traj,
                                prev_loss=loss,
                                prev_ended=prev_ended,
                                prev_obs=perm_obs,
                                prev_dialog=gen_dialog,
                                perm_idx=perm_idx,
                                help_requesters=help_requesters,
                                prev_last_dist=last_dist,
                                prev_rewards=rewards,
                                prev_hidden_states=hidden_states,
                                prev_policy_log_probs=policy_log_probs,
                                prev_masks=masks,
                                prev_entropys=entropys,
                                prev_ml_loss=loss,
                                train_rl=train_rl,
                            )
                        break
                        # if new_traj is not None:
                        #     loss = agent_loss
                        #     traj = new_traj
                        #     torch.cuda.empty_cache()
                        #     # print "break!"
                        #     break
                        # self.dataloader.reset(next_minibatch=False)
                        # for env_act in all_act:
                        #     self.dataloader.step(env_act)
                    # for i,insts in enumerate([nav_insts, ora_insts]):
                    #     insts_list = [insts[idx] for idx in perm_idx]
                    #     generated_turn = []
                    #     for i,inst in enumerate(insts_list):
                    #         instruction = self.dataloader.tok.decode_sentence( self.dataloader.tok.shrink(inst) )
                    #         role = 'navigator' if i==0 else 'oracle'
                    #         generated_turn.append({'role':role, 'message': instruction})
                    #     nav_ora_generated.append(generated_turn)

                    for i, ob in enumerate(perm_obs):
                        item = self.dataloader.batch[perm_idx[i]]
                        time_step = t - self.J
                        ask_spot = (
                            not train
                            and self.steps_to_next_q >= 0
                            and time_step % self.steps_to_next_q == 0
                        )
                        if (
                            time_step in item["request_locations"]
                            and self.steps_to_next_q < 0
                            or ask_spot
                        ):
                            # d_history = []
                            # # if self.speaker is not None and self.feedback == 'argmax':
                            # # print ("GENERATED:")
                            # generated = []
                            # for generated_turn in nav_ora_generated:
                            #     ob['generated_dialog_history'].append(generated_turn[i])
                            #     generated.append(generated_turn[i]['message'])
                            # # for g in generated:
                            # #     print(g)
                            # # print ("REAL:")
                            # nav_ins, ora_ins, end_index = item['request_locations'][t]
                            # real = [m['message'] for m in item['dialog_history'][:end_index][-2:]]
                            # # for r in real:
                            # #     print(r)
                            # # print(" ")
                            # print ob['generated_dialog_history']
                            d_history = ob["generated_dialog_history"]
                            if self.current_q_a_only:
                                d_history = d_history[-2:]
                            dia_inst = ""
                            sentences = []
                            seps = []
                            for turn in d_history:
                                sentences.append(turn["message"])
                                sep = (
                                    "<NAV>" if turn["role"] == "navigator" else "<ORA>"
                                )
                                seps.append(sep)
                                dia_inst += sep + " " + turn["message"] + " "
                            sentences.append(item["target"])
                            seps.append("<TAR>")
                            dia_inst += "<TAR> " + item["target"]
                            ob["instructions"] = dia_inst
                            dia_enc = self.dataloader.dataset.tok.encode_sentence(
                                sentences, seps=seps
                            )
                            ob["instr_encoding"] = dia_enc
                            resort_needed = True
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if resort_needed:
                    prev_perm_idx = perm_idx
                    new_obs = get_encoding_for_oscar(self.oscar_tokenizer, perm_obs)
                    (
                        seq,
                        segment_ids,
                        seq_mask,
                        seq_lengths,
                        perm_idx,
                    ) = self._sort_batch(new_obs)
                    ctx, h_t, c_t = self.encoder(
                        inputs=seq,
                        lengths=seq_lengths,
                        mask=seq_mask,
                        token_type_ids=segment_ids,
                    )
                    perm_obs = perm_obs[perm_idx]
                    # traj = [traj[id] for id in perm_idx]
                    new_a_t = Variable(
                        torch.ones(batch_size).long()
                        * self.model_actions.index("<start>"),
                        requires_grad=False,
                    ).to(self.args.device)
                    ended = np.array([ended[id] for id in perm_idx])
                    new_perm_idx = [x for x in perm_idx]
                    for i, idx in enumerate(perm_idx):
                        new_perm_idx[i] = prev_perm_idx[idx]
                        new_a_t[i] = input_a_t[idx]
                    perm_idx = new_perm_idx
                    input_a_t = new_a_t

            f_t = self._feature_variable(perm_obs)  # Image features from obs
            h_t, c_t, nav_logit, qa_logit = self.decoder(
                input_a_t,
                f_t,
                candidate_feat,
                h_t,
                c_t,
                ctx,
                ctx_mask,
            )

            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            if self.args.submit:  # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob["viewpoint"])
                    for c_id, c in enumerate(ob["candidate"]):
                        if c["viewpointId"] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            nav_logit.masked_fill_(candidate_mask, -float("inf"))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            loss += self.criterion(nav_logit, target)

            # Determine next model inputs
            if self.feedback == "teacher":
                a_t = target  # teacher forcing
            elif self.feedback == "argmax":
                _, a_t = nav_logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(nav_logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(
                    log_probs.gather(1, a_t.unsqueeze(1))
                )  # Gather the log_prob for each batch
            elif self.feedback == "sample":
                probs = F.softmax(nav_logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs["entropy"].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit("Invalid feedback option")

            if self.critic is not None:
                probs, log_probs = F.softmax(nav_logit, 1), F.log_softmax(
                    nav_logit, 1
                )  # Calculate the log_prob here
                policy_log_probs.append(
                    log_probs.gather(1, a_t.unsqueeze(1))
                )  # Gather the log_prob for each batch
                c = torch.distributions.Categorical(
                    probs
                )  # sampling an action from model
                entropys.append(c.entropy())  # For optimization

            # Updated 'ended' list and make environment action
            for i, idx in enumerate(perm_idx):
                # if train:
                #     action_idx = target[i].item()
                # else:
                #
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index("<end>"):
                    ended[i] = True
                if (
                    action_idx == self.model_actions.index("forward")
                    and not perm_obs[i]["candidate"][action_idx]["navigable"]
                ):  # len(perm_obs[i]['navigableLocations']) <= 1:
                    action_idx = self.model_actions.index("<ignore>")
                    print(perm_obs[i]["candidate"][action_idx])
                env_action[idx] = self.dataloader_actions[action_idx]
            all_env_action.append(list(env_action))

            # print "Target"
            # print target
            # print "a_t"
            # print env_action

            obs = np.array(self.dataloader.step(env_action))
            if self.turn_based:
                prev_perm_obs = [x for x in perm_obs]
                perm_obs = obs[perm_idx]  # Perm the obs for the resu
                for i, ob in enumerate(prev_perm_obs):
                    perm_obs[i]["instructions"] = ob["instructions"]
                    perm_obs[i]["instr_encoding"] = ob["instr_encoding"]
                    perm_obs[i]["generated_dialog_history"] = ob[
                        "generated_dialog_history"
                    ]
                    perm_obs[i]["action_probs"] = ob["action_probs"]
            else:
                perm_obs = obs[perm_idx]

            # Save the last trajectory output
            # obs_to_perm_idx = {idx:i for i,idx in enumerate(perm_idx) }
            for i, idx in enumerate(perm_idx):
                if not ended[i]:
                    ob = perm_obs[i]
                    traj[idx]["path"].append(
                        (ob["viewpoint"], ob["heading"], ob["elevation"])
                    )
                    gen_dialog[idx]["generated_dialog"] = ob["generated_dialog_history"]
                    # traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            cpu_a_t = a_t.cpu().numpy()
            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(obs):
                dist[i] = ob["distance"]
                if ended[i]:  # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.0
                    mask[i] = 0.0
                else:  # Calculate the reward
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:  # If the action now is end
                        if dist[i] < 3:  # Correct
                            reward[i] = 2.0
                        else:  # Incorrect
                            reward[i] = -2.0
                    else:  # The action is not end
                        reward[i] = -(dist[i] - last_dist[i])  # Change of distance
                        if reward[i] > 0:  # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        else:
                            # raise NameError("The action doesn't change the move")
                            pass
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            # Early exit if all ended
            if ended.all():
                break
            help_already_given = False

        if self.use_rl and rl_loss is None:
            rl_loss = 0.0
            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(
                h_t
            ).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(
                batch_size, np.float32
            )  # The inital reward is zero
            for i in range(batch_size):
                if not ended[
                    i
                ]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = (
                    discount_reward * self.gamma + rewards[t]
                )  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).to(
                    self.args.device
                )
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(
                    self.args.device
                )
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback != "teacher" or self.feedback != "argmax":
                    rl_loss += (-0.01 * entropys[t] * mask_).sum()

                total = total + np.sum(masks[t])

            # Normalize the loss function
            rl_loss /= total  # rl_loss /= batch_size
            rl_loss = Variable(rl_loss, requires_grad=False).to(self.args.device)

        if extract_loss:
            return loss  # .item() * 1.0 / self.episode_len
        elif extract_distance:
            if follower_distance is None:
                final_distances = []
                for ob in obs:
                    branching_metric = (
                        np.mean(ob["action_probs"])
                        if self.action_probs_branching
                        else ob["distance"]
                    )
                    final_distances.append(branching_metric)
                return (
                    np.array(final_distances).reshape((-1, 1)),
                    loss,
                    rl_loss,
                    traj,
                    gen_dialog,
                )
            return follower_distance, loss, rl_loss, traj, gen_dialog

        if self.agent_rl:
            if self.feedback != "teacher" and self.feedback != "argmax":
                loss += rl_loss
        self.loss = loss.float().mean()
        self.losses.append(self.loss.item() / episode_len)

        if self.turn_based:
            # print gen_dialog
            return traj, gen_dialog
        else:
            return traj

    def test(self, use_dropout=False, feedback="argmax", allow_cheat=False):
        """ Evaluate once on each instruction in the current environment """
        if not allow_cheat:  # permitted for purpose of calculating validation loss only
            assert feedback != "teacher"  # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            if self.use_rl:
                self.critic.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            if self.use_rl:
                self.critic.eval()
        super(Agent, self).test()

    def train(self, n_iters, feedback="teacher"):
        """ Train for a given number of iterations """
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        if self.use_rl:
            self.critic.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            if self.use_rl:
                self.critic_optimizer.zero_grad()
            self.rollout(speaker_branching=1)
            self.loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            if self.use_rl:
                self.critic_optimizer.step()
            if self.random_start:
                losses = [x for x in self.losses]
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                viewpointIds = self.dataloader.random_start(self.J)
                self.rollout(reset=False)
                self.dataloader.reset_viewpointIds(viewpointIds)
                self.loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.losses = losses

    def save(self, encoder_path, decoder_path):
        """ Snapshot models """
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        """ Loads parameters (but not training state) """
        # self.encoder.load_state_dict(torch.load(encoder_path))
        # self.decoder.load_state_dict(torch.load(decoder_path))

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
