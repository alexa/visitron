import base64
import json
import logging
import math
import random
import sys

import MatterSim
import networkx as nx
import numpy as np
import torch
import utils
from oscar_model import special_tokens_dict
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils_data import (
    load_classifier_data,
    load_datasets,
    load_nav_graphs,
    truncate_dialogs,
)

logger = logging.getLogger(__name__)


def ClassifierDataloader_collate_fn(batch):
    return batch


class EnvBatch:
    """A simple wrapper for a batch of MatterSim environments,
    using discretized viewpoints and pretrained features"""

    def __init__(self, feature_store, batch_size):

        if feature_store is None:
            self.features = None
            self.image_w = 600
            self.image_h = 600
            self.vfov = 80
        else:
            self.features = feature_store["features"]
            self.image_w = feature_store["image_w"]
            self.image_h = feature_store["image_h"]
            self.vfov = feature_store["vfov"]

        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

    def _make_id(self, scanId, viewpointId):
        return scanId + "_" + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        self.sim.newEpisode(scanIds, viewpointIds, headings, [0] * self.batch_size)

    def getStates(self):
        """ Get list of states augmented with precomputed image features. rgb field will be empty. """
        feature_states = []
        for state in self.sim.getState():
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        """Take an action using the full state dependent action interface (with batched input).
        Every action element should be an (index, heading, elevation) tuple."""
        ix = []
        heading = []
        elevation = []
        for i, h, e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

    def makeActionsatIndex(self, action, index):
        no_action = (0, 0, 0)
        ix = []
        heading = []
        elevation = []
        for i in range(self.batch_size):
            if i == index:
                ix.append(int(action[0]))
                heading.append(float(action[1]))
                elevation.append(float(action[2]))
            else:
                ix.append(no_action[0])
                heading.append(no_action[1])
                elevation.append(no_action[2])

        self.sim.makeAction(ix, heading, elevation)


class ClassifierDataset(Dataset):
    def __init__(
        self,
        args,
        splits=["train"],
        tokenizer=None,
        truncate_dialog=False,
    ):
        super(ClassifierDataset, self).__init__()

        assert tokenizer is not None

        self.args = args
        self.data = []
        self.scans = []

        use_oscar_settings = args.oscar_setting

        TAR_BACK = args.tar_back

        pad_token_id = 0

        cls_token_segment_id = 0
        pad_token_segment_id = 0
        sep_token_segment_id = 0

        tar_token_segment_id = 1
        ques_token_segment_id = 2
        ans_token_segment_id = 3

        MAX_SEQ_LENGTH = 512
        MAX_DIALOG_LEN = 512 - 4  # including [QUES]s and [ANS]s
        MAX_TARGET_LENGTH = 4 - 2  # [CLS], [TAR], [SEP] after QA and before Action

        ratios = []

        for item in load_classifier_data(splits):
            self.scans.append(item["scan"])

            new_item = dict(item)
            new_item["inst_idx"] = item["inst_idx"]

            token_target = tokenizer.tokenize(item["target"])
            token_target = token_target[:MAX_TARGET_LENGTH]
            new_item["token_target"] = token_target

            new_item["language"] = {}

            dialog_history = item["dialog_history"]
            for timestep, dialog in dialog_history.items():
                token_dialog_history = []
                for turn in dialog:
                    token_turn = tokenizer.tokenize(turn)
                    token_dialog_history.append(token_turn)
                if truncate_dialog:
                    # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                    token_dialog_history = truncate_dialogs(
                        token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                    )

                tokens = [tokenizer.cls_token]
                segment_ids = [cls_token_segment_id]

                if not TAR_BACK:
                    if use_oscar_settings:
                        sep_token = tokenizer.sep_token
                    else:
                        sep_token = tokenizer.tar_token

                    tokens += [sep_token] + token_target
                    segment_ids += [tar_token_segment_id] * (len(token_target) + 1)

                zeroth_time_tokens = tokens.copy()
                zeroth_time_segment_ids = segment_ids.copy()

                for i, turn in enumerate(token_dialog_history):
                    if use_oscar_settings:
                        sep_token = tokenizer.sep_token
                        segment_id = sep_token_segment_id
                    else:
                        if i % 2 == 0:
                            sep_token = tokenizer.ques_token
                            segment_id = ques_token_segment_id
                        else:
                            sep_token = tokenizer.ans_token
                            segment_id = ans_token_segment_id

                    tokens += [sep_token] + turn
                    segment_ids += [segment_id] * (len(turn) + 1)

                if TAR_BACK:
                    if use_oscar_settings:
                        sep_token = tokenizer.sep_token
                    else:
                        sep_token = tokenizer.tar_token

                    tokens += [sep_token] + token_target
                    segment_ids += [tar_token_segment_id] * (len(token_target) + 1)

                tokens += [tokenizer.sep_token]
                segment_ids += [sep_token_segment_id]

                tokens += [pad_token_id] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                segment_ids += [pad_token_segment_id] * (
                    MAX_SEQ_LENGTH - len(segment_ids) - 1
                )

                tokens_id = tokenizer.convert_tokens_to_ids(tokens)

                # tokens, segment_ids, tokens_id
                new_item["language"][timestep] = {
                    "tokens": tokens,
                    "tokens_id": tokens_id,
                    "segment_ids": segment_ids,
                }

            max_timestep = max(list(new_item["language"].keys()))

            total_navigation_instances = len(new_item["player_path"])
            total_question_asking_instances = len(list(new_item["language"].keys()))
            ratios.append([total_navigation_instances, total_question_asking_instances])
            for t in range(max_timestep):
                if t in new_item["language"].keys():
                    continue

                if t == 0:
                    new_item["language"][0] = {
                        "tokens": zeroth_time_tokens,
                        "tokens_id": tokenizer.convert_tokens_to_ids(
                            zeroth_time_tokens
                        ),
                        "segment_ids": zeroth_time_segment_ids,
                    }
                else:
                    new_item["language"][t] = new_item["language"][t - 1]

            new_item["max_timestep"] = max_timestep
            self.data.append(new_item)

        self.scans = set(self.scans)
        self.splits = splits

        temp1 = sum([i for i, j in ratios])
        temp2 = sum([j for i, j in ratios])
        logger.info(f"total/+ve Class Ratio Mean: {temp1/temp2}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


class ClassifierDataLoader(DataLoader):
    def __init__(
        self,
        feature_store=None,
        tokenizer=None,
        batch_size=100,
        splits=["train"],
        **kwargs,
    ):
        super(ClassifierDataLoader, self).__init__(batch_size=batch_size, **kwargs)

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.features = feature_store
        self.tokenizer = tokenizer
        self._load_nav_graphs()
        self.splits = splits

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        self.args = self.dataset.args

        self.batch = None

    def _load_nav_graphs(self):
        """ Load connectivity graph for each scan, useful for reasoning about shortest paths """
        logger.info("Loading navigation graphs for %d scans" % len(self.dataset.scans))
        self.graphs = load_nav_graphs(self.dataset.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _shortest_path_action(self, state, goalViewpointId):
        """ Determine next action on the shortest path to goal, for supervised training. """
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId  # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(
                        [scanId], [viewpointId], [0], [math.radians(-30)]
                    )
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (
                        loc.viewpointId not in adj_dict
                        or distance < adj_dict[loc.viewpointId]["distance"]
                    ):
                        adj_dict[loc.viewpointId] = {
                            "heading": loc_heading,
                            "elevation": loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "scanId": scanId,
                            "viewpointId": loc.viewpointId,  # Next viewpoint id
                            "pointId": ix,
                            "distance": distance,
                            "idx": j + 1,
                            "feature": np.concatenate((visual_feat, angle_feat), -1),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {
                    key: c[key]
                    for key in [
                        "normalized_heading",
                        "elevation",
                        "scanId",
                        "viewpointId",
                        "pointId",
                        "idx",
                    ]
                }
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new["pointId"]
                normalized_heading = c_new["normalized_heading"]
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new["heading"] = loc_heading
                angle_feat = utils.angle_feature(c_new["heading"], c_new["elevation"])
                c_new["feature"] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop("normalized_heading")
                candidate_new.append(c_new)
            return candidate_new

    def get_language_input(self, timestep, pad_token_id):
        seq_tensor = []
        segment_ids = []
        for item in self.batch:
            t = min(timestep, item["max_timestep"])
            try:
                tokens = item["language"][t]["tokens_id"]
                segment = item["language"][t]["segment_ids"]
            except:
                import pdb

                pdb.set_trace()
            seq_tensor.append(tokens)
            segment_ids.append(segment)

        seq_tensor = np.array(seq_tensor)
        segment_ids = np.array(segment_ids)

        seq_lengths = np.argmax(seq_tensor == pad_token_id, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

        mask = (seq_tensor == pad_token_id)[
            :, : seq_lengths[0]
        ]  # seq_lengths[0] is the Maximum length

        seq_tensor = torch.from_numpy(seq_tensor)
        segment_ids = torch.from_numpy(segment_ids)
        mask = torch.from_numpy(mask)

        return (
            Variable(seq_tensor, requires_grad=False).long().to(self.args.device),
            Variable(segment_ids, requires_grad=False).long().to(self.args.device),
            mask.byte().to(self.args.device),
            list(seq_lengths),
        )

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            base_view_id = state.viewIndex

            target = item["player_path"][-1]

            candidate = self.make_candidate(
                feature,
                state.scanId,
                state.location.viewpointId,
                state.viewIndex,
            )

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            if self.splits == ["test"]:
                teacher_action = state.location.viewpointId  # dummy
            else:
                teacher_action = self._shortest_path_action(state, target)

            obs.append(
                {
                    "inst_idx": item["inst_idx"],
                    "scan": state.scanId,
                    "viewpoint": state.location.viewpointId,
                    "viewIndex": state.viewIndex,
                    "heading": state.heading,
                    "elevation": state.elevation,
                    "feature": feature,
                    "candidate": candidate,
                    "step": state.step,
                    "navigableLocations": state.navigableLocations,
                    "teacher": teacher_action,
                }
            )
        return obs

    def reset(self):
        """ Load a new minibatch / episodes. """
        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item["start_pano"]["pano"] for item in self.batch]
        headings = [item["start_pano"]["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        """ Take action (same interface as makeActions) """
        self.env.makeActions(actions)
        return self._get_obs()
