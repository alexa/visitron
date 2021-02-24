# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import base64
import json
import logging
import math
import random
import sys

import MatterSim
import networkx as nx

# import csv
import numpy as np
from torch.utils.data import DataLoader, Dataset

import utils
from get_oscar_model import special_tokens_dict
from utils_data import load_datasets, load_nav_graphs, truncate_dialogs

logger = logging.getLogger(__name__)


def VLNDataloader_collate_fn(batch):
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
                feature = self.features[long_id][state.viewIndex, :]
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

    # def makeSimpleActions(self, simple_indices):
    #     """Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
    #     All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
    #     WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
    #     environment may not longer be navigable."""
    #     actions = []
    #     for i, index in enumerate(simple_indices):
    #         if index == 0:
    #             actions.append((1, 0, 0))
    #         elif index == 1:
    #             actions.append((0, -1, 0))
    #         elif index == 2:
    #             actions.append((0, 1, 0))
    #         elif index == 3:
    #             actions.append((0, 0, 1))
    #         elif index == 4:
    #             actions.append((0, 0, -1))
    #         else:
    #             sys.exit("Invalid simple action")
    #     self.makeActions(actions)


# class SingleBatchSimulator:
#     def __init__(self, image_w, image_h, vfov):

#         self.image_w = image_w
#         self.image_h = image_h
#         self.vfov = vfov
#         self.batch_size = 1

#         self.viewpoint_viewpointIdx = None
#         self.actions = None

#         self.sim = MatterSim.Simulator()
#         self.sim.setRenderingEnabled(False)
#         self.sim.setDiscretizedViewingAngles(True)
#         self.sim.setBatchSize(self.batch_size)
#         self.sim.setCameraResolution(self.image_w, self.image_h)
#         self.sim.setCameraVFOV(math.radians(self.vfov))
#         self.sim.initialize()

#         splits = ["train", "val_seen", "val_unseen", "test"]
#         self.scans = []
#         for split in splits:
#             with open("tasks/NDH/data/%s.json" % split) as f:
#                 items = json.load(f)
#                 new_scans = [item["scan"] for item in items]
#                 self.scans.extend(new_scans)
#         self.scans = set(self.scans)
#         self.graphs = load_nav_graphs(self.scans)

#     def shortest_path_action(self, state, nextViewpointId):
#         if state.location.viewpointId == nextViewpointId:
#             logger.info(f"Same viewpoint detected!")
#             return (0, 0, 0)  # do nothing

#         # Can we see the next viewpoint?
#         for i, loc in enumerate(state.navigableLocations):
#             if loc.viewpointId == nextViewpointId:
#                 # Look directly at the viewpoint before moving
#                 if loc.rel_heading > math.pi / 6.0:
#                     return (0, 1, 0)  # Turn right
#                 elif loc.rel_heading < -math.pi / 6.0:
#                     return (0, -1, 0)  # Turn left
#                 elif loc.rel_elevation > math.pi / 6.0 and state.viewIndex // 12 < 2:
#                     return (0, 0, 1)  # Look up
#                 elif loc.rel_elevation < -math.pi / 6.0 and state.viewIndex // 12 > 0:
#                     return (0, 0, -1)  # Look down
#                 else:
#                     return (i, 0, 0)  # Move
#         # Can't see it - first neutralize camera elevation
#         if state.viewIndex // 12 == 0:
#             return (0, 0, 1)  # Look up
#         elif state.viewIndex // 12 == 2:
#             return (0, 0, -1)  # Look down
#         # Otherwise decide which way to turn
#         pos = [state.location.x, state.location.y, state.location.z]
#         target_rel = self.graphs[state.scanId].node[nextViewpointId]["position"] - pos
#         target_heading = math.pi / 2.0 - math.atan2(
#             target_rel[1], target_rel[0]
#         )  # convert to rel to y axis
#         if target_heading < 0:
#             target_heading += 2.0 * math.pi
#         if state.heading > target_heading and state.heading - target_heading < math.pi:
#             return (0, -1, 0)  # Turn left
#         if target_heading > state.heading and target_heading - state.heading > math.pi:
#             return (0, -1, 0)  # Turn left
#         return (0, 1, 0)  # Turn right

#     def get_turn_actions(self, start_idx, end_idx):
#         start_elevation = start_idx // 12
#         end_elevation = end_idx // 12
#         if start_elevation != end_elevation:
#             delta_elevation = end_elevation - start_elevation
#             if delta_elevation >= 1:
#                 return (0, 0, 1)
#             else:
#                 return (0, 0, -1)
#         start_heading = start_idx % 12
#         end_heading = end_idx % 12

#         delta_heading = start_heading - end_heading
#         if delta_heading >= 6:
#             return (0, 1, 0)
#         elif delta_heading > 0:
#             return (0, -1, 0)
#         elif delta_heading > -6:
#             return (0, 1, 0)
#         elif delta_heading > -12:
#             return (0, -1, 0)
#         else:
#             import pdb

#             pdb.set_trace()
#             raise ValueError

#     def convert_actions_to_ints(self, actions):
#         env_actions = [
#             (0, -1, 0),  # left
#             (0, 1, 0),  # right
#             (0, 0, 1),  # up
#             (0, 0, -1),  # down
#             (1, 0, 0),  # forward
#             (0, 0, 0),  # <end>
#             (0, 0, 0),  # <start>
#             (0, 0, 0),  # <ignore>
#         ]
#         converted_actions = []
#         for action in actions:
#             if action == (0, -1, 0):
#                 new_action = 0
#             elif action == (0, 1, 0):
#                 new_action = 1
#             elif action == (0, 0, 1):
#                 new_action = 2
#             elif action == (0, 0, -1):
#                 new_action = 3
#             elif action == (1, 0, 0):
#                 new_action = 4
#             elif action == (0, 0, 0):
#                 print("end action detected!")
#                 new_action = 5
#             else:
#                 raise ValueError
#             converted_actions.append(new_action)
#         return converted_actions

#     def extract_actions_from_viewpoints(
#         self, path, scanId, start_heading, start_elevation, end_heading, end_elevation
#     ):
#         assert path != []

#         self.viewpoint_viewpointIdx = []
#         self.actions = []

#         start_viewpoint = path[0]
#         end_viewpoint = path[-1]

#         self.sim.newEpisode(
#             [scanId], [start_viewpoint], [start_heading], [start_elevation]
#         )

#         state = self.sim.getState()[0]
#         self.viewpoint_viewpointIdx.append(
#             (state.location.viewpointId, state.viewIndex)
#         )

#         if end_heading is None and end_elevation is None:
#             return self.actions, self.viewpoint_viewpointIdx

#         if len(path) == 1:
#             nextViewpointId = path[0]
#         else:
#             nextViewpointId = path[1]

#         skip = False
#         next_idx = 1
#         while state.location.viewpointId != end_viewpoint:
#             action = self.shortest_path_action(state, nextViewpointId)

#             # print(action, viewpoint_id_idx[-1])
#             self.sim.makeAction([action[0]], [action[1]], [action[2]])
#             if action[0] >= 1:
#                 next_idx += 1
#                 action = (1, 0, 0)
#             self.actions.append(action)

#             state = self.sim.getState()[0]
#             self.viewpoint_viewpointIdx.append(
#                 (state.location.viewpointId, state.viewIndex)
#             )

#             if next_idx == len(path):
#                 nextViewpointId = path[next_idx - 1]
#             else:
#                 nextViewpointId = path[next_idx]

#             if len(self.actions) >= 1000:
#                 skip = True
#                 break

#         if skip:
#             print(f"Skipping instr_idx: {item['inst_idx']}. Infinite loop detected!")
#             import pdb

#             pdb.set_trace()

#         assert state.location.viewpointId == end_viewpoint

#         intermediate_viewIndex = state.viewIndex
#         intermediate_heading = state.heading
#         intermediate_elevation = state.elevation

#         end_viewpoint = path[-1]
#         self.sim.newEpisode([scanId], [end_viewpoint], [end_heading], [end_elevation])
#         state = self.sim.getState()[0]
#         end_viewpoint_idx = state.viewIndex

#         if intermediate_viewIndex != end_viewpoint_idx:
#             self.sim.newEpisode(
#                 [scanId],
#                 [end_viewpoint],
#                 [intermediate_heading],
#                 [intermediate_elevation],
#             )

#             count = 0

#             while state.viewIndex != end_viewpoint_idx:
#                 action = self.get_turn_actions(state.viewIndex, end_viewpoint_idx)
#                 self.actions.append(action)
#                 self.sim.makeAction([action[0]], [action[1]], [action[2]])
#                 state = self.sim.getState()[0]
#                 self.viewpoint_viewpointIdx.append(
#                     (state.location.viewpointId, state.viewIndex)
#                 )
#                 count += 1

#                 if count >= 10:
#                     print("Infinite loop detected!")
#                     import pdb

#                     pdb.set_trace()

#         assert self.viewpoint_viewpointIdx[-1][0] == path[-1]
#         assert self.viewpoint_viewpointIdx[-1][1] == end_viewpoint_idx

#         self.actions = self.convert_actions_to_ints(self.actions)

#         return self.actions, self.viewpoint_viewpointIdx


class VLNDataset(Dataset):
    def __init__(
        self,
        args,
        splits=["train"],
        tokenizer=None,
        truncate_dialog=False,
        path_type="planner_path",
        add_ndh_data=True,
        add_r2r_data=False,
        add_r4r_data=False,
        add_rxr_data=False,
    ):
        super(VLNDataset, self).__init__()

        assert tokenizer is not None
        assert (add_ndh_data or add_r2r_data or add_r4r_data or add_rxr_data) is True

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
        # # TODO: ^^ add them as args ^^

        # # TOTAL 768

        count_planner = 0
        total = 0
        # # TOTAL 768
        if add_ndh_data:
            for item in load_datasets(splits, dataset_type="NDH"):

                self.scans.append(item["scan"])

                new_item = dict(item)
                new_item["inst_idx"] = item["inst_idx"]

                token_target = tokenizer.tokenize(item["target"])
                token_target = token_target[:MAX_TARGET_LENGTH]
                new_item["token_target"] = token_target

                token_dialog_history = []
                for turn in item["dialog_history"]:
                    token_turn = tokenizer.tokenize(turn["message"])
                    token_dialog_history.append(token_turn)

                if truncate_dialog:
                    # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                    token_dialog_history = truncate_dialogs(
                        token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                    )

                new_item["tokens_dialog_history"] = token_dialog_history

                tokens = [tokenizer.cls_token]
                segment_ids = [cls_token_segment_id]

                if not TAR_BACK:
                    if use_oscar_settings:
                        sep_token = tokenizer.sep_token
                    else:
                        sep_token = tokenizer.tar_token

                    tokens += [sep_token] + token_target
                    segment_ids += [tar_token_segment_id] * (len(token_target) + 1)

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

                new_item["target_dialog_tokens"] = tokens
                new_item["target_dialog_tokens_id"] = tokenizer.convert_tokens_to_ids(
                    tokens
                )
                new_item["target_dialog_segment_ids"] = segment_ids

                # If evaluating against 'trusted_path', we need to calculate the trusted path and instantiate it.
                if splits != ["test"] and path_type == "trusted_path":
                    # The trusted path is either the planner_path or the player_path depending on whether the player_path
                    # contains the planner_path goal (e.g., stricter planner oracle success of player_path
                    # indicates we can 'trust' it, otherwise we fall back to the planner path for supervision).
                    # Hypothesize that this will combine the strengths of good human exploration with the known good, if
                    # short, routes the planner uses.

                    total += 1
                    planner_goal = item["planner_path"][
                        -1
                    ]  # this could be length 1 if "plan" is to not move at all.
                    if (
                        planner_goal in item["player_path"][1:]
                    ):  # player walked through planner goal (did not start on it)
                        new_item["trusted_path"] = item["player_path"][
                            :
                        ]  # trust the player.
                    else:
                        count_planner += 1
                        new_item["trusted_path"] = item["planner_path"][
                            :
                        ]  # trust the planner.

                self.data.append(new_item)
        if path_type == "trusted_path":
            logger.info(
                f"Planner Paths: {count_planner}({100*count_planner/total}%) Player Paths: {total - count_planner}({100*(total - count_planner)/total}%) Total: {total}"
            )
        if add_r2r_data:
            for item in load_datasets(splits, dataset_type="R2R"):
                self.scans.append(item["scan"])

                for j, instr in enumerate(item["instructions"]):
                    new_item = dict(item)
                    new_item["inst_idx"] = "R2R_%s_%d" % (item["path_id"], j)

                    token_turn = tokenizer.tokenize(instr)
                    token_dialog_history = [token_turn]

                    if truncate_dialog:
                        # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                        token_dialog_history = truncate_dialogs(
                            token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                        )

                    new_item["tokens_dialog_history"] = token_dialog_history

                    tokens = [tokenizer.cls_token]
                    segment_ids = [cls_token_segment_id]

                    if use_oscar_settings:
                        sep_token = tokenizer.sep_token
                    else:
                        sep_token = tokenizer.tar_token

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

                    tokens += [tokenizer.sep_token]
                    segment_ids += [sep_token_segment_id]

                    tokens += [pad_token_id] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                    segment_ids += [pad_token_segment_id] * (
                        MAX_SEQ_LENGTH - len(segment_ids) - 1
                    )

                    new_item["target_dialog_tokens"] = tokens
                    new_item[
                        "target_dialog_tokens_id"
                    ] = tokenizer.convert_tokens_to_ids(tokens)
                    new_item["target_dialog_segment_ids"] = segment_ids

                    new_item["planner_path"] = item["path"]
                    new_item["player_path"] = item["path"]
                    new_item["trusted_path"] = item["path"]
                    new_item["nav_history"] = item["path"]

                    new_item["start_pano"] = {
                        "heading": item["heading"],
                        "elevation": 0,
                        "pano": item["path"][0],
                    }

                    self.data.append(new_item)

        if add_r4r_data:
            for item in load_datasets(splits, dataset_type="R4R"):
                self.scans.append(item["scan"])

                for j, instr in enumerate(item["instructions"]):
                    new_item = dict(item)
                    new_item["inst_idx"] = "R4R_%s_%d" % (item["path_id"], j)

                    token_turn = tokenizer.tokenize(instr)
                    token_dialog_history = [token_turn]

                    if truncate_dialog:
                        # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                        token_dialog_history = truncate_dialogs(
                            token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                        )

                    new_item["tokens_dialog_history"] = token_dialog_history

                    tokens = [tokenizer.cls_token]
                    segment_ids = [cls_token_segment_id]

                    if use_oscar_settings:
                        sep_token = tokenizer.sep_token
                    else:
                        sep_token = tokenizer.tar_token

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

                    tokens += [tokenizer.sep_token]
                    segment_ids += [sep_token_segment_id]

                    tokens += [pad_token_id] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                    segment_ids += [pad_token_segment_id] * (
                        MAX_SEQ_LENGTH - len(segment_ids) - 1
                    )

                    new_item["target_dialog_tokens"] = tokens
                    new_item[
                        "target_dialog_tokens_id"
                    ] = tokenizer.convert_tokens_to_ids(tokens)
                    new_item["target_dialog_segment_ids"] = segment_ids

                    new_item["planner_path"] = item["path"]
                    new_item["player_path"] = item["path"]
                    new_item["trusted_path"] = item["path"]
                    new_item["nav_history"] = item["path"]

                    new_item["start_pano"] = {
                        "heading": item["heading"],
                        "elevation": 0,
                        "pano": item["path"][0],
                    }

                    self.data.append(new_item)

        if add_rxr_data:
            for item in load_datasets(splits, dataset_type="RxR"):
                self.scans.append(item["scan"])
                new_item = dict(item)
                new_item["inst_idx"] = "RxR_%s" % item["instruction_id"]
                instr = item["instruction"]

                token_turn = tokenizer.tokenize(instr)
                token_dialog_history = [token_turn]

                if truncate_dialog:
                    # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                    token_dialog_history = truncate_dialogs(
                        token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                    )

                new_item["tokens_dialog_history"] = token_dialog_history

                tokens = [tokenizer.cls_token]
                segment_ids = [cls_token_segment_id]

                if use_oscar_settings:
                    sep_token = tokenizer.sep_token
                else:
                    sep_token = tokenizer.tar_token

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

                tokens += [tokenizer.sep_token]
                segment_ids += [sep_token_segment_id]

                tokens += [pad_token_id] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                segment_ids += [pad_token_segment_id] * (
                    MAX_SEQ_LENGTH - len(segment_ids) - 1
                )

                new_item["target_dialog_tokens"] = tokens
                new_item["target_dialog_tokens_id"] = tokenizer.convert_tokens_to_ids(
                    tokens
                )
                new_item["target_dialog_segment_ids"] = segment_ids

                new_item["planner_path"] = item["path"]
                new_item["player_path"] = item["path"]
                new_item["trusted_path"] = item["path"]
                new_item["nav_history"] = item["path"]

                new_item["start_pano"] = {
                    "heading": item["heading"],
                    "elevation": 0,
                    "pano": item["path"][0],
                }

                self.data.append(new_item)

        self.scans = set(self.scans)
        self.splits = splits
        self.path_type = path_type

        logger.info(
            "VLNDataset loaded with %d instructions, using splits: %s NDH: %r R2R: %r R4R: %r RxR: %r Supervision: %s"
            % (
                len(self.data),
                ",".join(splits),
                add_ndh_data,
                add_r2r_data,
                add_r4r_data,
                add_rxr_data,
                path_type,
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


class VLNDataLoader(DataLoader):
    def __init__(
        self,
        feature_store=None,
        tokenizer=None,
        batch_size=100,
        splits=["train"],
        **kwargs,
    ):
        super(VLNDataLoader, self).__init__(batch_size=batch_size, **kwargs)

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.features = feature_store
        self.tokenizer = tokenizer
        self._load_nav_graphs()
        self.splits = splits

        self.sim = utils.new_simulator()

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
            return (0, 0, 0)  # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i, loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi / 6.0:
                    return (0, 1, 0)  # Turn right
                elif loc.rel_heading < -math.pi / 6.0:
                    return (0, -1, 0)  # Turn left
                elif loc.rel_elevation > math.pi / 6.0 and state.viewIndex // 12 < 2:
                    return (0, 0, 1)  # Look up
                elif loc.rel_elevation < -math.pi / 6.0 and state.viewIndex // 12 > 0:
                    return (0, 0, -1)  # Look down
                else:
                    return (i, 0, 0)  # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex // 12 == 0:
            return (0, 0, 1)  # Look up
        elif state.viewIndex // 12 == 2:
            return (0, 0, -1)  # Look down
        # Otherwise decide which way to turn
        pos = [state.location.x, state.location.y, state.location.z]
        target_rel = self.graphs[state.scanId].node[nextViewpointId]["position"] - pos
        target_heading = math.pi / 2.0 - math.atan2(
            target_rel[1], target_rel[0]
        )  # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0 * math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0, -1, 0)  # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0, -1, 0)  # Turn left
        return (0, 1, 0)  # Turn right

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            if self.dataset.path_type in item:
                target = item[self.dataset.path_type][-1]
            else:
                target = item["start_pano"]["pano"]
            obs.append(
                {
                    "inst_idx": item["inst_idx"],
                    "scan": state.scanId,
                    "viewpoint": state.location.viewpointId,
                    "viewIndex": state.viewIndex,
                    "heading": state.heading,
                    "elevation": state.elevation,
                    "feature": feature,
                    "step": state.step,
                    "navigableLocations": state.navigableLocations,
                    "teacher": self._shortest_path_action(state, target),
                }
            )

        return obs

    def reset(self):
        """ Load a new minibatch / episodes. """
        scanIds = [item["scan"] for item in self.batch]
        if self.dataset.path_type in self.batch[0]:
            viewpointIds = [item[self.dataset.path_type][0] for item in self.batch]
        else:
            # In the test dataset there is no path provided, so we just load the start viewpoint
            viewpointIds = [item["start_pano"]["pano"] for item in self.batch]
        headings = [item["start_pano"]["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        """ Take action (same interface as makeActions) """
        self.env.makeActions(actions)
        return self._get_obs()
