import logging
import math
import random

import MatterSim
import networkx as nx
import numpy as np
import utils
# from get_oscar_model import special_tokens_dict
from torch.utils.data import DataLoader, Dataset
from utils_data import load_gameplay_data, load_nav_graphs, truncate_dialogs

from data_loader import EnvBatch

logger = logging.getLogger(__name__)


class R2RDataset(Dataset):
    def __init__(
        self,
        splits=["train"],
        tokenizer=None,
        path_type="planner_path",
        history="target",
        **kwargs
    ):
        super(R2RDataset, self).__init__()

        assert tokenizer is not None

        self.data = []
        self.scans = []
        self.tok = tokenizer
        for item in load_gameplay_data(
            splits,
        ):

            # For every dialog history, stitch together a single instruction string.
            self.scans.append(item["scan"])
            new_item = dict(item)
            new_item["inst_idx"] = item["inst_idx"]
            if history == "none":  # no language input at all
                new_item["instructions"] = ""
                if tokenizer:
                    new_item["instr_encoding"] = tokenizer.encode_sentence("")
            elif (
                history == "target" or len(item["dialog_history"]) == 0
            ):  # Have to use target only if no dialog history.
                tar = item["target"]
                new_item["instructions"] = "<TAR> " + tar
                if tokenizer:
                    new_item["instr_encoding"] = tokenizer.encode_sentence(
                        [tar], seps=["<TAR>"]
                    )
            elif history == "oracle_ans":
                ora_a = item["dialog_history"][-1][
                    "message"
                ]  # i.e., the last oracle utterance.
                tar = item["target"]
                new_item["instructions"] = "<ORA> " + ora_a + " <TAR> " + tar
                if tokenizer:
                    new_item["instr_encoding"] = tokenizer.encode_sentence(
                        [ora_a, tar], seps=["<ORA>", "<TAR>"]
                    )
            elif history == "nav_q_oracle_ans":
                nav_q = item["dialog_history"][-2]["message"]
                ora_a = item["dialog_history"][-1]["message"]
                tar = item["target"]
                new_item["instructions"] = (
                    "<NAV> " + nav_q + " <ORA> " + ora_a + " <TAR> " + tar
                )
                if tokenizer:
                    qa_enc = tokenizer.encode_sentence(
                        [nav_q, ora_a, tar], seps=["<NAV>", "<ORA>", "<TAR>"]
                    )
                    new_item["instr_encoding"] = qa_enc
            elif history == "all":
                dia_inst = ""
                sentences = []
                seps = []
                for turn in item["dialog_history"]:
                    sentences.append(turn["message"])
                    sep = "<NAV>" if turn["role"] == "navigator" else "<ORA>"
                    seps.append(sep)
                    dia_inst += sep + " " + turn["message"] + " "
                sentences.append(item["target"])
                seps.append("<TAR>")
                dia_inst += "<TAR> " + item["target"]
                new_item["instructions"] = dia_inst
                if tokenizer:
                    new_item["instr_encoding"] = tokenizer.encode_sentence(
                        sentences, seps=seps
                    )
            if tokenizer:
                if "ora_instructions" in item and len(item["ora_instructions"]) > 0:
                    new_item["ora_instr_encoding"] = tokenizer.encode_sentence(
                        item["ora_instructions"], seps="<ORA>"
                    )
                if "nav_instructions" in item and len(item["nav_instructions"]) > 0:
                    new_item["nav_instr_encoding"] = tokenizer.encode_sentence(
                        item["nav_instructions"], seps="<NAV>"
                    )
            # If evaluating against 'trusted_path', we need to calculate the trusted path and instantiate it.
            if path_type == "trusted_path":
                # The trusted path is either the planner_path or the player_path depending on whether the player_path
                # contains the planner_path goal (e.g., stricter planner oracle success of player_path
                # indicates we can 'trust' it, otherwise we fall back to the planner path for supervision).
                # Hypothesize that this will combine the strengths of good human exploration with the known good, if
                # short, routes the planner uses.
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
                    new_item["trusted_path"] = item["planner_path"][
                        :
                    ]  # trust the planner.

            self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits

        self.path_type = path_type
        logger.info(
            "R2RBatch loaded with %d instructions, using splits: %s"
            % (len(self.data), ",".join(splits))
        )

        # self.seed = seed
        # random.seed(self.seed)
        # random.shuffle(self.data)
        # self.ix = 0
        # self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


def R2RDataLoader_collate_fn(batch):
    return batch


class R2RDataLoader(DataLoader):
    def __init__(self, feature_store=None, batch_size=100, **kwargs):

        super(R2RDataLoader, self).__init__(batch_size=batch_size, **kwargs)

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.features = feature_store
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()

        self.buffered_state_dict = {}

        self.data_iter = None
        self.batch = None

        self.reset_epoch()

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

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            base_view_id = state.viewIndex

            if self.dataset.path_type in item:
                target = item[self.dataset.path_type][-1]
            else:
                target = item["start_pano"]["pano"]

            candidate = self.make_candidate(
                feature,
                state.scanId,
                state.location.viewpointId,
                state.viewIndex,
            )

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

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
                    "instructions": item["instructions"],
                    "teacher": teacher_action,
                    "generated_dialog_history": [],
                    "action_probs": [],
                }
            )

            if "instr_encoding" in item:
                obs[-1]["instr_encoding"] = item["instr_encoding"]
            if "nav_instr_encoding" in item:
                obs[-1]["nav_instr_encoding"] = item["nav_instr_encoding"]
            if "ora_instr_encoding" in item:
                obs[-1]["ora_instr_encoding"] = item["ora_instr_encoding"]
            obs[-1]["distance"] = self.distances[state.scanId][
                state.location.viewpointId
            ][item[self.dataset.path_type][-1]]
        return obs

    def _sample_items(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = self.__iter__()
            batch = next(self.data_iter)
        return batch

    def _next_minibatch(self):
        self.batch = self._sample_items()

        if len(self.batch) != self.batch_size:
            extra_batch = self._sample_items()
            new_batch = self.batch + extra_batch[: self.batch_size - len(self.batch)]
            assert len(new_batch) == self.batch_size
            self.batch = new_batch

    def reset_epoch(self):
        """Reset the data index to beginning of epoch. Primarily for testing.
        You must still call reset() for a new episode."""
        self.data_iter = self.__iter__()
        self.batch = None

    def reset(self, next_minibatch=True):
        """ Load a new minibatch / episodes. """
        if next_minibatch:
            self._next_minibatch()
        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item[self.dataset.path_type][0] for item in self.batch]
        headings = [item["start_pano"]["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        """ Take action (same interface as makeActions) """
        self.env.makeActions(actions)
        return self._get_obs()

    def random_start(self, J=0):
        viewpointIds = [item[self.dataset.path_type][0] for item in self.batch]
        for j in range(J):
            ids = []
            for i, (feature, state) in enumerate(self.env.getStates()):
                loc = np.random.choice(state.navigableLocations)
                ids.append(loc.viewpointId)
            self.reset_viewpointIds(ids)
        return viewpointIds

    def reset_viewpointIds(self, viewpointIds):
        for i, id in enumerate(viewpointIds):
            self.batch[i][self.dataset.path_type][0] = id
        self.reset(next_minibatch=False)
