# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import sys
import os
from multiprocessing import Pool
from itertools import repeat

sys.path.append("build")
import MatterSim

# import csv
import numpy as np
import math
import base64
import logging
import json
import random
import networkx as nx
from tqdm import tqdm
import argparse


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open("connectivity/%s_connectivity.json" % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array(
                                [item["pose"][3], item["pose"][7], item["pose"][11]]
                            )
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs


def load_datasets(splits, dataset_type="NDH"):
    data = []

    if dataset_type == "NDH":
        data_root = "tasks/NDH/data/"
    elif dataset_type == "R2R":
        data_root = "tasks/R2R/data/R2R_"
    else:
        raise NotImplementedError

    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open(data_root + "%s.json" % split) as f:
            data += json.load(f)
    return data


class SingleBatchSimulator:
    def __init__(self):

        self.image_w = 600
        self.image_h = 600
        self.vfov = 80
        self.batch_size = 1

        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

        splits = ["train", "val_seen", "val_unseen", "test"]
        self.scans = []
        for split in splits:
            with open("tasks/NDH/data/%s.json" % split) as f:
                items = json.load(f)
                new_scans = [item["scan"] for item in items]
                self.scans.extend(new_scans)
        self.scans = set(self.scans)
        self.graphs = load_nav_graphs(self.scans)

        self.env_actions = {
            "left": (0, -1, 0),  # left
            "right": (0, 1, 0),  # right
            "up": (0, 0, 1),  # up
            "down": (0, 0, -1),  # down
            "forward": (1, 0, 0),  # forward
            "<end>": (0, 0, 0),  # <end>
            "<start>": (0, 0, 0),  # <start>
            "<ignore>": (0, 0, 0),  # <ignore>
        }

    def newEpisode(self, scanId, viewpointId, heading, elevation):
        self.sim.newEpisode([scanId], [viewpointId], [heading], [elevation])

    def getState(self):
        return self.sim.getState()[0]

    def makeAction(self, action, verbose=False):
        ix = action[0]
        heading = action[1]
        elevation = action[2]
        if verbose:
            print(
                f"Before - Scan: {self.getState().scanId} Viewpoint: {self.getState().location.viewpointId} ViewIndex: {self.getState().viewIndex}"
            )
        self.sim.makeAction([ix], [heading], [elevation])
        if verbose:
            print(f"Action taken: {ix} {heading} {elevation}")
            print(
                f"After - Scan: {self.getState().scanId} Viewpoint: {self.getState().location.viewpointId} ViewIndex: {self.getState().viewIndex}"
            )

    def getCurrentViewpointViewIndex(self):
        return self.getState().viewIndex

    def goToNextViewpoint(self, nextViewpointId, nextViewpointViewData):
        state = self.getState()

        if state.location.viewpointId == nextViewpointId:
            print(f"Same viewpoint detected!")
            return  # do nothing

        src_point = state.viewIndex
        trg_point = nextViewpointViewData["pointId"]

        # print(src_point, trg_point)

        src_level = (src_point) // 12  # The point idx started from 0
        trg_level = (trg_point) // 12

        while src_level < trg_level:  # Tune up
            self.makeAction(self.env_actions["up"])  # , verbose=True)
            # print("turn up")
            src_level += 1

        while src_level > trg_level:  # Tune down
            self.makeAction(self.env_actions["down"])  # , verbose=True)
            # print("turn down")
            src_level -= 1

        while self.getState().viewIndex != trg_point:  # Turn right until the target
            self.makeAction(self.env_actions["right"])  # , verbose=True)
            # print("turn right")

        self.makeAction((nextViewpointViewData["idx"], 0, 0))  # , verbose=True)
        # print(f"move forward {nextViewpointViewData['idx']}")
        if self.getState().location.viewpointId != nextViewpointId:
            import pdb

            pdb.set_trace()


def loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def getNextViewpointViewData(
    scanId, current_viewpoint_id, current_heading, nextViewpointId, relative=False,
):
    sim = SingleBatchSimulator()
    adj_dict = {}
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(
                scanId,
                current_viewpoint_id,
                current_heading - math.pi if relative else 0,
                math.radians(-30),
            )
        elif ix % 12 == 0:
            sim.makeAction((0, 1.0, 1.0))
        else:
            sim.makeAction((0, 1.0, 0))

        state = sim.getState()

        # get adjacent locations
        for j, loc in enumerate(state.navigableLocations[1:]):
            # if a loc is visible from multiple view, use the closest
            # view (in angular distance) as its representation
            distance = loc_distance(loc)

            if (
                loc.viewpointId not in adj_dict
                or distance < adj_dict[loc.viewpointId]["distance"]
            ):
                adj_dict[loc.viewpointId] = {
                    "pointId": ix,
                    "distance": distance,
                    "idx": j + 1,
                }

    return adj_dict[nextViewpointId]


def extract_data(split, dataset_to_use, job_index, total_jobs):
    sim = SingleBatchSimulator()
    data = []

    dataset = load_datasets([split], dataset_type=dataset_to_use)
    dataset = dataset[job_index::total_jobs]

    for n, item in enumerate(tqdm(dataset)):

        if dataset_to_use == "NDH":
            path = item["planner_path"]
        elif dataset_to_use == "R2R":
            path = item["path"]

        if len(path) < 2:
            print(
                f"instr idx: {item['inst_idx']} length of path: {len(path)}, skipping..."
            )
            continue

        scanId = item["scan"]
        vId = path[0]

        if dataset_to_use == "NDH":
            heading = item["start_pano"]["heading"]
            elevation = item["start_pano"]["elevation"]
        elif dataset_to_use == "R2R":
            heading = item["heading"]
            elevation = 0

        sim.newEpisode(scanId, vId, heading, elevation)

        for i, curr_vId in enumerate(path[:-1]):

            next_vId = path[i + 1]

            current_view_index = sim.getCurrentViewpointViewIndex()
            next_view_data = getNextViewpointViewData(
                scanId, curr_vId, sim.getState().heading, next_vId, relative=False,
            )

            target_view_data = getNextViewpointViewData(
                scanId, curr_vId, sim.getState().heading, next_vId, relative=True,
            )

            sim.goToNextViewpoint(next_vId, next_view_data)

            new_item = {}

            new_item["scan"] = item["scan"]
            new_item["viewpoint"] = curr_vId
            new_item["current_view_index"] = current_view_index
            new_item["target_abs_view_index"] = next_view_data["pointId"]
            new_item["target_rel_view_index"] = target_view_data["pointId"]

            if dataset_to_use == "NDH":
                new_item["inst_idx"] = f"{item['inst_idx']}_{i}"
                new_item["dialog_history"] = item["dialog_history"]
                new_item["target"] = item["target"]
                data.append(new_item)
            elif dataset_to_use == "R2R":
                for instr_no, instr in enumerate(item["instructions"]):
                    new_new_item = dict(new_item)
                    new_new_item["inst_idx"] = f"{item['path_id']}_{i}_{instr_no}"
                    new_new_item["dialog_history"] = instr
                    data.append(new_new_item)

    with open(
        f"tasks/NDH/pretrain_data/{dataset_to_use}_{split}_{job_index}_{total_jobs}.json",
        "w",
    ) as f:
        json.dump(data, f)


def merge_jsons(split, dataset_to_use, total_jobs):
    final_data = []

    for job_index in range(total_jobs):
        with open(
            f"tasks/NDH/pretrain_data/{dataset_to_use}_{split}_{job_index}_{total_jobs}.json",
            "r",
        ) as f:
            data = json.load(f)
            print(f"Loaded data of length {len(data)}")
            final_data += data

    print(f"Final data of length {len(final_data)}")

    with open(f"tasks/NDH/pretrain_data/{dataset_to_use}_{split}.json", "w") as f:
        json.dump(final_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_to_use", type=str, required=True, choices=["NDH", "R2R"],
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "val_seen", "val_unseen"],
    )
    parser.add_argument(
        "--total_jobs", default=1, type=int,
    )

    args = parser.parse_args()

    processes = range(args.total_jobs)

    with Pool(processes=args.total_jobs) as pool:
        pool.starmap(
            extract_data,
            zip(
                repeat(args.split),
                repeat(args.dataset_to_use),
                processes,
                repeat(args.total_jobs),
            ),
        )

    merge_jsons(args.split, args.dataset_to_use, args.total_jobs)
