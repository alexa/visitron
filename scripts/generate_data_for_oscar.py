# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import sys
import json
import math
from tqdm import tqdm
import networkx as nx
import numpy as np
import MatterSim


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


splits = ["train", "val_seen", "val_unseen", "test"]
scans = []
for split in splits:
    with open("tasks/NDH/data/%s.json" % split) as f:
        items = json.load(f)
        new_scans = [item["scan"] for item in items]
        scans.extend(new_scans)
scans = set(scans)
graphs = load_nav_graphs(scans)


def shortest_path_action(state, nextViewpointId):
    if state.location.viewpointId == nextViewpointId:
        return (0, 0, 0)  # do nothing

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
    target_rel = graphs[state.scanId].node[nextViewpointId]["position"] - pos
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


def get_turn_actions(start_idx, end_idx):
    start_elevation = start_idx // 12
    end_elevation = end_idx // 12
    if start_elevation != end_elevation:
        delta_elevation = end_elevation - start_elevation
        if delta_elevation >= 1:
            return (0, 0, 1)
        else:
            return (0, 0, -1)
    start_heading = start_idx % 12
    end_heading = end_idx % 12

    delta_heading = start_heading - end_heading
    if delta_heading >= 6:
        return (0, 1, 0)
    elif delta_heading > 0:
        return (0, -1, 0)
    elif delta_heading > -6:
        return (0, 1, 0)
    elif delta_heading > -12:
        return (0, -1, 0)
    else:
        import pdb

        pdb.set_trace()
        raise ValueError


def load_data(splits):

    print(f"Processing {' '.join(splits)}")
    skipped_count = 0

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.setCameraResolution(600, 600)
    sim.setCameraVFOV(math.radians(80))
    sim.initialize()

    cvdn_data = {}
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open("tasks/CVDN/data/%s.json" % split) as f:
            items = json.load(f)
            id2item = {item["idx"]: item for item in items}
            cvdn_data.update(id2item)

    data = []
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open("tasks/NDH/data/%s.json" % split) as f:
            data += json.load(f)

    new_data = []
    for item in tqdm(data):
        new_item = {}
        new_item["inst_idx"] = item["inst_idx"]

        new_item["target"] = item["target"]
        question_answers = []
        q_a_item = None
        for i, turn in enumerate(item["dialog_history"]):
            if i % 2 == 0:
                assert turn["role"] == "navigator"
                q_a_item = [turn["message"]]
            else:
                assert turn["role"] == "oracle"
                q_a_item.append(turn["message"])
                question_answers.append(q_a_item)
        new_item["dialog_history"] = question_answers

        new_item["scanId"] = item["scan"]

        nav_history_path = item["nav_history"]

        if len(nav_history_path) == 0:
            continue

        start_viewpoint = nav_history_path[0]
        end_viewpoint = nav_history_path[-1]

        start_heading, start_elevation = 2.0, 17.5
        cvdn_item = cvdn_data[item["game_idx"]]
        if "nav_camera" in cvdn_item and len(cvdn_item["nav_camera"]) > 0:
            nav_camera = cvdn_item["nav_camera"][0]
            if "message" in nav_camera:
                start_heading = nav_camera["message"][-1]["heading"]
                start_elevation = nav_camera["message"][-1]["elevation"]

        actions = []
        viewpoint_id_idx = []

        sim.newEpisode([new_item["scanId"]], [start_viewpoint], [start_heading], [0])

        state = sim.getState()[0]
        viewpoint_id_idx.append((state.location.viewpointId, state.viewIndex))

        if len(nav_history_path) == 1:
            nextViewpointId = nav_history_path[0]
        else:
            nextViewpointId = nav_history_path[1]

        skip = False
        current_idx = 1
        while state.location.viewpointId != end_viewpoint:
            action = shortest_path_action(state, nextViewpointId)

            # print(action, viewpoint_id_idx[-1])
            sim.makeAction([action[0]], [action[1]], [action[2]])
            if action[0] >= 1:
                current_idx += 1
                action = (1, 0, 0)
            actions.append(action)

            state = sim.getState()[0]
            viewpoint_id_idx.append((state.location.viewpointId, state.viewIndex))

            if current_idx == len(nav_history_path):
                nextViewpointId = nav_history_path[current_idx - 1]
            else:
                nextViewpointId = nav_history_path[current_idx]

            if len(actions) >= 1000:
                skip = True
                break

        if skip:
            skipped_count += 1
            print(f"Skipping instr_idx: {item['inst_idx']}")
            continue

        nav_his_end_heading = state.heading
        nav_his_end_elevation = state.elevation
        nav_hist_end_viewIndex = state.viewIndex

        start_viewpoint = item["start_pano"]["pano"]
        start_heading = item["start_pano"]["heading"]

        sim.newEpisode([new_item["scanId"]], [start_viewpoint], [start_heading], [0])
        state = sim.getState()[0]
        if state.location.viewpointId != viewpoint_id_idx[-1][0]:
            print(
                f"state.location.viewpointId != viewpoint_id_idx[-1][0] for {item['inst_idx']}"
            )
            skipped_count += 1
            print(f"Skipping instr_idx: {item['inst_idx']}")
            continue

        start_viewIndex = state.viewIndex
        if start_viewIndex != viewpoint_id_idx[-1][1]:
            sim.newEpisode(
                [new_item["scanId"]],
                [start_viewpoint],
                [nav_his_end_heading],
                [nav_his_end_elevation],
            )

            count = 0
            while state.viewIndex != start_viewIndex:
                # print(state.viewIndex, start_viewIndex)
                action = get_turn_actions(state.viewIndex, start_viewIndex)
                actions.append(action)
                sim.makeAction([action[0]], [action[1]], [action[2]])
                state = sim.getState()[0]
                viewpoint_id_idx.append((state.location.viewpointId, state.viewIndex))
                # print(action, viewpoint_id_idx[-1])
                count += 1

                if count >= 10:
                    print("Infinite loop detected!")
                    import pdb

                    pdb.set_trace()

        new_item["viewpoint_id_idx"] = viewpoint_id_idx
        new_item["nav_history_actions"] = actions

        if len(item["planner_path"]) == 1:
            next_action = (0, 0, 0)
        else:
            nextViewpointId = item["planner_path"][1]
            next_action = shortest_path_action(state, nextViewpointId)
            if next_action[0] >= 1:
                next_action = (1, 0, 0)

        new_item["next_action"] = next_action

        new_data.append(new_item)

    print(f"Total skipped: {skipped_count}/{len(data)} in {' '.join(splits)}")
    return new_data


for split in ["val_unseen", "val_seen", "train"]:
    data = load_data([split])
    with open(f"viewpoint_action_data_{split}.json", "w") as f:
        json.dump(data, f)
