# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import csv
import sys
import os
import time
import math
import base64
import logging
from itertools import chain
import pickle
import networkx as nx
import numpy as np

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


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


def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open("tasks/NDH/data/%s.json" % split) as f:
            data += json.load(f)
    return data


def truncate_dialogs(sentences, amount, left=True):
    """
    Truncate `dialogs` at a token-level TO the specified `amount` FROM the direction specified by `left`
    Consider length of each dialog to be len(dialog) + 1 as `[QUES]` or `[ANS]` tag needs to be counted as well.
    """

    if amount is None:
        return sentences

    if (len(list(chain(*sentences))) + len(sentences)) <= amount:
        return sentences

    if left:
        reversed_sentences = sentences[::-1]
        reversed_truncated_sentences = []
        amount_appended = 0
        for turn in reversed_sentences:
            if amount_appended < amount:
                remaining_amount = amount - amount_appended
                if (len(turn) + 1) <= remaining_amount:
                    reversed_truncated_sentences.append(turn)
                    amount_appended += len(turn) + 1
                else:
                    reversed_truncated_sentences.append(turn[-remaining_amount + 1 :])
                    amount_appended += len(turn[-remaining_amount + 1 :]) + 1
                    break  # can break out of the loop at this point
        truncated_sentences = reversed_truncated_sentences[::-1]
        return truncated_sentences
    else:
        truncated_sentences = []
        amount_appended = 0
        for turn in sentences:
            if amount_appended < amount:
                remaining_amount = amount - amount_appended
                if (len(turn) + 1) <= remaining_amount:
                    truncated_sentences.append(turn)
                    amount_appended += len(turn) + 1
                else:
                    truncated_sentences.append(turn[: remaining_amount - 1])
                    amount_appended += len(turn[: remaining_amount - 1]) + 1
                    break  # can break out of the loop at this point
        return truncated_sentences


def load_img_pickle_features(img_feat_dir, img_feature_file, candidate_feature_file):
    t_s = time.time()

    img_feature_path = os.path.join(img_feat_dir, img_feature_file)

    logger.info(f"Loading Image Features from {img_feature_path}")

    with open(img_feature_path, "rb") as f:
        loaded_feature_data = pickle.load(f)

    image_w = loaded_feature_data[0]["image_w"]
    image_h = loaded_feature_data[0]["image_h"]
    vfov = loaded_feature_data[0]["vfov"]

    features = {"features": {}, "image_w": image_w, "image_h": image_h, "vfov": vfov}
    region_tokens = {}

    for item in loaded_feature_data:
        long_id = f"{item['scanId']}_{item['viewpointId']}"

        features["features"][long_id] = item["features"]

        region_tokens[long_id] = item["region_tokens"]

    t_e = time.time()
    logger.info(
        "Loaded Image Features from {} in time: {:0.2f} mins".format(
            img_feature_path, (t_e - t_s) / 60.0
        )
    )

    candidate_feature_path = os.path.join(img_feat_dir, candidate_feature_file)

    logger.info(f"Loading Candidate Image Features from {candidate_feature_path}")

    with open(candidate_feature_path, "rb") as f:
        candidate_feature_data = pickle.load(f)

    cand_features = {}

    for item in candidate_feature_data:
        long_id = f"{item['scanId']}_{item['viewpointId']}"

        cand_features[long_id] = item["features"]

    t_e = time.time()
    logger.info(
        "Loaded Candidate Image Features from {} in time: {:0.2f} mins".format(
            candidate_feature_path, (t_e - t_s) / 60.0
        )
    )

    return features, region_tokens, cand_features


def read_img_features(path=None, feature_size=2048, blind=False):
    if path:
        logger.info("Loading image features from %s" % path)
        if blind:
            logger.info("... and zeroing them out for 'blind' evaluation")
        tsv_fieldnames = [
            "scanId",
            "viewpointId",
            "image_w",
            "image_h",
            "vfov",
            "features",
        ]
        features = {}
        with open(path, "rt") as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter="\t", fieldnames=tsv_fieldnames
            )
            for item in reader:
                image_h = int(item["image_h"])
                image_w = int(item["image_w"])
                vfov = int(item["vfov"])
                long_id = item["scanId"] + "_" + item["viewpointId"]
                if not blind:
                    features[long_id] = np.frombuffer(
                        base64.b64decode(item["features"]), dtype=np.float32
                    ).reshape((36, feature_size))
                else:
                    features[long_id] = np.zeros((36, feature_size), dtype=np.float32)
    else:
        logger.info("Image features not provided")
        features = None
        image_w = 640
        image_h = 480
        vfov = 60

    dictionary = {
        "features": features,
        "image_w": image_w,
        "image_h": image_h,
        "vfov": vfov,
    }
    return dictionary


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
