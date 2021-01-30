# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import base64
import csv
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from itertools import chain

import lmdb
import networkx as nx
import numpy as np
from tqdm import tqdm

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


def get_data_root(dataset_type="NDH"):
    if dataset_type == "NDH":
        data_root = "srv/task_data/NDH/data/"
    elif dataset_type == "CVDN":
        data_root = "srv/task_data/CVDN/data/"
    elif dataset_type == "R2R":
        data_root = "srv/task_data/R2R/data/R2R_"
    elif dataset_type == "R4R":
        data_root = "srv/task_data/R4R/data/R4R_"
    elif dataset_type == "RxR":
        data_root = "srv/task_data/RxR/data"
    elif dataset_type == "PretrainNDH":
        data_root = "srv/task_data/pretrain_data/NDH_"
    elif dataset_type == "PretrainR2R":
        data_root = "srv/task_data/pretrain_data/R2R_"
    elif dataset_type == "PretrainR4R":
        data_root = "srv/task_data/pretrain_data/R4R_"
    elif dataset_type == "PretrainRxR":
        data_root = "srv/task_data/pretrain_data/RxR_"
    else:
        raise NotImplementedError
    return data_root


# def load_datasets(splits, dataset_type="NDH"):
#     data = []

#     data_root = get_data_root(dataset_type)

#     for split in splits:
#         assert split in ["train", "val_seen", "val_unseen", "test"]
#         with open(data_root + "%s.json" % split) as f:
#             data += json.load(f)
#     return data


def load_datasets(splits, dataset_type="NDH"):
    data = []

    data_root = get_data_root(dataset_type)

    if dataset_type == "RxR":
        import jsonlines

        assert splits == ["train"]
        with jsonlines.open(f"{data_root}/rxr_train_guide.jsonl") as f:
            for line in f.iter():
                data.append(line)
        return data

    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open(data_root + "%s.json" % split) as f:
            data += json.load(f)
    return data


def load_classifier_data(splits):
    data = []
    raw_data = []

    data_root = get_data_root("CVDN")
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        data_path = data_root + "%s.json" % split
        with open(data_path) as f:
            items = json.load(f)
        raw_data.extend(items)

    for item in raw_data:
        item["inst_idx"] = str(item["idx"])
        item["planner_path"] = item["planner_nav_steps"]
        item["player_path"] = item["nav_steps"]
        item["nav_history"] = item["player_path"]

        heading, elevation = 2.0, 17.5
        if "nav_camera" in item and len(item["nav_camera"]) > 0:
            nav_camera = item["nav_camera"][0]
            if "message" in nav_camera:
                heading = nav_camera["message"][-1]["heading"]
                elevation = nav_camera["message"][-1]["elevation"]
        item["start_pano"] = {
            "heading": heading,
            "elevation": elevation,
            "pano": item["planner_nav_steps"][0],
        }
        dialog = {0: []}
        last_timestep = 0

        for index, turn in enumerate(item["dialog_history"]):
            if index % 2 == 0:
                assert turn["role"] == "navigator"
                timestep = turn["nav_idx"]
                message = turn["message"]
                dialog_history = dialog[last_timestep]
                dialog_history.append(message)
                dialog[timestep] = dialog_history
                last_timestep = timestep
            else:
                if timestep != turn["nav_idx"]:
                    logger.info(
                        "Timestep for oracle and navigator mismatch, correcting it. "
                        f"Timestep: {timestep} turn['nav_idx']: {turn['nav_idx']}"
                    )
                    # print(item["dialog_history"])
                assert turn["role"] == "oracle"
                # assert timestep == turn["nav_idx"]
                message = turn["message"]
                dialog_history = dialog[timestep]
                dialog_history.append(message)
                dialog[timestep] = dialog_history

        item["dialog_history"] = dialog
        item["request_locations"] = list(dialog.keys())
        data.append(item)
    return data


def load_gameplay_data(splits):
    data = []
    data_root = get_data_root("CVDN")
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        logger.info("Using CVDN for " + split + "!\n\n\n")
        data_source = data_root + split + ".json"
        with open(data_source) as f:
            items = json.load(f)
            new_items = []
            for item in items:
                item["inst_idx"] = item["idx"]
                item["planner_path"] = item["planner_nav_steps"]
                item["player_path"] = item["nav_steps"]
                item["nav_history"] = item["player_path"]
                heading, elevation = 2.0, 17.5
                if "nav_camera" in item and len(item["nav_camera"]) > 0:
                    nav_camera = item["nav_camera"][0]
                    if "message" in nav_camera:
                        heading = nav_camera["message"][-1]["heading"]
                        elevation = nav_camera["message"][-1]["elevation"]
                item["start_pano"] = {
                    "heading": heading,
                    "elevation": elevation,
                    "pano": item["planner_nav_steps"][0],
                }
                nav_ins, ora_ins, request_locations, nav_seen, ora_seen, nav_idx = (
                    [],
                    [],
                    {},
                    [],
                    [],
                    0,
                )
                for index, turn in enumerate(item["dialog_history"]):
                    if turn["role"] == "navigator":
                        nav_ins.append(turn["message"])
                        if len(ora_seen) > 0:
                            request_locations[nav_idx] = [
                                " ".join(nav_seen),
                                " ".join(ora_seen),
                                index,
                            ]
                            ora_seen = []
                            nav_seen = []
                        nav_seen.append(turn["message"])
                    else:
                        ora_ins.append(turn["message"])
                        if len(nav_seen) > 0:
                            nav_idx = int(turn["nav_idx"])
                            ora_seen.append(turn["message"])
                if len(ora_seen) > 0:
                    request_locations[nav_idx] = [
                        nav_seen[-1],
                        ora_seen[-1],
                        len(item["dialog_history"]),
                    ]  # [' '.join(nav_seen), ' '.join(ora_seen), len(item['dialog_history'])]
                item["nav_instructions"] = " ".join(nav_ins)
                item["ora_instructions"] = " ".join(ora_ins)
                if (
                    len(item["nav_instructions"]) == 0
                    or len(item["ora_instructions"]) == 0
                ):
                    continue
                item["request_locations"] = request_locations
                item["inst_idx"] = str(item["inst_idx"])
                assert len(item["player_path"]) > 1, item["player_path"]
                new_items.append(item)
            data += new_items
    return data


def save_preprocessed_data(data, splits, version, dataset_type="NDH"):
    data_root = get_data_root(dataset_type)
    combined_split = "_".join(splits)
    path = f"{data_root}{combined_split}_preprocessed_{version}.pickle"
    logger.info(f"Saving preprocessed data to {path}")
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=-1)


def check_and_load_preprocessed_data(splits, version, dataset_type="NDH"):
    if dataset_type == "NDH":
        data_root = "srv/task_data/NDH/data/"
    elif dataset_type == "R2R":
        data_root = "srv/task_data/R2R/data/R2R_"
    elif dataset_type == "R4R":
        data_root = "srv/task_data/R4R/data/R4R_"
    elif dataset_type == "RxR":
        data_root = "srv/task_data/RxR/data/RxR_"
    elif dataset_type == "PretrainNDH":
        data_root = "srv/task_data/pretrain_data/NDH_"
    elif dataset_type == "PretrainR2R":
        data_root = "srv/task_data/pretrain_data/R2R_"
    elif dataset_type == "PretrainR4R":
        data_root = "srv/task_data/pretrain_data/R4R_"
    elif dataset_type == "PretrainRxR":
        data_root = "srv/task_data/pretrain_data/RxR_"
    else:
        raise NotImplementedError

    combined_split = "_".join(splits)
    path = f"{data_root}{combined_split}_preprocessed_{version}.pickle"
    if os.path.exists(path) and os.path.isfile(path):
        logger.info(f"Loading preprocessed data from {path}")
        t_s = time.time()
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        t_e = time.time()
        logger.info(
            "Loaded Image Features from {} in time: {:0.2f} mins".format(
                path, (t_e - t_s) / 60.0
            )
        )
        return data
    return False


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


# def load_per_view_img_pickle_features(img_feat_dir, img_feature_file):
#     t_s = time.time()

#     img_feature_path = os.path.join(img_feat_dir, img_feature_file)

#     logger.info(f"Loading Image Features from {img_feature_path}")

#     with open(img_feature_path, "rb") as f:
#         loaded_feature_data = pickle.load(f)

#     image_w = loaded_feature_data[0]["image_w"]
#     image_h = loaded_feature_data[0]["image_h"]
#     vfov = loaded_feature_data[0]["vfov"]

#     features = {"features": {}, "image_w": image_w, "image_h": image_h, "vfov": vfov}
#     region_tokens = {}

#     for item in loaded_feature_data:
#         long_id = f"{item['scanId']}_{item['viewpointId']}_{item['featureViewIndex']}"

#         features["features"][long_id] = item["features"]

#         region_tokens[long_id] = item["region_tokens"]

#     t_e = time.time()
#     logger.info(
#         "Loaded Image Features from {} in time: {:0.2f} mins".format(
#             img_feature_path, (t_e - t_s) / 60.0
#         )
#     )

#     return features, region_tokens


# def load_img_pickle_features(img_feat_dir, img_feature_file, candidate_feature_file):
#     t_s = time.time()

#     img_feature_path = os.path.join(img_feat_dir, img_feature_file)

#     logger.info(f"Loading Image Features from {img_feature_path}")

#     with open(img_feature_path, "rb") as f:
#         loaded_feature_data = pickle.load(f)

#     image_w = loaded_feature_data[0]["image_w"]
#     image_h = loaded_feature_data[0]["image_h"]
#     vfov = loaded_feature_data[0]["vfov"]

#     features = {"features": {}, "image_w": image_w, "image_h": image_h, "vfov": vfov}
#     region_tokens = {}

#     for item in loaded_feature_data:
#         long_id = f"{item['scanId']}_{item['viewpointId']}"

#         features["features"][long_id] = item["features"]

#         region_tokens[long_id] = item["region_tokens"]

#     t_e = time.time()
#     logger.info(
#         "Loaded Image Features from {} in time: {:0.2f} mins".format(
#             img_feature_path, (t_e - t_s) / 60.0
#         )
#     )

#     candidate_feature_path = os.path.join(img_feat_dir, candidate_feature_file)

#     logger.info(f"Loading Candidate Image Features from {candidate_feature_path}")

#     with open(candidate_feature_path, "rb") as f:
#         candidate_feature_data = pickle.load(f)

#     cand_features = {}

#     for item in candidate_feature_data:
#         long_id = f"{item['scanId']}_{item['viewpointId']}"

#         cand_features[long_id] = item["features"]

#     t_e = time.time()
#     logger.info(
#         "Loaded Candidate Image Features from {} in time: {:0.2f} mins".format(
#             candidate_feature_path, (t_e - t_s) / 60.0
#         )
#     )

#     return features, region_tokens, cand_features


def read_tsv_img_features(path=None, feature_size=2048, blind=False):
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


def load_detector_classes(UPDOWN_DATA="srv/detector_classes_attributes"):
    classes = ["__background__"]
    with open(os.path.join(UPDOWN_DATA, "objects_vocab.txt")) as f:
        for object in f.readlines():
            classes.append(object.split(",")[0].lower().strip())
    return classes


class FeaturesReader:
    def __init__(self, path, use_lmdb=True, in_memory=False):
        self.use_lmdb = use_lmdb

        if not self.use_lmdb:
            (
                self.keys,
                self.features,
                self.region_tokens,
                self.image_w,
                self.image_h,
                self.vfov,
            ) = self.load_features_from_pickle(path)
        else:
            img_feature_path = path + ".lmdb"
            logger.info(f"Loading lmdb features from {img_feature_path}")
            # open database
            self.env = lmdb.open(
                img_feature_path,
                readonly=True,
                readahead=False,
                max_readers=1,
                lock=False,
            )

            # get keys
            with self.env.begin(write=False) as txn:
                self.keys = pickle.loads(txn.get("keys".encode()))

            key = self.keys[0]
            with self.env.begin(write=False) as txn:
                item = pickle.loads(txn.get(key))
                self.image_w = item["image_w"]
                self.image_h = item["image_h"]
                self.vfov = item["vfov"]

            region_labels_path = path + "-region_labels.pickle"

            with open(region_labels_path, "rb") as handle:
                self.region_tokens = pickle.load(handle)
            logger.info(f"Loaded region labels from {region_labels_path}")

        # get viewpoints
        self.viewpoints = {}
        for key in self.keys:
            scan_id, viewpoint_id, feature_view_index = key.decode().split("_")
            if scan_id not in self.viewpoints:
                self.viewpoints[scan_id] = set()
            self.viewpoints[scan_id].add(viewpoint_id)

        # # initialize memory
        # self._in_memory = in_memory
        # if self._in_memory:
        #     self.indices = set()
        #     self.features = [None] * len(self.keys)
        #     self.headings = [None] * len(self.keys)
        #     self.elevations = [None] * len(self.keys)

    def load_features_from_pickle(self, path):
        t_s = time.time()
        img_feature_path = path + ".pickle"
        logger.info(f"Loading Image Features from {img_feature_path}")

        with open(img_feature_path, "rb") as f:
            loaded_feature_data = pickle.load(f)

        image_w = loaded_feature_data[0]["image_w"]
        image_h = loaded_feature_data[0]["image_h"]
        vfov = loaded_feature_data[0]["vfov"]

        keys = []
        features = {}
        region_tokens = {}

        for item in loaded_feature_data:
            long_id = (
                f"{item['scanId']}_{item['viewpointId']}_{item['featureViewIndex']}"
            ).encode()

            features[long_id] = item["features"]
            region_tokens[long_id] = item["region_tokens"]
            keys.append(long_id)

        t_e = time.time()
        logger.info(
            "Loaded Image Features from {} in time: {:0.2f} mins".format(
                img_feature_path, (t_e - t_s) / 60.0
            )
        )
        return keys, features, region_tokens, image_w, image_h, vfov

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key):
        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")
        if self.use_lmdb:
            # load from disk
            with self.env.begin(write=False) as txn:
                item = pickle.loads(txn.get(key))
            return item["features"]
        else:
            return self.features[key]

    def get_region_tokens(self, key):
        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")
        return self.region_tokens[key]


def get_encoding_for_oscar(tokenizer, obs):
    truncate_dialog = True

    use_oscar_settings = True

    TAR_BACK = False

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
    new_obs = []
    for item in obs:
        instruction = item["instructions"]

        target = instruction.split("<TAR>")[1]
        rest = instruction.split("<TAR>")[0]
        dialog_history = re.split("<NAV>|<ORA>", rest)
        dialog_history = [item for item in dialog_history if item != ""]

        token_target = tokenizer.tokenize(target)
        token_target = token_target[:MAX_TARGET_LENGTH]

        token_dialog_history = []
        for turn in dialog_history:
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
        segment_ids += [pad_token_segment_id] * (MAX_SEQ_LENGTH - len(segment_ids) - 1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        new_obs.append({"instr_encoding": token_ids, "segment_ids": segment_ids})

    # "tokens": tokens

    return new_obs
