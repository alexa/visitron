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
import torch
from torch.utils.data import DataLoader, Dataset

from get_oscar_model import special_tokens_dict
from utils_data import load_datasets, load_nav_graphs, truncate_dialogs

logger = logging.getLogger(__name__)


angle_inc = np.pi / 6.0


def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex, 0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex, 96:] = np.cos(rel_elevation)
    return embedding


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)
]


class PretrainDataset(Dataset):
    def __init__(
        self,
        args,
        splits=["train"],
        feature_store=None,
        region_labels=None,
        tokenizer=None,
        truncate_dialog=False,
        add_ndh_data=True,
        add_r2r_data=False,
        add_r4r_data=False,
    ):
        super(PretrainDataset, self).__init__()

        assert tokenizer is not None
        assert (add_ndh_data or add_r2r_data) is True

        self.args = args
        self.tokenizer = tokenizer
        self.feature_store = feature_store
        self.region_labels = region_labels
        self.data = []

        use_oscar_settings = True

        cls_token_segment_id = 0
        pad_token_segment_id = 0
        sep_token_segment_id = 0

        tar_token_segment_id = 1
        ques_token_segment_id = 2
        ans_token_segment_id = 3

        MAX_SEQ_LENGTH = 512
        MAX_REGION_LABELS_LENGTH = 180 - 1
        MAX_DIALOG_LEN = 512 - 180 - 4  # including [QUES]s and [ANS]s
        MAX_TARGET_LENGTH = 4 - 2  # [CLS], [TAR], [SEP] after QA and before Action
        # # TODO: ^^ add them as args ^^

        if add_ndh_data:
            for item in load_datasets(splits, dataset_type="PretrainNDH"):

                new_item = dict(item)
                new_item["inst_idx"] = f"{item['inst_idx']}"

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

                new_item["region_tokens"] = self._extract_region_labels(
                    item["scan"], item["viewpoint"], MAX_REGION_LABELS_LENGTH
                )

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

                tokens += [sep_token] + token_target
                segment_ids += [tar_token_segment_id] * (len(token_target) + 1)

                tokens += new_item["region_tokens"]
                segment_ids += [sep_token_segment_id] * len(new_item["region_tokens"])

                tokens += [tokenizer.sep_token]
                segment_ids += [sep_token_segment_id]

                tokens += [tokenizer.pad_token] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                segment_ids += [pad_token_segment_id] * (
                    MAX_SEQ_LENGTH - len(segment_ids) - 1
                )

                new_item["target_dialog_tokens"] = tokens
                new_item["target_dialog_tokens_id"] = tokenizer.convert_tokens_to_ids(
                    tokens
                )
                new_item["target_dialog_tokens_id"] = torch.LongTensor(
                    new_item["target_dialog_tokens_id"]
                )
                new_item["target_dialog_segment_ids"] = segment_ids

                self.data.append(new_item)
        if add_r2r_data:
            for item in load_datasets(splits, dataset_type="PretrainR2R"):

                # for j, instr in enumerate(item["instructions"]):

                new_item = dict(item)
                new_item["inst_idx"] = f"{item['inst_idx']}"

                token_turn = tokenizer.tokenize(new_item["dialog_history"])
                token_dialog_history = [token_turn]

                if truncate_dialog:
                    # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                    token_dialog_history = truncate_dialogs(
                        token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                    )

                new_item["tokens_dialog_history"] = token_dialog_history

                new_item["region_tokens"] = self._extract_region_labels(
                    item["scan"], item["viewpoint"], MAX_REGION_LABELS_LENGTH
                )

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

                tokens += new_item["region_tokens"]
                segment_ids += [sep_token_segment_id] * len(new_item["region_tokens"])

                tokens += [tokenizer.sep_token]
                segment_ids += [sep_token_segment_id]

                tokens += [tokenizer.pad_token] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                segment_ids += [pad_token_segment_id] * (
                    MAX_SEQ_LENGTH - len(segment_ids) - 1
                )

                new_item["target_dialog_tokens"] = tokens
                new_item["target_dialog_tokens_id"] = tokenizer.convert_tokens_to_ids(
                    tokens
                )
                new_item["target_dialog_tokens_id"] = torch.LongTensor(
                    new_item["target_dialog_tokens_id"]
                )
                new_item["target_dialog_segment_ids"] = segment_ids

                self.data.append(new_item)

        if add_r4r_data:
            for item in load_datasets(splits, dataset_type="PretrainR4R"):

                new_item = dict(item)
                new_item["inst_idx"] = f"{item['inst_idx']}"

                token_turn = tokenizer.tokenize(new_item["dialog_history"])
                token_dialog_history = [token_turn]

                if truncate_dialog:
                    # max_seq_length - 4 as accounting for [CLS], [TAR], Target, [SEP]
                    token_dialog_history = truncate_dialogs(
                        token_dialog_history, amount=MAX_DIALOG_LEN, left=True
                    )

                new_item["tokens_dialog_history"] = token_dialog_history

                new_item["region_tokens"] = self._extract_region_labels(
                    item["scan"], item["viewpoint"], MAX_REGION_LABELS_LENGTH
                )

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

                tokens += new_item["region_tokens"]
                segment_ids += [sep_token_segment_id] * len(new_item["region_tokens"])

                tokens += [tokenizer.sep_token]
                segment_ids += [sep_token_segment_id]

                tokens += [tokenizer.pad_token] * (MAX_SEQ_LENGTH - len(tokens) - 1)
                segment_ids += [pad_token_segment_id] * (
                    MAX_SEQ_LENGTH - len(segment_ids) - 1
                )

                new_item["target_dialog_tokens"] = tokens
                new_item["target_dialog_tokens_id"] = tokenizer.convert_tokens_to_ids(
                    tokens
                )
                new_item["target_dialog_tokens_id"] = torch.LongTensor(
                    new_item["target_dialog_tokens_id"]
                )
                new_item["target_dialog_segment_ids"] = segment_ids

                self.data.append(new_item)

        self.splits = splits

        logger.info(
            "PretrainDataset loaded with %d instructions, using splits: %s NDH: %r R2R: %r R4R: %r"
            % (
                len(self.data),
                ",".join(splits),
                add_ndh_data,
                add_r2r_data,
                add_r4r_data,
            )
        )

    def _extract_region_labels(self, scan_id, viewpoint_id, MAX_REGION_LABELS_LENGTH):
        region_labels = []
        for view_idx in range(36):
            long_id = f"{scan_id}_{viewpoint_id}_{view_idx}"
            if self.args.debug:
                region_label = ["wall"] * 5
            else:
                region_label = self.region_labels[long_id][:5]
            region_labels.extend(region_label)

        region_labels = set(region_labels)
        region_labels = " ".join(region_labels)
        token_region_labels = self.tokenizer.tokenize(region_labels)
        token_region_labels = token_region_labels[-MAX_REGION_LABELS_LENGTH:]
        return token_region_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        output = self._preprocess_item(item)
        return {
            key: (torch.tensor(value) if not isinstance(value, torch.Tensor) else value)
            for key, value in output.items()
        }

    def _mask_tokens(self, inputs):

        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        special_tokens_mask = [
            val in self.tokenizer.all_special_ids for val in labels.tolist()
        ]
        att_mask = [val == self.tokenizer.pad_token_id for val in labels.tolist()]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        # masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).type(torch.ByteTensor)
        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)

        attention_mask = torch.full(labels.shape, 1, dtype=torch.bool).masked_fill_(
            torch.tensor(att_mask, dtype=torch.bool), value=0
        )
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool)
            & masked_indices
        )

        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word

        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool)
            & masked_indices
            & ~indices_replaced
        )

        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels, attention_mask

    def _extract_img_features(self, scan_id, viewpoint_id, view_index):
        img_features = []
        view_indices = []
        region_labels = []
        for idx in range(36):
            long_id = f"{scan_id}_{viewpoint_id}_{idx}"
            if self.args.debug:
                feature = torch.rand(5, 2054)
            else:
                feature = self.feature_store["features"][long_id][:5]
            img_features.append(feature)
            view_indices.extend([idx] * feature.shape[0])
        img_features = np.concatenate(img_features, axis=0)
        # img_features = img_features[-MAX_IMG_FEATURES_LENGTH:]

        location_embeddings = []
        for idx in view_indices:
            embed = _static_loc_embeddings[view_index][np.newaxis, idx]
            location_embeddings.append(embed)
        location_embeddings = np.concatenate(location_embeddings, axis=0)

        return img_features, location_embeddings

    def _preprocess_item(self, item):
        inputs, labels, attention_mask = self._mask_tokens(
            item["target_dialog_tokens_id"]
        )

        attention_mask = attention_mask.tolist()

        img_features, location_embeddings = self._extract_img_features(
            item["scan"], item["viewpoint"], item["current_view_index"]
        )
        target_view_index = item["target_rel_view_index"]

        img_features = torch.from_numpy(img_features)
        location_embeddings = torch.from_numpy(location_embeddings)

        if img_features.shape[0] > self.args.max_img_seq_length:
            img_features = img_features[
                -self.args.max_img_seq_length :,
            ]
            location_embeddings = location_embeddings[
                -self.args.max_img_seq_length :,
            ]
            if self.args.max_img_seq_length > 0:
                attention_mask = attention_mask + [1] * img_features.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.args.max_img_seq_length > 0:
                attention_mask = attention_mask + [1] * img_features.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros(
                (
                    self.args.max_img_seq_length - img_features.shape[0],
                    img_features.shape[1],
                ),
                dtype=img_features.dtype,
            )
            location_embed_padding_matrix = torch.zeros(
                (
                    self.args.max_img_seq_length - img_features.shape[0],
                    location_embeddings.shape[1],
                ),
                dtype=location_embeddings.dtype,
            )
            img_features = torch.cat((img_features, padding_matrix), dim=0)
            location_embeddings = torch.cat(
                (location_embeddings, location_embed_padding_matrix), dim=0
            )
            if self.args.max_img_seq_length > 0:
                attention_mask = attention_mask + [0] * padding_matrix.shape[0]

        labels = labels.tolist() + [-1] * img_features.shape[0]
        labels = torch.LongTensor(labels)

        output = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
            "img_feats": img_features,
            "img_location_embeddings": location_embeddings,
            "next_action": target_view_index,
        }
        return output
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            (location_embeddings, location_embed_padding_matrix), dim=0
            )
            if self.args.max_img_seq_length > 0:
                attention_mask = attention_mask + [0] * padding_matrix.shape[0]

        labels = labels.tolist() + [-1] * img_features.shape[0]
        labels = torch.LongTensor(labels)

        output = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
            "img_feats": img_features,
            "img_location_embeddings": location_embeddings,
            "next_action": target_view_index,
        }
        return output
