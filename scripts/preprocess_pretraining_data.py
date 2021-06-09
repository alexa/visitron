import os
from itertools import repeat
from multiprocessing import Pool

import torch
from tqdm import tqdm

from oscar.transformers_src.pytorch_transformers import BertConfig, BertTokenizer
from utils_data import (
    FeaturesReader,
    check_and_load_preprocessed_data,
    load_datasets,
    save_preprocessed_data,
    truncate_dialogs,
)

model_name_or_path = "srv/oscar_weights/base-vg-labels/ep_107_1192087"
img_feat_dir = "srv/img_features"
img_feature_file = "ResNet-101-faster-rcnn-genome-worientation.lmdb"
splits = ["train"]

tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path,
    do_lower_case=True,
)
tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

img_feature_path = os.path.join(img_feat_dir, img_feature_file)
features_reader = FeaturesReader(
    path=img_feature_path,
    in_memory=False,
)

truncate_dialog = True

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


def _extract_region_labels(scan_id, viewpoint_id, MAX_REGION_LABELS_LENGTH):
    region_labels = []
    for view_idx in range(36):
        long_id = f"{scan_id}_{viewpoint_id}_{view_idx}"
        region_label = features_reader.get_region_tokens(long_id.encode())[:5]
        region_labels.extend(region_label)

    region_labels = set(region_labels)
    region_labels = " ".join(region_labels)
    token_region_labels = tokenizer.tokenize(region_labels)
    token_region_labels = token_region_labels[-MAX_REGION_LABELS_LENGTH:]
    return token_region_labels


def preprocess_data(start_index, total_jobs):
    version = start_index
    r4r_data = []
    for item in tqdm(
        load_datasets(splits, dataset_type="PretrainR4R")[start_index::total_jobs],
        miniters=1000,
        desc="loading PretrainR4R",
    ):

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

        new_item["region_tokens"] = _extract_region_labels(
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
        segment_ids += [pad_token_segment_id] * (MAX_SEQ_LENGTH - len(segment_ids) - 1)

        new_item["target_dialog_tokens"] = tokens
        new_item["target_dialog_tokens_id"] = tokenizer.convert_tokens_to_ids(tokens)
        new_item["target_dialog_tokens_id"] = torch.LongTensor(
            new_item["target_dialog_tokens_id"]
        )
        new_item["target_dialog_segment_ids"] = segment_ids

        r4r_data.append(new_item)

    save_preprocessed_data(r4r_data, splits, version, dataset_type="PretrainR4R")


total_jobs = 56

processes = range(total_jobs)

with Pool(processes=total_jobs) as pool:
    pool.starmap(
        preprocess_data,
        zip(
            processes,
            repeat(total_jobs),
        ),
    )


data = []
for i in range(total_jobs):
    new_data = check_and_load_preprocessed_data(splits, i, dataset_type="PretrainR4R")
    data.extend(new_data)

save_preprocessed_data(data, splits, "v1", dataset_type="PretrainR4R")
