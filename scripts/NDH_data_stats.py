# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json


def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open("tasks/NDH/data/%s.json" % split) as f:
            data += json.load(f)
    return data


train_data = load_datasets(["train"])
val_seen_data = load_datasets(["val_seen"])
val_unseen_data = load_datasets(["val_unseen"])

for supervision in ["planner_path", "player_path"]:
    train_lengths = []
    val_seen_lengths = []
    val_unseen_lengths = []

    for item in train_data:
        l = len(item[supervision])
        train_lengths.append(l)

    print(
        f"Train data {supervision} Min: {min(train_lengths)} Max: {max(train_lengths)} Mean: {sum(train_lengths)/len(train_lengths)}"
    )

    for item in val_seen_data:
        l = len(item[supervision])
        val_seen_lengths.append(l)

    print(
        f"Val Seen data {supervision} Min: {min(val_seen_lengths)} Max: {max(val_seen_lengths)} Mean: {sum(val_seen_lengths)/len(val_seen_lengths)}"
    )

    for item in val_seen_data:
        l = len(item[supervision])
        val_unseen_lengths.append(l)

    print(
        f"Val Unseen data {supervision} Min: {min(val_unseen_lengths)} Max: {max(val_unseen_lengths)} Mean: {sum(val_unseen_lengths)/len(val_unseen_lengths)}"
    )
