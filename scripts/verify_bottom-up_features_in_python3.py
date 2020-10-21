# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import csv
import sys
import base64
import time
import numpy as np
import pickle

csv.field_size_limit(sys.maxsize)


def load_features(feature_store):
    logger = print
    features = {}

    new_data = []

    if feature_store:
        logger("Loading image features from %s" % feature_store)

        tsv_fieldnames = [
            "scanId",
            "viewpointId",
            "image_w",
            "image_h",
            "vfov",
            "features",
            "region_tokens",
            "boxes",
            "cls_prob",
            "attr_prob",
            "featureViewIndex",
            "featureHeading",
            "featureElevation",
            "viewHeading",
            "viewElevation",
        ]
        features = {}
        with open(feature_store, "rt") as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter="\t", fieldnames=tsv_fieldnames
            )
            for item in reader:

                item["scanId"] = item["scanId"]
                item["viewpointId"] = item["viewpointId"]
                item["image_h"] = int(item["image_h"])
                item["image_w"] = int(item["image_w"])
                item["vfov"] = int(item["vfov"])

                item["boxes"] = np.frombuffer(
                    base64.b64decode(item["boxes"]), dtype=np.float32
                ).reshape((-1, 4))

                item["features"] = np.frombuffer(
                    base64.b64decode(item["features"]), dtype=np.float32
                ).reshape((-1, 2048))

                top_left_x = item["boxes"][:, 0]
                top_left_y = item["boxes"][:, 1]
                bottom_right_x = item["boxes"][:, 2]
                bottom_right_y = item["boxes"][:, 3]

                region_width = bottom_right_x - top_left_x + 1
                region_height = bottom_right_y - top_left_y + 1

                norm_top_left_x = top_left_x / item["image_w"]
                norm_top_left_y = top_left_y / item["image_h"]
                norm_bottom_right_x = bottom_right_x / item["image_w"]
                norm_bottom_right_y = bottom_right_y / item["image_h"]
                norm_region_width = region_width / item["image_w"]
                norm_region_height = region_height / item["image_h"]

                orientation_feature = np.concatenate(
                    [
                        norm_top_left_x[:, np.newaxis],
                        norm_top_left_y[:, np.newaxis],
                        norm_bottom_right_x[:, np.newaxis],
                        norm_bottom_right_y[:, np.newaxis],
                        norm_region_width[:, np.newaxis],
                        norm_region_height[:, np.newaxis],
                    ],
                    axis=1,
                )

                # item["features"] = np.concatenate(
                #     [item["features"], orientation_feature], axis=1
                # )

                item["region_tokens"] = eval(item["region_tokens"])

                # item["viewHeading"] = float(item["viewHeading"])
                item["viewHeading"] = np.frombuffer(
                    base64.b64decode(item["viewHeading"]), dtype=np.float32
                )

                # item["viewElevation"] = float(item["viewElevation"])
                item["viewElevation"] = np.frombuffer(
                    base64.b64decode(item["viewElevation"]), dtype=np.float32
                )

                item["featureHeading"] = np.frombuffer(
                    base64.b64decode(item["featureHeading"]), dtype=np.float32
                )
                item["featureElevation"] = np.frombuffer(
                    base64.b64decode(item["featureElevation"]), dtype=np.float32
                )
                item["cls_prob"] = np.frombuffer(
                    base64.b64decode(item["cls_prob"]), dtype=np.float32
                ).reshape((-1, 1601))
                item["attr_prob"] = np.frombuffer(
                    base64.b64decode(item["attr_prob"]), dtype=np.float32
                ).reshape((-1, 401))
                item["featureViewIndex"] = item["featureViewIndex"]

                long_id = (
                    item["scanId"]
                    + "_"
                    + item["viewpointId"]
                    # + "_"
                    # + item["featureViewIndex"]
                )

                features[long_id] = item["features"]
                new_data.append(item)

        feature_size = next(iter(features.values())).shape[-1]
        logger("The feature size is %d" % feature_size)
    else:
        logger("Image features not provided")

    return new_data, features


start = time.time()
new_data, features = load_features(
    # "img_features/ResNet-101-faster-rcnn-genome-pano.tsv"
    "img_features/ResNet-101-faster-rcnn-genome-candidate.tsv"
)
now = time.time()
print("Time taken for loading TSV: %0.4f mins" % ((now - start) / 60))
print(len(new_data))

start = time.time()
with open(
    # "img_features/ResNet-101-faster-rcnn-genome-pano-worientation.pickle", "wb"
    "img_features/ResNet-101-faster-rcnn-genome-candidate-worientation.pickle",
    "wb",
) as f:
    pickle.dump(new_data, f)
now = time.time()
print("Time taken for saving Pickle: %0.4f mins" % ((now - start) / 60))

start = time.time()
with open(
    # "img_features/ResNet-101-faster-rcnn-genome-pano-worientation.pickle", "rb"
    "img_features/ResNet-101-faster-rcnn-genome-candidate-worientation.pickle",
    "rb",
) as f:
    loaded_data = pickle.load(f)
now = time.time()
print("Time taken for loading Pickle: %0.4f mins" % ((now - start) / 60))


# start = time.time()
# features = {}
# region_tokens = {}

# for item in loaded_data:
#     long_id = f"{item['scanId']}_{item['viewpointId']}_{item['featureViewIndex']}"
#     features[long_id] = item["features"]
#     region_tokens[long_id] = item["region_tokens"]
# now = time.time()
# print("Time taken for creating dictionary: %0.4f mins" % ((now - start) / 60))

import pdb

pdb.set_trace()

