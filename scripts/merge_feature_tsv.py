# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import sys
import csv

csv.field_size_limit(sys.maxsize)

# OUTFILE = "img_features/ResNet-101-faster-rcnn-genome-pano/ResNet-101-faster-rcnn-genome-pano.tsv.%d.%d"
# MERGED = "img_features/ResNet-101-faster-rcnn-genome-pano.tsv"

OUTFILE = "img_features/ResNet-101-faster-rcnn-genome-candidate/ResNet-101-faster-rcnn-genome-candidate.tsv.%d.%d"
MERGED = "img_features/ResNet-101-faster-rcnn-genome-candidate.tsv"

NUM_GPUS = 8

TSV_FILENAMES = [
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


def merge_tsvs():
    test = [OUTFILE % (j, i) for i in range(NUM_GPUS) for j in range(10)]
    with open(MERGED, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FILENAMES)
        for infile in test:
            print(infile)
            with open(infile, "rt") as tsv_in_files:
                reader = csv.DictReader(
                    tsv_in_files, delimiter="\t", fieldnames=TSV_FILENAMES
                )
                for item in reader:
                    try:
                        writer.writerow(item)
                    except Exception as e:
                        print(e)
                        print(
                            item["scanId"],
                            item["viewpointId"],
                            item["featureViewIndex"],
                        )


if __name__ == "__main__":
    merge_tsvs()
