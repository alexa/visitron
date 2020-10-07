# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import cv2
import json
import math
import base64
import csv
import sys
import os
import random
from multiprocessing import Pool

import argparse

from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torchvision.models
import torchvision.transforms.functional as F

import MatterSim

from timer import Timer

sys.path.insert(0, "/root/mount/Matterport3DSimulator/models/")
import bit_pytorch.models as bit_models

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()

parser.add_argument("--num-gpus", type=int, default=1, help="")
parser.add_argument(
    "--model",
    type=str,
    default="ResNet-152",
    choices=[
        "ResNet-152",
        "BiT-M-R50x1",
        "BiT-M-R50x3",
        "BiT-M-R50x1",
        "BiT-M-R101x1",
        "BiT-M-R101x3",
        "BiT-M-R152x4",
        "BiT-S-R50x1",
        "BiT-S-R50x3",
        "BiT-S-R101x1",
        "BiT-S-R101x3",
        "BiT-S-R152x4",
    ],
)
parser.add_argument("--img-features-dir", type=str, default="img_features")
parser.add_argument("--models-dir", type=str, default="models/")
parser.add_argument("--output-feature-file", type=str, default="")
parser.add_argument("--seed", type=int, default=1, help="")
parser.add_argument("--batch-size", type=int, default=12, help="")

args = parser.parse_args()

FEATURE_SIZES = {
    "ResNet-152": 2048,
    "BiT-M-R50x1": 2048,
    "BiT-M-R50x3": 6144,
    "BiT-M-R101x1": 2048,
    "BiT-M-R101x3": 6144,
    "BiT-M-R152x4": 8192,
    "BiT-S-R50x1": 2048,
    "BiT-S-R50x3": 6144,
    "BiT-S-R101x1": 2048,
    "BiT-S-R101x3": 6144,
    "BiT-S-R152x4": 8192,
}

NUM_GPUS = args.num_gpus
MODEL_NAME = args.model
FEATURE_SIZE = FEATURE_SIZES[MODEL_NAME]
BATCH_SIZE = (
    args.batch_size
)  # Some fraction of viewpoint size - batch size 4 equals 11GB memory

if args.output_feature_file == "":
    OUTFILE = "%s-imagenet-pytorch.tsv" % MODEL_NAME
else:
    OUTFILE = args.output_feature_file

OUTFILE = os.path.join(args.img_features_dir, OUTFILE)
MERGED = OUTFILE

if NUM_GPUS != 1:
    OUTFILE = OUTFILE + ".%d"

MODELS = args.models_dir

GRAPHS = "connectivity/"

SEED = args.seed
print("SEED: %d" % SEED)

# --------------------------------------------
# --------------------------------------------

TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
GPU_ID = 0

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def load_model(model_name):
    if model_name == "ResNet-152":
        resnet_full = torchvision.models.resnet152(pretrained=True)
        resnet = nn.Sequential(*list(resnet_full.children())[:-1])
        return resnet
    elif "BiT" in model_name:  # BiT-M-R50x1
        model = bit_models.KNOWN_MODELS[model_name](head_size=1000, zero_head=True)
        model.load_from(np.load(MODELS + model_name + ".npz"))

        all_layers = list(model.children())
        main_stack = all_layers[:-1]
        last_layer_wihout_fc = all_layers[-1][:-1]
        model_without_fc = main_stack + [last_layer_wihout_fc]
        bit = nn.Sequential(*model_without_fc)
        return bit


def load_viewpointids(gpu_id=0):
    viewpointIds = []
    with open(GRAPHS + "scans.txt") as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    random.seed(SEED)
    random.shuffle(viewpointIds)
    if NUM_GPUS != 1:
        viewpointIds = viewpointIds[gpu_id::NUM_GPUS]
    print("%d: Loaded %d viewpoints" % (gpu_id, len(viewpointIds)))
    return viewpointIds


def transform_img_resnet(im):
    """ Prep opencv 3 channel image for the network """
    np_im = np.array(im, copy=True).astype(np.float32)
    np_im = np_im[..., ::-1]

    np_im = np_im.transpose((2, 0, 1))  # (3, H, W)
    np_im = np.ascontiguousarray(np_im, dtype=np.float32)
    im = torch.from_numpy(np_im)
    im /= 255.0
    return F.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def transform_img_bit(im):
    np_im = np.array(im, copy=True).astype(np.float32)
    np_im = np_im[..., ::-1]

    np_im = np_im.transpose((2, 0, 1))  # (3, H, W)
    np_im = np.ascontiguousarray(np_im, dtype=np.float32)
    im = torch.from_numpy(np_im)
    im /= 255.0
    return F.normalize(im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def build_tsv(gpu_id=0):
    print("%d: build_tsv" % gpu_id)

    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    with torch.no_grad():

        device = torch.device("cuda:%d" % gpu_id)

        model = load_model(MODEL_NAME).to(device)
        model.eval()

        count = 0
        t_render = Timer()
        t_net = Timer()

        if NUM_GPUS == 1:
            output_file = OUTFILE
        else:
            output_file = OUTFILE % gpu_id

        with open(output_file, "wt") as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)

            # Loop all the viewpoints in the simulator
            viewpointIds = load_viewpointids(gpu_id)

            for scanId, viewpointId in viewpointIds:
                t_render.tic()
                # Loop all discretized views from this location
                blobs = []
                features = np.empty([VIEWPOINT_SIZE, FEATURE_SIZE], dtype=np.float32)

                for ix in range(VIEWPOINT_SIZE):
                    if ix == 0:
                        sim.newEpisode(
                            [scanId], [viewpointId], [0], [math.radians(-30)]
                        )
                    elif ix % 12 == 0:
                        sim.makeAction([0], [1.0], [1.0])
                    else:
                        sim.makeAction([0], [1.0], [0])

                    state = sim.getState()[0]
                    assert state.viewIndex == ix

                    # Transform and save generated image
                    if "ResNet" in MODEL_NAME:
                        transformed_im = transform_img_resnet(state.rgb)
                    elif "BiT" in MODEL_NAME:
                        transformed_im = transform_img_bit(state.rgb)
                    blobs.append(transformed_im)

                t_render.toc()
                t_net.tic()

                # Run as many forward passes as necessary
                assert VIEWPOINT_SIZE % BATCH_SIZE == 0
                forward_passes = VIEWPOINT_SIZE // BATCH_SIZE

                ix = 0
                data = torch.empty(
                    (BATCH_SIZE, 3, HEIGHT, WIDTH), dtype=torch.float32, device=device
                )

                for f in range(forward_passes):
                    for n in range(BATCH_SIZE):
                        # Copy image blob to the net
                        data[n, :, :, :] = blobs[ix]
                        ix += 1
                    # Forward pass
                    features[f * BATCH_SIZE : (f + 1) * BATCH_SIZE, :] = (
                        model(data).squeeze().cpu().detach().numpy()
                    )
                writer.writerow(
                    {
                        "scanId": scanId,
                        "viewpointId": viewpointId,
                        "image_w": WIDTH,
                        "image_h": HEIGHT,
                        "vfov": VFOV,
                        "features": str(base64.b64encode(features), "utf-8"),
                    }
                )
                count += 1
                t_net.toc()
                if count % 100 == 0:
                    print(
                        "Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours"
                        % (
                            count,
                            len(viewpointIds),
                            t_render.average_time,
                            t_net.average_time,
                            (t_render.average_time + t_net.average_time)
                            * len(viewpointIds)
                            / 3600,
                        )
                    )


def merge_tsvs():
    test = [OUTFILE % i for i in range(NUM_GPUS)]
    with open(MERGED, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for infile in test:
            print(infile)
            with open(infile, "rt") as tsv_in_files:
                reader = csv.DictReader(
                    tsv_in_files, delimiter="\t", fieldnames=TSV_FIELDNAMES
                )
                for item in reader:
                    try:
                        writer.writerow(item)
                    except Exception as e:
                        print(e)
                        print(item["image_id"])


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item["scanId"] = item["scanId"]
            item["image_h"] = int(item["image_h"])
            item["image_w"] = int(item["image_w"])
            item["vfov"] = int(item["vfov"])
            item["features"] = np.frombuffer(
                base64.b64decode(item["features"]), dtype=np.float32
            ).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


if __name__ == "__main__":

    if NUM_GPUS == 1:
        build_tsv()
        data = read_tsv(OUTFILE)
        print("Completed %d viewpoints" % len(data))
    else:
        gpu_ids = range(NUM_GPUS)
        p = Pool(NUM_GPUS)
        p.map(build_tsv, gpu_ids)
        merge_tsvs()
    data = read_tsv(MERGED)
    print("Completed %d viewpoints" % len(data))
