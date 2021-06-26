# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import base64
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

SEED = 1


random.seed(SEED)
csv.field_size_limit(sys.maxsize)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import MatterSim

caffe_root = "bottom-up"
sys.path.insert(0, caffe_root + "/caffe/python")
import caffe

sys.path.insert(0, caffe_root + "/lib")
sys.path.insert(0, caffe_root + "/lib/rpn")
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import _get_blobs, im_detect

from timer import Timer

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

# DRY_RUN = True
DRY_RUN = False

parser = argparse.ArgumentParser()

parser.add_argument("--num-gpus", type=int, default=8, help="")
parser.add_argument("--gpu-id", type=int, default=-1, help="")
args = parser.parse_args()
NUM_GPUS = args.num_gpus

FEATURE_SIZE = 2048

NUM_SWEEPS = 3
VIEW_PER_SWEEP = 12
VIEWPOINT_SIZE = NUM_SWEEPS * VIEW_PER_SWEEP
HEADING_INC = 360 / VIEW_PER_SWEEP
ANGLE_MARGIN = 5
ELEVATION_START = -30
ELEVATION_INC = 30

FEATURE_SIZE = 2048
PROTO = caffe_root + "/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt"
MODEL = caffe_root + "/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel"
CFG_FILE = caffe_root + "/experiments/cfgs/faster_rcnn_end2end_resnet.yml"
UPDOWN_DATA = caffe_root + "/data/genome/1600-400-20"

GRAPHS = "connectivity/"
OUTFILE = "img_features/ResNet-101-faster-rcnn-genome.tsv.%d"
MERGED = "img_features/ResNet-101-faster-rcnn-genome.tsv"
MERGED_PICKLE = "img_features/ResNet-101-faster-rcnn-genome.pickle"

WIDTH = 600
HEIGHT = 600
VFOV = 80
ASPECT = WIDTH / HEIGHT
HFOV = math.degrees(2 * math.atan(math.tan(math.radians(VFOV / 2)) * ASPECT))
FOC = (HEIGHT / 2) / math.tan(math.radians(VFOV / 2))  # focal length

MIN_LOCAL_BOXES = 1
MAX_LOCAL_BOXES = 20
MAX_TOTAL_BOXES = 10
NMS_THRESH = 0.3
CONF_THRESH = 0.4


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


def transform_img(im):
    """ Prep opencv BGR 3 channel image for the network """
    np_im = np.array(im, copy=True)
    return np_im


def visual_overlay(im, dets, classes, attributes):
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    objects = np.argmax(dets["cls_prob"][:, 1:], axis=1)
    obj_conf = np.max(dets["cls_prob"][:, 1:], axis=1)

    attr_threshold = 0.1
    attr = np.argmax(dets["attr_prob"][:, 1:], axis=1)
    attr_conf = np.max(dets["attr_prob"][:, 1:], axis=1)

    boxes = dets["boxes"]

    for i in range(len(dets["cls_prob"])):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i] + 1]
        if attr_conf[i] > attr_threshold:
            cls = attributes[attr[i] + 1] + " " + cls
        cls += " %.2f" % obj_conf[i]
        plt.gca().add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
                alpha=0.5,
            )
        )
        plt.gca().text(
            bbox[0],
            bbox[1] - 2,
            "%s" % cls,
            bbox=dict(facecolor="blue", alpha=0.5),
            fontsize=10,
            color="white",
        )
    return fig


def get_detections_from_im(record, net, im, conf_thresh=CONF_THRESH):
    scores, boxes, attr_scores, _ = im_detect(net, im)

    rois = net.blobs["rois"].data.copy()

    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs["cls_prob"].data
    attr_prob = net.blobs["attr_prob"].data
    pool5 = net.blobs["pool5_flat"].data

    max_conf = np.zeros((rois.shape[0]))

    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, NMS_THRESH))
        max_conf[keep] = np.where(
            cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
        )

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_LOCAL_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_LOCAL_BOXES]
    elif len(keep_boxes) > MAX_LOCAL_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_LOCAL_BOXES]

    center_x = 0.5 * (cls_boxes[:, 0] + cls_boxes[:, 2])
    center_y = 0.5 * (cls_boxes[:, 1] + cls_boxes[:, 3])

    heading = record["viewHeading"]
    elevation = record["viewElevation"]

    featureHeading = heading + np.arctan2(center_x[keep_boxes] - WIDTH / 2, FOC)
    featureHeading = np.mod(featureHeading, math.pi * 2)
    featureHeading = np.expand_dims(
        np.mod(featureHeading + math.pi * 2, math.pi * 2), axis=1
    )
    featureHeading = np.where(
        featureHeading > math.pi, featureHeading - math.pi * 2, featureHeading
    )
    featureElevation = np.expand_dims(
        elevation + np.arctan2(-center_y[keep_boxes] + HEIGHT / 2, FOC), axis=1
    )

    record["boxes"] = cls_boxes[keep_boxes]
    record["cls_prob"] = cls_prob[keep_boxes]
    record["attr_prob"] = attr_prob[keep_boxes]
    record["featureHeading"] = featureHeading
    record["featureElevation"] = featureElevation

    record["features"] = pool5[keep_boxes]

    return record


def filter(record, max_boxes):
    feat_dist = pairwise_distances(record["features"], metric="cosine")
    heading_diff = pairwise_distances(record["featureHeading"], metric="euclidean")
    heading_diff = np.minimum(heading_diff, 2 * math.pi - heading_diff)
    elevation_diff = pairwise_distances(record["featureElevation"], metric="euclidean")
    feat_dist = feat_dist + heading_diff + elevation_diff

    feat_dist += 10 * np.identity(feat_dist.shape[0], dtype=np.float32)
    feat_dist[np.triu_indices(feat_dist.shape[0])] = 10.0
    ind = np.unravel_index(np.argsort(feat_dist, axis=None), feat_dist.shape)

    keep = set(range(feat_dist.shape[0]))
    ix = 0

    while len(keep) > max_boxes:
        i = ind[0][ix]
        j = ind[1][ix]
        if i not in keep or j not in keep:
            ix += 1
            continue
        if record["cls_prob"][i, 1:].max() > record["cls_prob"][j, 1:].max():
            keep.remove(j)
        else:
            keep.remove(i)
        ix += 1

    for k, v in record.items():
        if k in [
            "boxes",
            "cls_prob",
            "attr_prob",
            "features",
            "featureHeading",
            "featrueElevation",
        ]:
            record[k] = v[sorted(keep)]


def extract_region_tokens(record, classes, attributes):
    objects = np.argmax(record["cls_prob"][:, 1:], axis=1)

    attr_threshold = 0.1
    attr = np.argmax(record["attr_prob"][:, 1:], axis=1)
    attr_conf = np.max(record["attr_prob"][:, 1:], axis=1)

    boxes = record["boxes"]

    region_tokens = []

    for i in range(len(record["cls_prob"])):
        cls = classes[objects[i] + 1]
        if attr_conf[i] > attr_threshold:
            cls = attributes[attr[i] + 1] + " " + cls
        region_tokens.append(cls)

    record["region_tokens"] = region_tokens


def load_classes():
    classes = ["__background__"]
    with open(os.path.join(UPDOWN_DATA, "objects_vocab.txt")) as f:
        for object in f.readlines():
            classes.append(object.split(",")[0].lower().strip())

    attributes = ["__no_attribute__"]
    with open(os.path.join(UPDOWN_DATA, "attributes_vocab.txt")) as f:
        for att in f.readlines():
            attributes.append(att.split(",")[0].lower().strip())
    return classes, attributes


def build_tsv(gpu_id=0):
    print("%d: build_tsv" % gpu_id)

    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    cfg_from_file(CFG_FILE)
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(PROTO, caffe.TEST, weights=MODEL)
    classes, attributes = load_classes()

    count = 0
    t_render = Timer()
    t_net = Timer()
    with open(OUTFILE % gpu_id, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FILENAMES)

        viewpointIds = load_viewpointids(gpu_id)

        for scanId, viewpointId in viewpointIds:
            t_net.tic()

            ims = []
            sim.newEpisode(
                [scanId], [viewpointId], [0], [math.radians(ELEVATION_START)]
            )
            for ix in range(VIEWPOINT_SIZE):
                state = sim.getState()[0]

                record = {
                    "scanId": state.scanId,
                    "viewpointId": state.location.viewpointId,
                    "featureViewIndex": ix,
                    "viewHeading": state.heading,
                    "viewElevation": state.elevation,
                    "image_h": HEIGHT,
                    "image_w": WIDTH,
                    "vfov": VFOV,
                }

                im = transform_img(state.rgb)

                record = get_detections_from_im(record, net, im)

                if DRY_RUN:
                    print(
                        "%d: Detected %d objects in an image"
                        % (gpu_id, record["features"].shape[0])
                    )

                filter(record, MAX_TOTAL_BOXES)

                extract_region_tokens(record, classes, attributes)

                if DRY_RUN:
                    print(
                        "%d: Reduced to %d objects in an image"
                        % (gpu_id, record["features"].shape[0])
                    )

                    fig = visual_overlay(im, record, classes, attributes)
                    fig.savefig(
                        "img_features/examples/%s-%s-%d.png"
                        % (
                            record["scanId"],
                            record["viewpointId"],
                            record["featureViewIndex"],
                        )
                    )
                    print(
                        "Saved %s-%s-%s"
                        % (
                            record["scanId"],
                            record["viewpointId"],
                            record["featureViewIndex"],
                        )
                    )
                    plt.close()

                for k, v in record.items():
                    if isinstance(v, np.ndarray):
                        record[k] = str(base64.b64encode(v)).encode("utf-8")

                import pdb

                pdb.set_trace()
                writer.writerow(record)

                elev = 0.0
                heading_chg = math.pi * 2 / VIEW_PER_SWEEP
                view = ix % VIEW_PER_SWEEP
                sweep = ix // VIEW_PER_SWEEP
                if view + 1 == VIEW_PER_SWEEP:
                    elev = math.radians(ELEVATION_INC)
                sim.makeAction([0], [heading_chg], [elev])

            count += 1
            t_net.toc()

            if count % 10 == 0:
                print(
                    "%d: Processed %d / %d viewpoints, %.1fs avg time, projected %.1f hours"
                    % (
                        gpu_id,
                        count,
                        len(viewpointIds),
                        t_net.average_time,
                        (t_net.average_time * len(viewpointIds) / 3600),
                    )
                )


def merge_tsvs():
    test = [OUTFILE % i for i in range(NUM_GPUS)]
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


def read_tsv(infile):
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FILENAMES)
        for item in reader:
            try:
                item["scanId"] = item["scanId"]
                item["viewpointId"] = item["viewpointId"]
                item["image_h"] = int(item["image_h"])
                item["image_w"] = int(item["image_w"])
                item["vfov"] = int(item["vfov"])

                item["features"] = np.frombuffer(
                    base64.b64decode(item["features"]), dtype=np.float32
                ).reshape((-1, FEATURE_SIZE))

                item["region_tokens"] = item["region_tokens"]

                item["boxes"] = np.frombuffer(
                    base64.b64decode(item["boxes"]), dtype=np.float32
                ).reshape((-1, 4))

                item["viewHeading"] = item["viewHeading"]

                item["viewElevation"] = item["viewElevation"]

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
            except Exception as inst:
                print(inst)
                import pdb

                pdb.set_trace()

            in_data.append(item)
    print("Read features of length %d" % len(in_data))
    return in_data


if __name__ == "__main__":

    if args.gpu_id == -1:
        gpu_ids = range(NUM_GPUS)
        p = Pool(NUM_GPUS)
        p.map(build_tsv, gpu_ids)
        merge_tsvs()
    else:
        build_tsv(gpu_id=args.gpu_id)
        print("Run merge_tsvs() after running build_tsv for all gpu_ids")
        # Uncomment next line to merge all tsvs
        # merge_tsvs()

    data = read_tsv(MERGED)
    with open(MERGED_PICKLE, "wb") as handle:
        pickle.dump(data, handle)

    with open(MERGED_PICKLE, "rb") as handle:
        data = pickle.load(handle)
    print("Pickle file loaded!")
