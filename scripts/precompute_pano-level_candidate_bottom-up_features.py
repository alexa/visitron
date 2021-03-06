# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import cv2
import json
import math
import base64
import csv
import os
import sys

import random
from multiprocessing import Pool

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

SEED = 1


random.seed(SEED)
csv.field_size_limit(sys.maxsize)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import MatterSim

caffe_root = "bottom-up-matterport"
sys.path.insert(0, caffe_root + "/caffe/python")
import caffe

sys.path.insert(0, caffe_root + "/lib")
sys.path.insert(0, caffe_root + "/lib/rpn")
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

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

NUM_GPUS = 8

FEATURE_SIZE = 2048

NUM_SWEEPS = 3
VIEW_PER_SWEEP = 12
VIEWPOINT_SIZE = NUM_SWEEPS * VIEW_PER_SWEEP
HEADING_INC = 360 / VIEW_PER_SWEEP
ANGLE_MARGIN = 5
ELEVATION_START = -30
ELEVATION_INC = 30

PROTO = caffe_root + "/models/faster_rcnn_end2end_final.prototxt"
MODEL = caffe_root + "/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel"
CFG_FILE = caffe_root + "/experiments/cfgs/faster_rcnn_end2end_resnet.yml"
UPDOWN_DATA = caffe_root + "/data/genome/1600-400-20"

GRAPHS = "connectivity/"
OUTFILE = "img_features/ResNet-101-faster-rcnn-genome-candidate/ResNet-101-faster-rcnn-genome-candidate.tsv.%d.%d"
MERGED = "img_features/ResNet-101-faster-rcnn-genome-candidate.tsv"

WIDTH = 600
HEIGHT = 600
VFOV = 80
ASPECT = WIDTH / HEIGHT
HFOV = math.degrees(2 * math.atan(math.tan(math.radians(VFOV / 2)) * ASPECT))
FOC = (HEIGHT / 2) / math.tan(math.radians(VFOV / 2))  # focal length

MIN_LOCAL_BOXES = 5
MAX_LOCAL_BOXES = 20
MAX_TOTAL_BOXES = 100
NMS_THRESH = 0.3
CONF_THRESH = 0.4


def load_viewpointids(node_id=0, gpu_id=0):
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
    # random.shuffle(viewpointIds)

    start_length = 1000
    end_length = 1000
    if node_id == 9:
        end_length = 2000

    if NUM_GPUS != 1:
        viewpointIds = viewpointIds[node_id * start_length : (node_id + 1) * end_length]
        print(
            "%d to %d viewpoints" % (node_id * start_length, (node_id + 1) * end_length)
        )
        viewpointIds = viewpointIds[gpu_id::NUM_GPUS]
    print("%d: Loaded %d viewpoints" % (gpu_id, len(viewpointIds)))
    return viewpointIds


def transform_img(im):
    """ Prep opencv BGR 3 channel image for the network """
    np_im = np.array(im, copy=True)
    return np_im


def visual_overlay(im, dets, ix, classes, attributes):
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    valid = np.where(dets["featureViewIndex"] == ix)[0]
    objects = np.argmax(dets["cls_prob"][valid, 1:], axis=1)
    obj_conf = np.max(dets["cls_prob"][valid, 1:], axis=1)

    attr_threshold = 0.1
    attr = np.argmax(dets["attr_prob"][valid, 1:], axis=1)
    attr_conf = np.max(dets["attr_prob"][valid, 1:], axis=1)

    boxes = dets["boxes"][valid]

    for i in range(len(valid)):
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


def extract_region_tokens(cls_prob, attr_prob, classes, attributes):
    objects = np.argmax(cls_prob[:, 1:], axis=1)

    attr_threshold = 0.1
    attr = np.argmax(attr_prob[:, 1:], axis=1)
    attr_conf = np.max(attr_prob[:, 1:], axis=1)

    region_tokens = []

    for i in range(len(cls_prob)):
        cls = classes[objects[i] + 1]
        if attr_conf[i] > attr_threshold:
            cls = attributes[attr[i] + 1] + " " + cls
        region_tokens.append(cls)

    return region_tokens


def get_detections_from_im(
    record, net, im, classes, attributes, conf_thresh=CONF_THRESH
):

    if "feature" not in record:
        ix = 0
    elif record["featureViewIndex"].shape[0] == 0:
        ix = 0
    else:
        ix = int(record["featureViewIndex"][-1]) + 1

    boxes = [[0.0, 0.0, 599.0, 599.0]]
    boxes = np.array(boxes)
    force_boxes = True

    scores, boxes, attr_scores, _ = im_detect(
        net, im, boxes=boxes, force_boxes=force_boxes
    )

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

    hor_thresh = FOC * math.tan(math.radians(HEADING_INC / 2 + ANGLE_MARGIN))
    vert_thresh = FOC * math.tan(math.radians(ELEVATION_INC / 2 + ANGLE_MARGIN))
    center_x = 0.5 * (cls_boxes[:, 0] + cls_boxes[:, 2])
    center_y = 0.5 * (cls_boxes[:, 1] + cls_boxes[:, 3])
    reject = (center_x < WIDTH / 2 - hor_thresh) | (center_x > WIDTH / 2 + hor_thresh)
    heading = record["viewHeading"][ix]
    elevation = record["viewElevation"][ix]

    if ix >= VIEW_PER_SWEEP:
        reject |= center_y > HEIGHT / 2 + vert_thresh
    if ix < VIEWPOINT_SIZE - VIEW_PER_SWEEP:
        reject |= center_y < HEIGHT / 2 - vert_thresh

    keep_boxes = np.setdiff1d(keep_boxes, np.argwhere(reject))

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

    if "features" not in record:
        record["boxes"] = cls_boxes[keep_boxes]
        record["cls_prob"] = cls_prob[keep_boxes]
        record["attr_prob"] = attr_prob[keep_boxes]
        record["features"] = pool5[keep_boxes]
        record["region_tokens"] = extract_region_tokens(
            cls_prob[keep_boxes], attr_prob[keep_boxes], classes, attributes
        )
        record["featureViewIndex"] = (
            np.ones((len(keep_boxes), 1), dtype=np.float32) * ix
        )
        record["featureHeading"] = featureHeading
        record["featureElevation"] = featureElevation
    else:
        record["boxes"] = np.vstack([record["boxes"], cls_boxes[keep_boxes]])
        record["cls_prob"] = np.vstack([record["cls_prob"], cls_prob[keep_boxes]])
        record["attr_prob"] = np.vstack([record["attr_prob"], attr_prob[keep_boxes]])
        record["features"] = np.vstack([record["features"], pool5[keep_boxes]])
        record["region_tokens"].extend(
            extract_region_tokens(
                cls_prob[keep_boxes], attr_prob[keep_boxes], classes, attributes
            )
        )
        record["featureViewIndex"] = np.vstack(
            [
                record["featureViewIndex"],
                np.ones((len(keep_boxes), 1), dtype=np.float32) * ix,
            ]
        )
        record["featureHeading"] = np.vstack([record["featureHeading"], featureHeading])
        record["featureElevation"] = np.vstack(
            [record["featureElevation"], featureElevation]
        )
    return


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
            "featureViewIndex",
            "featureHeading",
            "featrueElevation",
        ]:
            record[k] = v[sorted(keep)]
        elif k in ["region_tokens"]:
            record[k] = [v[i] for i in sorted(keep)]


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
    NODE_ID = 9

    print("NODE_ID: %d GPU_ID: %d build_tsv" % (NODE_ID, gpu_id))

    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    cfg_from_file(CFG_FILE)
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(PROTO, caffe.TEST, weights=MODEL)
    classes, attributes = load_classes()

    count = 0
    t_render = Timer()
    t_net = Timer()
    with open(OUTFILE % (NODE_ID, gpu_id), "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FILENAMES)

        viewpointIds = load_viewpointids(NODE_ID, gpu_id)

        for scanId, viewpointId in viewpointIds:
            t_render.tic()

            ims = []
            sim.newEpisode(
                [scanId], [viewpointId], [0], [math.radians(ELEVATION_START)]
            )
            for ix in range(VIEWPOINT_SIZE):
                state = sim.getState()[0]

                ims.append(transform_img(state.rgb))

                if ix == 0:
                    record = {
                        "scanId": state.scanId,
                        "viewpointId": state.location.viewpointId,
                        "viewHeading": np.zeros(VIEWPOINT_SIZE, dtype=np.float32),
                        "viewElevation": np.zeros(VIEWPOINT_SIZE, dtype=np.float32),
                        "image_h": HEIGHT,
                        "image_w": WIDTH,
                        "vfov": VFOV,
                    }

                record["viewHeading"][ix] = state.heading
                record["viewElevation"][ix] = state.elevation

                elev = 0.0
                heading_chg = math.pi * 2 / VIEW_PER_SWEEP
                view = ix % VIEW_PER_SWEEP
                sweep = ix // VIEW_PER_SWEEP
                if view + 1 == VIEW_PER_SWEEP:
                    elev = math.radians(ELEVATION_INC)
                sim.makeAction([0], [heading_chg], [elev])

            t_render.toc()
            t_net.tic()

            for ix in range(VIEWPOINT_SIZE):
                get_detections_from_im(record, net, ims[ix], classes, attributes)
            if DRY_RUN:
                print(
                    "%d: Detected %d objects in pano"
                    % (gpu_id, record["features"].shape[0])
                )

            filter(record, MAX_TOTAL_BOXES)

            if DRY_RUN:
                print(
                    "%d: Reduced to %d objects in pano"
                    % (gpu_id, record["features"].shape[0])
                )

                for ix in range(VIEWPOINT_SIZE):
                    fig = visual_overlay(ims[ix], record, ix, classes, attributes)

                    fig.savefig(
                        "img_features/examples/%s-%s-%d.png"
                        % (record["scanId"], record["viewpointId"], ix,)
                    )
                    print(
                        "Saved %s-%s-%s"
                        % (record["scanId"], record["viewpointId"], ix,)
                    )
                    plt.close()

            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    record[k] = str(base64.b64encode(v)).encode("utf-8")
            writer.writerow(record)

            count += 1
            t_net.toc()

            if count % 10 == 0:
                print(
                    "%d: Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours"
                    % (
                        gpu_id,
                        count,
                        len(viewpointIds),
                        t_render.average_time,
                        t_net.average_time,
                        (
                            (t_render.average_time + t_net.average_time)
                            * len(viewpointIds)
                            / 3600
                        ),
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

            in_data.append(item)
    print("Read features of length %d" % len(in_data))
    return in_data


if __name__ == "__main__":

    gpu_ids = range(NUM_GPUS)
    p = Pool(NUM_GPUS)
    p.map(build_tsv, gpu_ids)
    # merge_tsvs()

    # build_tsv(gpu_id=0)

    # data = read_tsv(MERGED)
    # import pdb

    # pdb.set_trace()
