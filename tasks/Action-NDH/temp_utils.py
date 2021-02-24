# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import random
import math
import torch
import numpy as np


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def angle_feature(heading, elevation):
    import math

    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array(
        [
            math.sin(heading),
            math.cos(heading),
            math.sin(elevation),
            math.cos(elevation),
        ],
        dtype=np.float32,
    )


def get_point_angle_feature(baseViewId=0):
    sim = new_simulator()

    angle_feat_size = 4

    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(
                ["ZMojNkEp431"],
                ["2f4d90acd4024c269fb0efe49a8ac540"],
                [0],
                [math.radians(-30)],
            )
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature


def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]


def new_simulator():
    import MatterSim

    # Simulator image parameters
    WIDTH = 600
    HEIGHT = 600
    VFOV = 80

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.initialize()

    return sim


def length2mask(length, device, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (
        torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
        > (torch.LongTensor(length) - 1).unsqueeze(1)
    ).to(device)
    return mask
