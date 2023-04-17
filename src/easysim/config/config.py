# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

"""

Derived from:
https://github.com/facebookresearch/detectron2/blob/6e7def97f723bedd25ad6d2aa788802bf982c72c/detectron2/config/defaults.py
"""

from yacs.config import CfgNode as CN

from easysim.config.bullet import BULLET_CONFIG
from easysim.config.isaac_gym import ISAAC_GYM_CONFIG
from easysim.config.isaac_sim import ISAAC_SIM_CONFIG


_C = CN()

# ---------------------------------------------------------------------------- #
# Simulation config
# ---------------------------------------------------------------------------- #
_C.SIM = CN()

# Simulator choice
#     Valid options: ("bullet", "isaac_gym")
_C.SIM.SIMULATOR = "bullet"

_C.SIM.RENDER = False

_C.SIM.GRAVITY = (0.0, 0.0, -9.8)

_C.SIM.USE_DEFAULT_STEP_PARAMS = True

_C.SIM.TIME_STEP = 1.0 / 240.0

_C.SIM.SUBSTEPS = 1

_C.SIM.NUM_ENVS = 1

_C.SIM.SIM_DEVICE = "cpu"

_C.SIM.USE_GPU_PIPELINE = False

_C.SIM.LOAD_GROUND_PLANE = True

# ---------------------------------------------------------------------------- #
# Viewer config
# ---------------------------------------------------------------------------- #
_C.SIM.VIEWER = CN()

_C.SIM.VIEWER.INIT_CAMERA_POSITION = (None, None, None)

_C.SIM.VIEWER.INIT_CAMERA_TARGET = (None, None, None)

# ---------------------------------------------------------------------------- #
# Ground plane config
# ---------------------------------------------------------------------------- #
_C.SIM.GROUND_PLANE = CN()

_C.SIM.GROUND_PLANE.DISTANCE = 0.0

# ---------------------------------------------------------------------------- #
# Bullet config
# ---------------------------------------------------------------------------- #
_C.SIM.BULLET = BULLET_CONFIG

# ---------------------------------------------------------------------------- #
# Isaac Gym config
# ---------------------------------------------------------------------------- #
_C.SIM.ISAAC_GYM = ISAAC_GYM_CONFIG

# ---------------------------------------------------------------------------- #
# Isaac Sim config
# ---------------------------------------------------------------------------- #
_C.SIM.ISAAC_SIM = ISAAC_SIM_CONFIG


cfg = _C


def get_cfg():
    """ """
    return _C.clone()
