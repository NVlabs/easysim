# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

"""

Derived from:
https://github.com/facebookresearch/detectron2/blob/6e7def97f723bedd25ad6d2aa788802bf982c72c/detectron2/config/defaults.py
"""

from yacs.config import CfgNode as CN


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

_C.SIM.TIME_STEP = 1 / 240

_C.SIM.SUBSTEPS = 1

_C.SIM.INIT_VIEWER_CAMERA_POSITION = (None, None, None)

_C.SIM.INIT_VIEWER_CAMERA_TARGET = (None, None, None)

_C.SIM.NUM_ENVS = 1

_C.SIM.SIM_DEVICE = "cpu"

# ---------------------------------------------------------------------------- #
# Isaac Gym config
# ---------------------------------------------------------------------------- #
_C.SIM.ISAAC_GYM = CN()

_C.SIM.ISAAC_GYM.GRAPHICS_DEVICE_ID = 0

_C.SIM.ISAAC_GYM.USE_GPU_PIPELINE = False

_C.SIM.ISAAC_GYM.SPACING = 2.0

_C.SIM.ISAAC_GYM.RENDER_FRAME_RATE = 60

# ---------------------------------------------------------------------------- #
# Isaac Gym PhysX config
# ---------------------------------------------------------------------------- #
_C.SIM.ISAAC_GYM.PHYSX = CN()

_C.SIM.ISAAC_GYM.PHYSX.MAX_DEPENETRATION_VELOCITY = 100.0

# Contact collection mode
#     0: Don't collect any contacts.
#     1: Collect contacts for last substep only.
#     2: Collect contacts for all substeps.
_C.SIM.ISAAC_GYM.PHYSX.CONTACT_COLLECTION = 2


cfg = _C


def get_cfg():
    """ """
    return _C.clone()
