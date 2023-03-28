# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from yacs.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Isaac Gym config
# ---------------------------------------------------------------------------- #
ISAAC_GYM_CONFIG = CN()

ISAAC_GYM_CONFIG.GRAPHICS_DEVICE_ID = 0

ISAAC_GYM_CONFIG.SPACING = 2.0

ISAAC_GYM_CONFIG.ENABLE_CAMERA_SENSORS = False

# ---------------------------------------------------------------------------- #
# Isaac Gym PhysX config
# ---------------------------------------------------------------------------- #
ISAAC_GYM_CONFIG.PHYSX = CN()

ISAAC_GYM_CONFIG.PHYSX.MAX_DEPENETRATION_VELOCITY = 100.0

# Contact collection mode
#     0: Don't collect any contacts.
#     1: Collect contacts for last substep only.
#     2: Collect contacts for all substeps.
ISAAC_GYM_CONFIG.PHYSX.CONTACT_COLLECTION = 2

# ---------------------------------------------------------------------------- #
# Isaac Gym Viewer config
# ---------------------------------------------------------------------------- #
ISAAC_GYM_CONFIG.VIEWER = CN()

ISAAC_GYM_CONFIG.VIEWER.RENDER_FRAME_RATE = 60.0

ISAAC_GYM_CONFIG.VIEWER.DRAW_AXES = True
