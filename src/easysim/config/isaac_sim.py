# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from yacs.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Isaac Sim config
# ---------------------------------------------------------------------------- #
ISAAC_SIM_CONFIG = CN()

ISAAC_SIM_CONFIG.MODULES_DISABLED_AT_LAUNCH = ()

ISAAC_SIM_CONFIG.SPACING = 4.0

ISAAC_SIM_CONFIG.ADD_DISTANT_LIGHT = True

# ---------------------------------------------------------------------------- #
# Isaac Sim Viewer config
# ---------------------------------------------------------------------------- #
ISAAC_SIM_CONFIG.VIEWER = CN()

ISAAC_SIM_CONFIG.VIEWER.RENDER_FRAME_RATE = 60.0
