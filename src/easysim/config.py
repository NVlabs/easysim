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
#     Valid options: ('bullet', 'isaac_gym')
_C.SIM.SIMULATOR = "bullet"

_C.SIM.RENDER = False

_C.SIM.GRAVITY = (0.0, 0.0, -9.8)

_C.SIM.USE_DEFAULT_STEP_PARAMS = True

_C.SIM.TIME_STEP = 1 / 240

_C.SIM.SUBSTEPS = 1

_C.SIM.INIT_VIEWER_CAMERA_POSITION = (None, None, None)

_C.SIM.INIT_VIEWER_CAMERA_TARGET = (None, None, None)

_C.SIM.SIM_DEVICE = "cpu"

#
# Isaac Gym specific config
#

_C.SIM.GRAPHICS_DEVICE_ID = 0

_C.SIM.USE_GPU_PIPELINE = False

_C.SIM.NUM_ENVS = 1

_C.SIM.SPACING = 2.0

_C.SIM.RENDER_FRAME_RATE = 60


cfg = _C


def get_cfg():
    """ """
    return _C.clone()
