# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


class GeometryType:
    """ """

    URDF = 0
    SPHERE = 1
    BOX = 2


class DoFControlMode:
    """ """

    NONE = 0
    POSITION_CONTROL = 1
    VELOCITY_CONTROL = 2
    TORQUE_CONTROL = 3


class MeshNormalMode:
    """ """

    FROM_ASSET = 0
    COMPUTE_PER_VERTEX = 1
    COMPUTE_PER_FACE = 2
