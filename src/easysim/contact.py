# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import numpy as np

_DTYPE = np.dtype(
    {
        "names": [
            "body_id_a",
            "body_id_b",
            "link_id_a",
            "link_id_b",
            "position_a_world",
            "position_b_world",
            "position_a_link",
            "position_b_link",
            "normal",
            "force",
        ],
        "formats": [
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
            np.float32,
        ],
        "offsets": [0, 4, 8, 12, 16, 28, 40, 52, 64, 76],
        "itemsize": 80,
    }
)


def create_contact_array(
    length,
    body_id_a=None,
    body_id_b=None,
    link_id_a=None,
    link_id_b=None,
    position_a_world=None,
    position_b_world=None,
    position_a_link=None,
    position_b_link=None,
    normal=None,
    force=None,
):
    """ """
    array = np.empty(length, dtype=_DTYPE)
    if length > 0:
        array["body_id_a"] = body_id_a
        array["body_id_b"] = body_id_b
        array["link_id_a"] = link_id_a
        array["link_id_b"] = link_id_b
        array["position_a_world"] = position_a_world
        array["position_b_world"] = position_b_world
        array["position_a_link"] = position_a_link
        array["position_b_link"] = position_b_link
        array["normal"] = normal
        array["force"] = force
    return array
