# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import abc


class Simulator(abc.ABC):
    """Simulator."""

    def __init__(self, cfg):
        """ """
        self._cfg = cfg

    @abc.abstractmethod
    def reset(self, bodies, env_ids):
        """ """

    @abc.abstractmethod
    def step(self, bodies):
        """ """

    @property
    @abc.abstractmethod
    def contact(self):
        """ """

    @abc.abstractmethod
    def close(self):
        """ """