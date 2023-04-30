# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import abc


class Simulator(abc.ABC):
    """Simulator."""

    def __init__(self, cfg, scene):
        """ """
        self._cfg = cfg
        self._scene = scene

        self._init()

        self._scene.set_device(self.device, self.graphics_device)

    @abc.abstractmethod
    def _init(self):
        """ """

    @property
    @abc.abstractmethod
    def device(self):
        """ """

    @property
    @abc.abstractmethod
    def graphics_device(self):
        """ """

    @abc.abstractmethod
    def reset(self, env_ids):
        """ """

    @abc.abstractmethod
    def step(self):
        """ """

    @property
    @abc.abstractmethod
    def contact(self):
        """ """

    @abc.abstractmethod
    def close(self):
        """ """
