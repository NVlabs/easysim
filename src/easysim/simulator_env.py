# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import gym
import abc

from easysim.scene import Scene
from easysim.simulators.registration import make


class SimulatorEnv(gym.Env, abc.ABC):
    """ """

    def __init__(self, cfg, **kwargs):
        """ """
        self._cfg = cfg

        self._scene = Scene()

        self.init(**kwargs)

        self._simulator = make(self.cfg.SIM.SIMULATOR, cfg=self.cfg.SIM)

    @property
    def cfg(self):
        """ """
        return self._cfg

    @property
    def scene(self):
        """ """
        return self._scene

    @abc.abstractmethod
    def init(self, **kwargs):
        """ """

    def reset(self, env_ids=None, **kwargs):
        """ """
        self.pre_reset(env_ids, **kwargs)

        self._simulator.reset(self.scene.bodies, env_ids)

        observation = self.post_reset(env_ids, **kwargs)

        return observation

    @abc.abstractmethod
    def pre_reset(self, env_ids, **kwargs):
        """ """

    @abc.abstractmethod
    def post_reset(self, env_ids, **kwargs):
        """ """

    def step(self, action):
        """ """
        self.pre_step(action)

        self._simulator.step(self.scene.bodies)

        observation, reward, done, info = self.post_step(action)

        return observation, reward, done, info

    @abc.abstractmethod
    def pre_step(self, action):
        """ """

    @abc.abstractmethod
    def post_step(self, action):
        """ """

    @property
    def contact(self):
        """ """
        return self._simulator.contact

    def close(self):
        """ """
        self._simulator.close()


class SimulatorWrapper(gym.Wrapper):
    """ """

    def reset(self, env_ids=None, **kwargs):
        """ """
        return self.env.reset(env_ids=env_ids, **kwargs)

    def step(self, action):
        """ """
        return self.env.step(action)

    def close(self):
        """ """
        return self.env.close()
