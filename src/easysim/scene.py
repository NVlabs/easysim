# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from easysim.body import Body


class Scene:
    """ """

    def __init__(self):
        # A `list` is adopted here rather than a `set`, because for some simulators, the simulation
        # outcome is variant to the loading order of bodies. Using a `list` can enforce a
        # deterministic and controllable order.
        self._bodies = list()

        self._name_to_body = {}

    @property
    def bodies(self):
        """ """
        return self._bodies

    def add_body(self, body):
        """ """
        if not isinstance(body, Body):
            raise TypeError("body must be a Body")
        if body in self.bodies:
            raise ValueError("body already in scene")
        if body.name is None:
            raise ValueError("body.name must not be None")
        if body.name in self._name_to_body:
            raise ValueError(f"Cannot add body with duplicated name: '{body.name}'")
        self.bodies.append(body)

        self._name_to_body[body.name] = body

    def remove_body(self, body):
        """ """
        if body not in self.bodies:
            raise ValueError("body not in the scene")
        self.bodies.remove(body)

        del self._name_to_body[body.name]

    def get_body(self, name):
        """ """
        if name not in self._name_to_body:
            raise ValueError(f"Non-existent body name: '{name}'")
        return self._name_to_body[name]
