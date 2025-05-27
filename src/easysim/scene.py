# Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from easysim.body import Body
from easysim.camera import Camera


class Scene:
    """ """

    def __init__(self):
        # A `list` is adopted here rather than a `set`, because for some simulators, the simulation
        # outcome is variant to the loading order of bodies. Using a `list` can enforce a
        # deterministic and controllable order.
        self._bodies = list()

        self._name_to_body = {}

        self._cameras = set()

        self._name_to_camera = {}

        self._init_device()
        self._init_callback()

    @property
    def bodies(self):
        """ """
        return self._bodies

    @property
    def cameras(self):
        """ """
        return self._cameras

    def _init_device(self):
        """ """
        self._device = None
        self._graphics_device = None

    def _init_callback(self):
        """ """
        self._callback_add_camera = None
        self._callback_remove_camera = None

    def set_device(self, device, graphics_device):
        """ """
        self._device = device
        self._graphics_device = graphics_device

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

        body.set_device(self._device)

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

    def add_camera(self, camera):
        """ """
        if not isinstance(camera, Camera):
            raise TypeError("camera must be a Camera")
        if camera in self.cameras:
            raise ValueError("camera already in scene")
        if camera.name is None:
            raise ValueError("camera.name must not be None")
        if camera.name in self._name_to_camera:
            raise ValueError(f"Cannot add camera with duplicated name: '{camera.name}")

        camera.set_device(self._graphics_device)

        self.cameras.add(camera)
        self._name_to_camera[camera.name] = camera

        if self._callback_add_camera is not None:
            self._callback_add_camera(camera)

    def remove_camera(self, camera):
        """ """
        if camera not in self.cameras:
            raise ValueError("camera not in the scene")

        self.cameras.remove(camera)

        del self._name_to_camera[camera.name]

        if self._callback_remove_camera is not None:
            self._callback_remove_camera(camera)

    def get_camera(self, name):
        """ """
        if name not in self._name_to_camera:
            raise ValueError(f"Non-existent camera name: '{name}'")
        return self._name_to_camera[name]

    def set_callback_add_camera(self, callback):
        """ """
        self._callback_add_camera = callback

    def set_callback_remove_camera(self, callback):
        """ """
        self._callback_remove_camera = callback
