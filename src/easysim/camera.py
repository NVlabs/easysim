# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import numpy as np
import torch

from contextlib import contextmanager


class Camera:
    """ """

    _ATTR_ARRAY_NDIM = {
        "width": 0,
        "height": 0,
        "vertical_fov": 0,
        "near": 0,
        "far": 0,
        "position": 1,
        "target": 1,
        "up_vector": 1,
        "orientation": 1,
    }

    _created = False

    def __init__(
        self,
        name=None,
        device=None,
        width=None,
        height=None,
        vertical_fov=None,
        near=None,
        far=None,
        position=None,
        target=None,
        up_vector=None,
        orientation=None,
    ):
        """ """
        self._init_attr_array_pipeline()
        self._init_callback()

        self.name = name
        self.device = device

        self.width = width
        self.height = height
        self.vertical_fov = vertical_fov
        self.near = near
        self.far = far

        self.position = position
        self.target = target
        self.up_vector = up_vector
        self.orientation = orientation

        self.color = None
        self.depth = None
        self.segmentation = None

        self._created = True

    def __setattr__(self, key, value):
        """ """
        # Exclude `color`, `depth`, and `segmentation` to prevent infinite recursion in property
        # calls.
        if (
            self._created
            and key not in ("color", "depth", "segmentation")
            and not hasattr(self, key)
        ):
            raise TypeError(f"Unrecognized Camera attribute '{key}'")
        object.__setattr__(self, key, value)

    def _init_attr_array_pipeline(self):
        """ """
        self._attr_array_locked = {}
        self._attr_array_dirty_flag = {}
        self._attr_array_dirty_mask = {}
        for attr in self._ATTR_ARRAY_NDIM:
            self._attr_array_locked[attr] = False
            self._attr_array_dirty_flag[attr] = False

    @property
    def attr_array_locked(self):
        """ """
        return self._attr_array_locked

    @property
    def attr_array_dirty_flag(self):
        """ """
        return self._attr_array_dirty_flag

    @property
    def attr_array_dirty_mask(self):
        """ """
        return self._attr_array_dirty_mask

    def _init_callback(self):
        """ """
        self._callback_render_color = None
        self._callback_render_depth = None
        self._callback_render_segmentation = None

    @property
    def name(self):
        """ """
        return self._name

    @name.setter
    def name(self, value):
        """ """
        self._name = value

    @property
    def device(self):
        """ """
        return self._device

    @device.setter
    def device(self, value):
        """ """
        self._device = value

        if hasattr(self, "_color") and self.color is not None:
            self.color = self.color.to(value)
        if hasattr(self, "_depth") and self.depth is not None:
            self.depth = self.depth.to(value)
        if hasattr(self, "_segmentation") and self.segmentation is not None:
            self.segmentation = self.segmentation.to(value)

    @property
    def width(self):
        """ """
        return self._width

    @width.setter
    def width(self, value):
        """ """
        assert not self._attr_array_locked[
            "width"
        ], "'width' cannot be directly changed after simulation starts. Use 'update_attr_array()'."
        if value is not None:
            value = np.asanyarray(value, dtype=np.int64)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["width"],
                self._ATTR_ARRAY_NDIM["width"] + 1,
            ):
                raise ValueError(
                    "'width' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['width']} or {self._ATTR_ARRAY_NDIM['width'] + 1}"
                )
        self._width = value

    @property
    def height(self):
        """ """
        return self._height

    @height.setter
    def height(self, value):
        """ """
        assert not self._attr_array_locked[
            "height"
        ], "'height' cannot be directly changed after simulation starts. Use 'update_attr_array()'."
        if value is not None:
            value = np.asanyarray(value, dtype=np.int64)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["height"],
                self._ATTR_ARRAY_NDIM["height"] + 1,
            ):
                raise ValueError(
                    "'height' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['height']} or {self._ATTR_ARRAY_NDIM['height'] + 1}"
                )
        self._height = value

    @property
    def vertical_fov(self):
        """ """
        return self._vertical_fov

    @vertical_fov.setter
    def vertical_fov(self, value):
        """ """
        assert not self._attr_array_locked["vertical_fov"], (
            "'vertical_fov' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["vertical_fov"],
                self._ATTR_ARRAY_NDIM["vertical_fov"] + 1,
            ):
                raise ValueError(
                    "'vertical_fov' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['vertical_fov']} or "
                    f"{self._ATTR_ARRAY_NDIM['vertical_fov'] + 1}"
                )
        self._vertical_fov = value

    @property
    def near(self):
        """ """
        return self._near

    @near.setter
    def near(self, value):
        """ """
        assert not self._attr_array_locked[
            "near"
        ], "'near' cannot be directly changed after simulation starts. Use 'update_attr_array()'."
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (self._ATTR_ARRAY_NDIM["near"], self._ATTR_ARRAY_NDIM["near"] + 1):
                raise ValueError(
                    f"'near' must have a number of dimensions of {self._ATTR_ARRAY_NDIM['near']} "
                    f"or {self._ATTR_ARRAY_NDIM['near'] + 1}"
                )
        self._near = value

    @property
    def far(self):
        """ """
        return self._far

    @far.setter
    def far(self, value):
        """ """
        assert not self._attr_array_locked[
            "far"
        ], "'far' cannot be directly changed after simulation starts. Use 'update_attr_array()'."
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (self._ATTR_ARRAY_NDIM["far"], self._ATTR_ARRAY_NDIM["far"] + 1):
                raise ValueError(
                    f"'far' must have a number of dimensions of {self._ATTR_ARRAY_NDIM['far']} "
                    f"or {self._ATTR_ARRAY_NDIM['far'] + 1}"
                )
        self._far = value

    @property
    def position(self):
        """ """
        return self._position

    @position.setter
    def position(self, value):
        """ """
        assert not self._attr_array_locked["position"], (
            "'position' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["position"],
                self._ATTR_ARRAY_NDIM["position"] + 1,
            ):
                raise ValueError(
                    "'position' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['position']} or "
                    f"{self._ATTR_ARRAY_NDIM['position'] + 1}"
                )
            if value.shape[-1] != 3:
                raise ValueError("'position' must have the last dimension of size 3")
        self._position = value

    @property
    def target(self):
        """ """
        return self._target

    @target.setter
    def target(self, value):
        """ """
        assert not self._attr_array_locked[
            "target"
        ], "'target' cannot be directly changed after simulation starts. Use 'update_attr_array()'."
        if value is not None:
            if self.orientation is not None:
                raise ValueError(
                    "('target', 'up_vector') and 'orientation' cannot both be set at the same time"
                )
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["target"],
                self._ATTR_ARRAY_NDIM["target"] + 1,
            ):
                raise ValueError(
                    "'target' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['target']} or {self._ATTR_ARRAY_NDIM['target'] + 1}"
                )
            if value.shape[-1] != 3:
                raise ValueError("'target' must have the last dimension of size 3")
        self._target = value

    @property
    def up_vector(self):
        """ """
        return self._up_vector

    @up_vector.setter
    def up_vector(self, value):
        """ """
        assert not self._attr_array_locked["up_vector"], (
            "'up_vector' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            if self.orientation is not None:
                raise ValueError(
                    "('target', 'up_vector') and 'orientation' cannot both be set at the same time"
                )
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["up_vector"],
                self._ATTR_ARRAY_NDIM["up_vector"] + 1,
            ):
                raise ValueError(
                    "'up_vector' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['up_vector']} or "
                    f"{self._ATTR_ARRAY_NDIM['up_vector'] + 1}"
                )
            if value.shape[-1] != 3:
                raise ValueError("'up_vector' must have the last dimension of size 3")
        self._up_vector = value

    @property
    def orientation(self):
        """ """
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """ """
        assert not self._attr_array_locked["orientation"], (
            "'orientation' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            if self.target is not None or self.up_vector is not None:
                raise ValueError(
                    "('target', 'up_vector') and 'orientation' cannot both be set at the same time"
                )
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["orientation"],
                self._ATTR_ARRAY_NDIM["orientation"] + 1,
            ):
                raise ValueError(
                    "'orientation' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['orientation']} or "
                    f"{self._ATTR_ARRAY_NDIM['orientation'] + 1}"
                )
            if value.shape[-1] != 4:
                raise ValueError("'orientation' must have the last dimension of size 4")
        self._orientation = value

    @property
    def color(self):
        """ """
        if self._color is None and self._callback_render_color is not None:
            self._callback_render_color(self)
        return self._color

    @color.setter
    def color(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.uint8, device=self.device)
        self._color = value

    @property
    def depth(self):
        """ """
        if self._depth is None and self._callback_render_depth is not None:
            self._callback_render_depth(self)
        return self._depth

    @depth.setter
    def depth(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self._depth = value

    @property
    def segmentation(self):
        """ """
        if self._segmentation is None and self._callback_render_segmentation is not None:
            self._callback_render_segmentation(self)
        return self._segmentation

    @segmentation.setter
    def segmentation(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.int32, device=self.device)
        self._segmentation = value

    def get_attr_array(self, attr, idx):
        """ """
        return self._get_attr(attr, self._ATTR_ARRAY_NDIM[attr], idx)

    def _get_attr(self, attr, ndim, idx):
        """ """
        array = getattr(self, attr)
        if array.ndim == ndim:
            return array
        if array.ndim == ndim + 1:
            return array[idx]

    def lock_attr_array(self):
        """ """
        for k in self._attr_array_locked:
            if not self._attr_array_locked[k]:
                self._attr_array_locked[k] = True
            if getattr(self, k) is not None:
                getattr(self, k).flags.writeable = False

    def update_attr_array(self, attr, env_ids, value):
        """ """
        if getattr(self, attr).ndim != self._ATTR_ARRAY_NDIM[attr] + 1:
            raise ValueError(
                f"'{attr}' can only be updated when a per-env specification (ndim: "
                f"{self._ATTR_ARRAY_NDIM[attr] + 1}) is used"
            )
        if len(env_ids) == 0:
            return

        env_ids_np = env_ids.cpu().numpy()

        with self._make_attr_array_writeable(attr):
            getattr(self, attr)[env_ids_np] = value

        if not self._attr_array_dirty_flag[attr]:
            self._attr_array_dirty_flag[attr] = True
        try:
            self._attr_array_dirty_mask[attr][env_ids_np] = True
        except KeyError:
            self._attr_array_dirty_mask[attr] = np.zeros(len(getattr(self, attr)), dtype=bool)
            self._attr_array_dirty_mask[attr][env_ids_np] = True

    @contextmanager
    def _make_attr_array_writeable(self, attr):
        """ """
        try:
            getattr(self, attr).flags.writeable = True
            yield
        finally:
            getattr(self, attr).flags.writeable = False

    def set_callback_render_color(self, callback):
        """ """
        self._callback_render_color = callback

    def set_callback_render_depth(self, callback):
        """ """
        self._callback_render_depth = callback

    def set_callback_render_segmentation(self, callback):
        """ """
        self._callback_render_segmentation = callback
