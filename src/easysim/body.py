# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import numpy as np

from contextlib import contextmanager


class Body:
    """ """

    _ATTR_ARRAY_NDIM = {
        "link_color": 2,
        "link_collision_filter": 1,
        "link_lateral_friction": 1,
        "link_spinning_friction": 1,
        "link_rolling_friction": 1,
        "link_restitution": 1,
        "link_linear_damping": 0,
        "link_angular_damping": 0,
        "dof_control_mode": 0,
        "dof_max_force": 1,
        "dof_max_velocity": 1,
        "dof_position_gain": 1,
        "dof_velocity_gain": 1,
        "dof_armature": 1,
    }

    _created = False

    def __init__(
        self,
        name=None,
        urdf_file=None,
        device=None,
        use_fixed_base=None,
        use_self_collision=None,
        vhacd_enabled=None,
        vhacd_params=None,
        mesh_normal_mode=None,
        initial_base_position=None,
        initial_base_velocity=None,
        initial_dof_position=None,
        initial_dof_velocity=None,
        link_color=None,
        link_collision_filter=None,
        link_lateral_friction=None,
        link_spinning_friction=None,
        link_rolling_friction=None,
        link_restitution=None,
        link_linear_damping=None,
        link_angular_damping=None,
        dof_control_mode=None,
        dof_max_force=None,
        dof_max_velocity=None,
        dof_position_gain=None,
        dof_velocity_gain=None,
        dof_armature=None,
        dof_target_position=None,
        dof_target_velocity=None,
        dof_force=None,
    ):
        """ """
        self._init_attr_array_pipeline()

        self.name = name
        self.urdf_file = urdf_file
        self.device = device
        self.use_fixed_base = use_fixed_base
        self.use_self_collision = use_self_collision
        self.vhacd_enabled = vhacd_enabled
        self.vhacd_params = vhacd_params
        self.mesh_normal_mode = mesh_normal_mode
        self.initial_base_position = initial_base_position
        self.initial_base_velocity = initial_base_velocity
        self.initial_dof_position = initial_dof_position
        self.initial_dof_velocity = initial_dof_velocity

        self.link_color = link_color
        self.link_collision_filter = link_collision_filter
        self.link_lateral_friction = link_lateral_friction
        self.link_spinning_friction = link_spinning_friction
        self.link_rolling_friction = link_rolling_friction
        self.link_restitution = link_restitution
        self.link_linear_damping = link_linear_damping
        self.link_angular_damping = link_angular_damping

        self.dof_control_mode = dof_control_mode
        self.dof_max_force = dof_max_force
        self.dof_max_velocity = dof_max_velocity
        self.dof_position_gain = dof_position_gain
        self.dof_velocity_gain = dof_velocity_gain
        self.dof_armature = dof_armature

        self.dof_target_position = dof_target_position
        self.dof_target_velocity = dof_target_velocity
        self.dof_force = dof_force

        self.env_ids_reset_dof_state = None

        self.dof_state = None
        self.link_state = None
        self.contact_id = None

        self._created = True

    def __setattr__(self, key, value):
        """ """
        if self._created and not hasattr(self, key):
            raise TypeError(f"Unrecognized Body attribute '{key}': {self.name}")
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
    def name(self):
        """ """
        return self._name

    @name.setter
    def name(self, value):
        """ """
        self._name = value

    @property
    def urdf_file(self):
        """ """
        return self._urdf_file

    @urdf_file.setter
    def urdf_file(self, value):
        """ """
        self._urdf_file = value

    @property
    def device(self):
        """ """
        return self._device

    @device.setter
    def device(self, value):
        """ """
        self._device = value

        if hasattr(self, "_initial_base_position") and self.initial_base_position is not None:
            self.initial_base_position = self.initial_base_position.to(value)
        if hasattr(self, "_initial_base_velocity") and self.initial_base_velocity is not None:
            self.initial_base_velocity = self.initial_base_velocity.to(value)
        if hasattr(self, "_initial_dof_position") and self.initial_dof_position is not None:
            self.initial_dof_position = self.initial_dof_position.to(value)
        if hasattr(self, "_initial_dof_velocity") and self.initial_dof_velocity is not None:
            self.initial_dof_velocity = self.initial_dof_velocity.to(value)

        if hasattr(self, "_dof_target_position") and self.dof_target_position is not None:
            self.dof_target_position = self.dof_target_position.to(value)
        if hasattr(self, "_dof_target_velocity") and self.dof_target_velocity is not None:
            self.dof_target_velocity = self.dof_target_velocity.to(value)
        if hasattr(self, "_dof_force") and self.dof_force is not None:
            self.dof_force = self.dof_force.to(value)

        if hasattr(self, "_env_ids_reset_dof_state") and self.env_ids_reset_dof_state is not None:
            self.env_ids_reset_dof_state = self.env_ids_reset_dof_state.to(value)

        if hasattr(self, "_dof_state") and self.dof_state is not None:
            self.dof_state = self.dof_state.to(value)
        if hasattr(self, "_link_state") and self.link_state is not None:
            self.link_state = self.link_state.to(value)

    @property
    def initial_base_position(self):
        """ """
        return self._initial_base_position

    @initial_base_position.setter
    def initial_base_position(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError(
                    "'initial_base_position' must have a number of dimensions of 1 or 2"
                )
            if value.shape[-1] != 7:
                raise ValueError("'initial_base_position' must have the last dimension of size 7")
        self._initial_base_position = value

    @property
    def initial_base_velocity(self):
        """ """
        return self._initial_base_velocity

    @initial_base_velocity.setter
    def initial_base_velocity(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError(
                    "'initial_base_velocity' must have a number of dimensions of 1 or 2"
                )
            if value.shape[-1] != 6:
                raise ValueError("'initial_base_velocity' must have the last dimension of size 6")
        self._initial_base_velocity = value

    @property
    def use_fixed_base(self):
        """ """
        return self._use_fixed_base

    @use_fixed_base.setter
    def use_fixed_base(self, value):
        """ """
        self._use_fixed_base = value

    @property
    def use_self_collision(self):
        """ """
        return self._use_self_collision

    @use_self_collision.setter
    def use_self_collision(self, value):
        """ """
        self._use_self_collision = value

    @property
    def vhacd_enabled(self):
        """ """
        return self._vhacd_enabled

    @vhacd_enabled.setter
    def vhacd_enabled(self, value):
        """ """
        self._vhacd_enabled = value

    @property
    def vhacd_params(self):
        """ """
        return self._vhacd_params

    @vhacd_params.setter
    def vhacd_params(self, value):
        """ """
        self._vhacd_params = value

    @property
    def mesh_normal_mode(self):
        """ """
        return self._mesh_normal_mode

    @mesh_normal_mode.setter
    def mesh_normal_mode(self, value):
        """ """
        self._mesh_normal_mode = value

    @property
    def initial_dof_position(self):
        """ """
        return self._initial_dof_position

    @initial_dof_position.setter
    def initial_dof_position(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError(
                    "'initial_dof_position' must have a number of dimensions of 1 or 2"
                )
        self._initial_dof_position = value

    @property
    def initial_dof_velocity(self):
        """ """
        return self._initial_dof_velocity

    @initial_dof_velocity.setter
    def initial_dof_velocity(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError(
                    "'initial_dof_velocity' must have a number of dimensions of 1 or 2"
                )
        self._initial_dof_velocity = value

    @property
    def link_color(self):
        """ """
        return self._link_color

    @link_color.setter
    def link_color(self, value):
        """ """
        assert not self._attr_array_locked["link_color"], (
            "'link_color' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_color"],
                self._ATTR_ARRAY_NDIM["link_color"] + 1,
            ):
                raise ValueError(
                    "'link_color' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_color']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_color'] + 1}"
                )
        self._link_color = value

    @property
    def link_collision_filter(self):
        """ """
        return self._link_collision_filter

    @link_collision_filter.setter
    def link_collision_filter(self, value):
        """ """
        assert not self._attr_array_locked["link_collision_filter"], (
            "'link_collision_filter' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.int64)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_collision_filter"],
                self._ATTR_ARRAY_NDIM["link_collision_filter"] + 1,
            ):
                raise ValueError(
                    "'link_collision_filter' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_collision_filter']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_collision_filter'] + 1}"
                )
        self._link_collision_filter = value

    @property
    def link_lateral_friction(self):
        """ """
        return self._link_lateral_friction

    @link_lateral_friction.setter
    def link_lateral_friction(self, value):
        """ """
        assert not self._attr_array_locked["link_lateral_friction"], (
            "'link_lateral_friction' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_lateral_friction"],
                self._ATTR_ARRAY_NDIM["link_lateral_friction"] + 1,
            ):
                raise ValueError(
                    "'link_lateral_friction' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_lateral_friction']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_lateral_friction'] + 1}"
                )
        self._link_lateral_friction = value

    @property
    def link_spinning_friction(self):
        """ """
        return self._link_spinning_friction

    @link_spinning_friction.setter
    def link_spinning_friction(self, value):
        """ """
        assert not self._attr_array_locked["link_spinning_friction"], (
            "'link_spinning_friction' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_spinning_friction"],
                self._ATTR_ARRAY_NDIM["link_spinning_friction"] + 1,
            ):
                raise ValueError(
                    "'link_spinning_friction' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_spinning_friction']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_spinning_friction'] + 1}"
                )
        self._link_spinning_friction = value

    @property
    def link_rolling_friction(self):
        """ """
        return self._link_rolling_friction

    @link_rolling_friction.setter
    def link_rolling_friction(self, value):
        """ """
        assert not self._attr_array_locked["link_rolling_friction"], (
            "'link_rolling_friction' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_rolling_friction"],
                self._ATTR_ARRAY_NDIM["link_rolling_friction"] + 1,
            ):
                raise ValueError(
                    "'link_rolling_friction' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_rolling_friction']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_rolling_friction'] + 1}"
                )
        self._link_rolling_friction = value

    @property
    def link_restitution(self):
        """ """
        return self._link_restitution

    @link_restitution.setter
    def link_restitution(self, value):
        """ """
        assert not self._attr_array_locked["link_restitution"], (
            "'link_restitution' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_restitution"],
                self._ATTR_ARRAY_NDIM["link_restitution"] + 1,
            ):
                raise ValueError(
                    "'link_restitution' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_restitution']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_restitution'] + 1}"
                )
        self._link_restitution = value

    @property
    def link_linear_damping(self):
        """ """
        return self._link_linear_damping

    @link_linear_damping.setter
    def link_linear_damping(self, value):
        """ """
        assert not self._attr_array_locked["link_linear_damping"], (
            "'link_linear_damping' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_linear_damping"],
                self._ATTR_ARRAY_NDIM["link_linear_damping"] + 1,
            ):
                raise ValueError(
                    "'link_linear_damping' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_linear_damping']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_linear_damping'] + 1}"
                )
        self._link_linear_damping = value

    @property
    def link_angular_damping(self):
        """ """
        return self._link_angular_damping

    @link_angular_damping.setter
    def link_angular_damping(self, value):
        """ """
        assert not self._attr_array_locked["link_angular_damping"], (
            "'link_angular_damping' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["link_angular_damping"],
                self._ATTR_ARRAY_NDIM["link_angular_damping"] + 1,
            ):
                raise ValueError(
                    "'link_angular_damping' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['link_angular_damping']} or "
                    f"{self._ATTR_ARRAY_NDIM['link_angular_damping'] + 1}"
                )
        self._link_angular_damping = value

    @property
    def dof_control_mode(self):
        """ """
        return self._dof_control_mode

    @dof_control_mode.setter
    def dof_control_mode(self, value):
        """ """
        assert not self._attr_array_locked["dof_control_mode"], (
            "'dof_control_mode' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.int64)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["dof_control_mode"],
                self._ATTR_ARRAY_NDIM["dof_control_mode"] + 1,
            ):
                raise ValueError(
                    "'dof_control_mode' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['dof_control_mode']} or "
                    f"{self._ATTR_ARRAY_NDIM['dof_control_mode'] + 1}"
                )
        self._dof_control_mode = value

    @property
    def dof_max_force(self):
        """ """
        return self._dof_max_force

    @dof_max_force.setter
    def dof_max_force(self, value):
        """ """
        assert not self._attr_array_locked["dof_max_force"], (
            "'dof_max_force' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["dof_max_force"],
                self._ATTR_ARRAY_NDIM["dof_max_force"] + 1,
            ):
                raise ValueError(
                    "'dof_max_force' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['dof_max_force']} or "
                    f"{self._ATTR_ARRAY_NDIM['dof_max_force'] + 1}"
                )
        self._dof_max_force = value

    @property
    def dof_max_velocity(self):
        """ """
        return self._dof_max_velocity

    @dof_max_velocity.setter
    def dof_max_velocity(self, value):
        """ """
        assert not self._attr_array_locked["dof_max_velocity"], (
            "'dof_max_velocity' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["dof_max_velocity"],
                self._ATTR_ARRAY_NDIM["dof_max_velocity"] + 1,
            ):
                raise ValueError(
                    "'dof_max_velocity' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['dof_max_velocity']} or "
                    f"{self._ATTR_ARRAY_NDIM['dof_max_velocity'] + 1}"
                )
        self._dof_max_velocity = value

    @property
    def dof_position_gain(self):
        """ """
        return self._dof_position_gain

    @dof_position_gain.setter
    def dof_position_gain(self, value):
        """ """
        assert not self._attr_array_locked["dof_position_gain"], (
            "'dof_position_gain' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["dof_position_gain"],
                self._ATTR_ARRAY_NDIM["dof_position_gain"] + 1,
            ):
                raise ValueError(
                    "'dof_position_gain' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['dof_position_gain']} or "
                    f"{self._ATTR_ARRAY_NDIM['dof_position_gain'] + 1}"
                )
        self._dof_position_gain = value

    @property
    def dof_velocity_gain(self):
        """ """
        return self._dof_velocity_gain

    @dof_velocity_gain.setter
    def dof_velocity_gain(self, value):
        """ """
        assert not self._attr_array_locked["dof_velocity_gain"], (
            "'dof_velocity_gain' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["dof_velocity_gain"],
                self._ATTR_ARRAY_NDIM["dof_velocity_gain"] + 1,
            ):
                raise ValueError(
                    "'dof_velocity_gain' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['dof_velocity_gain']} or "
                    f"{self._ATTR_ARRAY_NDIM['dof_velocity_gain'] + 1}"
                )
        self._dof_velocity_gain = value

    @property
    def dof_armature(self):
        """ """
        return self._dof_armature

    @dof_armature.setter
    def dof_armature(self, value):
        """ """
        assert not self._attr_array_locked["dof_armature"], (
            "'dof_armature' cannot be directly changed after simulation starts. Use "
            "'update_attr_array()'."
        )
        if value is not None:
            value = np.asanyarray(value, dtype=np.float32)
            if value.ndim not in (
                self._ATTR_ARRAY_NDIM["dof_armature"],
                self._ATTR_ARRAY_NDIM["dof_armature"] + 1,
            ):
                raise ValueError(
                    "'dof_armature' must have a number of dimensions of "
                    f"{self._ATTR_ARRAY_NDIM['dof_armature']} or "
                    f"{self._ATTR_ARRAY_NDIM['dof_armature'] + 1}"
                )
        self._dof_armature = value

    @property
    def dof_target_position(self):
        """ """
        return self._dof_target_position

    @dof_target_position.setter
    def dof_target_position(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError("'dof_target_position' must have a number of dimensions of 1 or 2")
        self._dof_target_position = value

    @property
    def dof_target_velocity(self):
        """ """
        return self._dof_target_velocity

    @dof_target_velocity.setter
    def dof_target_velocity(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError("'dof_target_velocity' must have a number of dimensions of 1 or 2")
        self._dof_target_velocity = value

    @property
    def dof_force(self):
        """ """
        return self._dof_force

    @dof_force.setter
    def dof_force(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim not in (1, 2):
                raise ValueError("'dof_force' must have a number of dimensions of 1 or 2")
        self._dof_force = value

    @property
    def env_ids_reset_dof_state(self):
        """ """
        return self._env_ids_reset_dof_state

    @env_ids_reset_dof_state.setter
    def env_ids_reset_dof_state(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
            if value.ndim != 1:
                raise ValueError("'env_ids_reset_dof_state' must have a number of dimensions of 1")
        self._env_ids_reset_dof_state = value

    @property
    def dof_state(self):
        """ """
        return self._dof_state

    @dof_state.setter
    def dof_state(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim != 3:
                raise ValueError("'dof_state' must have a number of dimensions of 3")
        self._dof_state = value

    @property
    def link_state(self):
        """ """
        return self._link_state

    @link_state.setter
    def link_state(self, value):
        """ """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim != 3:
                raise ValueError("'link_state' must have a number of dimensions of 3")
        self._link_state = value

    @property
    def contact_id(self):
        """ """
        return self._contact_id

    @contact_id.setter
    def contact_id(self, value):
        """ """
        if value is not None:
            value = np.asanyarray(value, dtype=np.int64)
            if value.ndim != 0:
                raise ValueError("'contact_id' must have a number of dimensions of 0")
        self._contact_id = value

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

        with self._make_attr_array_writeable(attr):
            getattr(self, attr)[env_ids] = value

        if not self._attr_array_dirty_flag[attr]:
            self._attr_array_dirty_flag[attr] = True
        try:
            self._attr_array_dirty_mask[attr][env_ids] = True
        except KeyError:
            self._attr_array_dirty_mask[attr] = np.zeros(len(getattr(self, attr)), dtype=bool)
            self._attr_array_dirty_mask[attr][env_ids] = True

    @contextmanager
    def _make_attr_array_writeable(self, attr):
        """ """
        try:
            getattr(self, attr).flags.writeable = True
            yield
        finally:
            getattr(self, attr).flags.writeable = False

    @property
    def attr_array_dirty_flag(self):
        """ """
        return self._attr_array_dirty_flag

    @property
    def attr_array_dirty_mask(self):
        """ """
        return self._attr_array_dirty_mask
