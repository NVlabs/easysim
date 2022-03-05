# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import os
import numpy as np
import torch
import time

try:
    from isaacgym import gymapi, gymtorch, gymutil
except ImportError:
    # A temporary workaround for requirement of importing `issacgym` before `torch` in
    # `isaacgym/gymdeps.py`. Remove this exception handler if this requirement has been removed.
    torch_module = sys.modules.pop("torch")
    from isaacgym import gymapi

    sys.modules["torch"] = torch_module
    from isaacgym import gymtorch, gymutil

from easysim.simulators.simulator import Simulator
from easysim.constants import DoFControlMode, MeshNormalMode
from easysim.contact import create_contact_array


class IsaacGym(Simulator):
    """Isaac Gym simulator."""

    _DOF_CONTROL_MODE_MAP = {
        DoFControlMode.NONE: gymapi.DOF_MODE_NONE,
        DoFControlMode.POSITION_CONTROL: gymapi.DOF_MODE_POS,
        DoFControlMode.VELOCITY_CONTROL: gymapi.DOF_MODE_VEL,
        DoFControlMode.TORQUE_CONTROL: gymapi.DOF_MODE_EFFORT,
    }
    _MESH_NORMAL_MODE_MAP = {
        MeshNormalMode.FROM_ASSET: gymapi.FROM_ASSET,
        MeshNormalMode.COMPUTE_PER_VERTEX: gymapi.COMPUTE_PER_VERTEX,
        MeshNormalMode.COMPUTE_PER_FACE: gymapi.COMPUTE_PER_FACE,
    }

    def __init__(self, cfg):
        """ """
        super().__init__(cfg)

        x = self._cfg.SIM_DEVICE.split(":")
        sim_device_type = x[0]
        if len(x) > 1:
            self._sim_device_id = int(x[1])
        else:
            self._sim_device_id = 0

        self._device = "cpu"
        if self._cfg.USE_GPU_PIPELINE:
            if sim_device_type == "cuda":
                self._device = "cuda:" + str(self._sim_device_id)
            else:
                print("GPU pipeline can only be used with GPU simulation. Forcing CPU pipeline.")
                self._cfg.USE_GPU_PIPELINE = False

        if not self._cfg.RENDER and self._cfg.GRAPHICS_DEVICE_ID != -1:
            self._cfg.GRAPHICS_DEVICE_ID = -1

        # Support only PhysX for now.
        self._physics_engine = gymapi.SIM_PHYSX
        self._sim_params = self._parse_sim_params(self._cfg, sim_device_type)

        self._num_envs = self._cfg.NUM_ENVS

        self._gym = gymapi.acquire_gym()

        self._created = False
        self._last_render_time = 0.0
        self._counter_render = 0
        self._render_time_step = max(1.0 / self._cfg.RENDER_FRAME_RATE, self._cfg.TIME_STEP)
        self._render_steps = self._render_time_step / self._cfg.TIME_STEP

    def _parse_sim_params(self, cfg, sim_device_type):
        """ """
        sim_params = gymapi.SimParams()

        if cfg.USE_DEFAULT_STEP_PARAMS:
            cfg.TIME_STEP = sim_params.dt
            cfg.SUBSTEPS = sim_params.substeps
        else:
            sim_params.dt = cfg.TIME_STEP
            sim_params.substeps = cfg.SUBSTEPS
        sim_params.gravity = gymapi.Vec3(*cfg.GRAVITY)
        sim_params.up_axis = gymapi.UP_AXIS_Z

        sim_params.physx.use_gpu = sim_device_type == "cuda"

        sim_params.use_gpu_pipeline = cfg.USE_GPU_PIPELINE

        return sim_params

    def reset(self, bodies, env_ids):
        """ """
        if not self._created:
            self._sim = self._create_sim(
                self._sim_device_id,
                self._cfg.GRAPHICS_DEVICE_ID,
                self._physics_engine,
                self._sim_params,
            )

            self._load_ground_plane()
            self._load_assets(bodies)
            self._create_envs(
                self._num_envs, self._cfg.SPACING, int(np.sqrt(self._num_envs)), bodies
            )
            self._cache_and_set_props(bodies)
            self._set_callback(bodies)

            self._gym.prepare_sim(self._sim)
            self._acquire_physics_state_tensors()

            self._set_viewer()
            self._allocate_buffers()

            self._created = True

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self._reset_idx(bodies, env_ids)

        self._clear_state(bodies)
        self._contact = None

    def _create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        """ """
        sim = self._gym.create_sim(
            compute_device=compute_device,
            graphics_device=graphics_device,
            type=physics_engine,
            params=sim_params,
        )
        if sim is None:
            raise RuntimeError("Failed to create sim")

        return sim

    def _load_ground_plane(self):
        """ """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self._gym.add_ground(self._sim, plane_params)

    def _load_assets(self, bodies):
        """ """
        self._assets = {}

        self._asset_dof_slice = {}
        self._asset_rigid_body_slice = {}
        self._asset_rigid_body_mapping = {-1: [0, 0]}
        self._asset_rigid_shape_count = {}

        counter_dof = 0
        counter_rigid_body = 0

        for b, body in enumerate(bodies):
            asset_root, asset_file = os.path.split(body.urdf_file)

            asset_options = gymapi.AssetOptions()
            if body.use_fixed_base is not None:
                asset_options.fix_base_link = body.use_fixed_base
            if body.use_self_collision is not None:
                raise ValueError(
                    "For Isaac Gym, keep 'use_self_collision' to None and set self-collision with "
                    f"'collision_filter' (0: enabled): '{body.name}'"
                )
            if body.link_linear_damping is not None:
                if body.link_linear_damping.ndim != 0:
                    raise ValueError(
                        "For Isaac Gym, 'link_linear_damping' must have a number of dimensions of "
                        f"0: '{body.name}'"
                    )
                asset_options.linear_damping = body.link_linear_damping
            if body.link_angular_damping is not None:
                if body.link_angular_damping.ndim != 0:
                    raise ValueError(
                        "For Isaac Gym, 'link_angular_damping' must have a number of dimensions of "
                        f"0: '{body.name}'"
                    )
                asset_options.angular_damping = body.link_angular_damping
            asset_options.override_com = True
            asset_options.override_inertia = True
            if body.vhacd_enabled is not None:
                asset_options.vhacd_enabled = body.vhacd_enabled
            if body.vhacd_params is not None:
                for attr in body.vhacd_params:
                    setattr(asset_options.vhacd_params, attr, body.vhacd_params[attr])
            asset_options.use_mesh_materials = True
            if body.mesh_normal_mode is not None:
                asset_options.mesh_normal_mode = self._MESH_NORMAL_MODE_MAP[body.mesh_normal_mode]

            self._assets[body.name] = self._gym.load_asset(
                self._sim, asset_root, asset_file, options=asset_options
            )

            num_dofs = self._gym.get_asset_dof_count(self._assets[body.name])
            self._asset_dof_slice[body.name] = slice(counter_dof, counter_dof + num_dofs)
            counter_dof += num_dofs

            num_rigid_bodies = self._gym.get_asset_rigid_body_count(self._assets[body.name])
            self._asset_rigid_body_slice[body.name] = slice(
                counter_rigid_body, counter_rigid_body + num_rigid_bodies
            )
            for i in range(num_rigid_bodies):
                self._asset_rigid_body_mapping[counter_rigid_body + i] = [b, i]
            counter_rigid_body += num_rigid_bodies

            self._asset_rigid_shape_count[body.name] = self._gym.get_asset_rigid_shape_count(
                self._assets[body.name]
            )

            body.contact_id = b

    def _create_envs(self, num_envs, spacing, num_per_row, bodies):
        """ """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(+spacing, +spacing, spacing)

        self._envs = []

        self._actor_handles = [{} for _ in range(num_envs)]
        self._actor_indices = [[] for _ in range(num_envs)]

        for i in range(num_envs):
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)

            for body in bodies:
                actor_handle = self._gym.create_actor(
                    env_ptr, self._assets[body.name], gymapi.Transform(), name=body.name, group=i
                )
                actor_index = self._gym.get_actor_index(env_ptr, actor_handle, gymapi.DOMAIN_SIM)
                self._actor_handles[i][body.name] = actor_handle
                self._actor_indices[i].append(actor_index)

            self._envs.append(env_ptr)

        self._actor_indices = torch.tensor(
            self._actor_indices, dtype=torch.int32, device=self._device
        )

    def _acquire_physics_state_tensors(self):
        """ """
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state = self._gym.acquire_dof_state_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)

        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        self._actor_root_state = gymtorch.wrap_tensor(actor_root_state)
        self._dof_state = gymtorch.wrap_tensor(dof_state)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        if self._actor_root_state is None:
            self._initial_actor_root_state = None
        else:
            self._initial_actor_root_state = self._actor_root_state.clone()
        if self._dof_state is None:
            self._initial_dof_state = None
        else:
            self._initial_dof_state = self._dof_state.clone()

    def _set_viewer(self):
        """ """
        self._enable_viewer_sync = True
        self._viewer = None

        if self._cfg.RENDER:
            self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_ESCAPE, "quit")
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

            axes_geom = gymutil.AxesGeometry(1.0)
            for env_ptr in self._envs:
                gymutil.draw_lines(axes_geom, self._gym, self._viewer, env_ptr, gymapi.Transform())

            if (
                self._cfg.INIT_VIEWER_CAMERA_POSITION
                != (
                    None,
                    None,
                    None,
                )
                and self._cfg.INIT_VIEWER_CAMERA_TARGET != (None, None, None)
            ):
                cam_pos = gymapi.Vec3(*self._cfg.INIT_VIEWER_CAMERA_POSITION)
                cam_target = gymapi.Vec3(*self._cfg.INIT_VIEWER_CAMERA_TARGET)

                self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

    def _allocate_buffers(self):
        """ """
        if self._dof_state is None:
            self._dof_control_buffer = None
        else:
            self._dof_control_buffer = torch.zeros(
                (self._num_envs, len(self._dof_state) // self._num_envs),
                dtype=torch.float32,
                device=self._device,
            )

    def _cache_and_set_props(self, bodies):
        """ """
        self._bodies = type(bodies)()

        for body in bodies:
            x = type(body)()
            x.name = body.name
            self._bodies.append(x)

            for attr in (
                "dof_control_mode",
                "dof_max_force",
                "dof_max_velocity",
                "dof_position_gain",
                "dof_velocity_gain",
                "dof_armature",
            ):
                if getattr(body, attr) is not None:
                    if self._get_slice_length(self._asset_dof_slice[body.name]) == 0:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )

            for idx in range(self._num_envs):
                if body.link_color is not None:
                    self._set_link_color(body, idx)

                if (
                    body.link_collision_filter is not None
                    or body.link_lateral_friction is not None
                    or body.link_spinning_friction is not None
                    or body.link_rolling_friction is not None
                    or body.link_restitution is not None
                ):
                    self._set_rigid_shape_props(body, idx)

                if self._get_slice_length(self._asset_dof_slice[body.name]) > 0 and (
                    body.dof_control_mode is not None
                    or body.dof_max_velocity is not None
                    or body.dof_max_force is not None
                    or body.dof_position_gain is not None
                    or body.dof_velocity_gain is not None
                    or body.dof_armature is not None
                ):
                    self._set_dof_props(body, idx, set_drive_mode=True)

            body.lock_attr_array()

    def _get_slice_length(self, slice_):
        """ """
        return slice_.stop - slice_.start

    def _set_link_color(self, body, idx):
        """ """
        link_color = body.get_attr_array("link_color", idx)
        if len(link_color) != self._get_slice_length(self._asset_rigid_body_slice[body.name]):
            raise ValueError(
                f"Size of 'link_color' in the link dimension ({len(link_color)}) should match the "
                "number of links "
                f"({self._get_slice_length(self._asset_rigid_body_slice[body.name])}): "
                f"'{body.name}'"
            )
        for i in range(self._get_slice_length(self._asset_rigid_body_slice[body.name])):
            self._gym.set_rigid_body_color(
                self._envs[idx],
                self._actor_handles[idx][body.name],
                i,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*link_color[i]),
            )

    def _set_rigid_shape_props(self, body, idx):
        """ """
        for attr in (
            "link_collision_filter",
            "link_lateral_friction",
            "link_spinning_friction",
            "link_rolling_friction",
            "link_restitution",
        ):
            if (
                getattr(body, attr) is not None
                and len(body.get_attr_array(attr, idx)) != self._asset_rigid_shape_count[body.name]
            ):
                raise ValueError(
                    f"Size of '{attr}' in the link dimension "
                    f"({len(body.get_attr_array(attr, idx))}) should match the number of rigid "
                    f"shapes ({self._asset_rigid_shape_count[body.name]}): '{body.name}'"
                )
        rigid_shape_props = self._gym.get_asset_rigid_shape_properties(self._assets[body.name])
        if body.link_collision_filter is not None:
            link_collision_filter = body.get_attr_array("link_collision_filter", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.filter = link_collision_filter[i]
        if body.link_lateral_friction is not None:
            link_lateral_friction = body.get_attr_array("link_lateral_friction", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.friction = link_lateral_friction[i]
        if body.link_spinning_friction is not None:
            link_spinning_friction = body.get_attr_array("link_spinning_friction", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.torsion_friction = link_spinning_friction[i]
        if body.link_rolling_friction is not None:
            link_rolling_friction = body.get_attr_array("link_rolling_friction", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.rolling_fiction = link_rolling_friction[i]
        if body.link_restitution is not None:
            link_restitution = body.get_attr_array("link_restitution", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.restitution = link_restitution[i]
        self._gym.set_actor_rigid_shape_properties(
            self._envs[idx], self._actor_handles[idx][body.name], rigid_shape_props
        )

    def _set_dof_props(self, body, idx, set_drive_mode=False):
        """ """
        dof_props = self._gym.get_actor_dof_properties(
            self._envs[idx], self._actor_handles[idx][body.name]
        )
        if set_drive_mode:
            if body.dof_control_mode is not None:
                if body.dof_control_mode.ndim == 0:
                    dof_props["driveMode"] = self._DOF_CONTROL_MODE_MAP[
                        body.dof_control_mode.item()
                    ]
                if body.dof_control_mode.ndim == 1:
                    dof_props["driveMode"] = [
                        self._DOF_CONTROL_MODE_MAP[x] for x in body.dof_control_mode
                    ]
        if body.dof_max_velocity is not None:
            dof_props["velocity"] = body.get_attr_array("dof_max_velocity", idx)
        if body.dof_max_force is not None:
            dof_props["effort"] = body.get_attr_array("dof_max_force", idx)
        if body.dof_position_gain is not None:
            dof_props["stiffness"] = body.get_attr_array("dof_position_gain", idx)
        if body.dof_velocity_gain is not None:
            dof_props["damping"] = body.get_attr_array("dof_velocity_gain", idx)
        if body.dof_armature is not None:
            dof_props["armature"] = body.get_attr_array("dof_armature", idx)
        self._gym.set_actor_dof_properties(
            self._envs[idx], self._actor_handles[idx][body.name], dof_props
        )

    def _set_callback(self, bodies):
        """ """
        for body in bodies:
            body.set_callback_collect_dof_state(self._collect_dof_state)
            body.set_callback_collect_link_state(self._collect_link_state)

    def _collect_dof_state(self, body):
        """ """
        if not self._dof_state_refreshed:
            self._gym.refresh_dof_state_tensor(self._sim)
            self._dof_state_refreshed = True

        if self._get_slice_length(self._asset_dof_slice[body.name]) > 0:
            body.dof_state = self._dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name]
            ].clone()

    def _collect_link_state(self, body):
        """ """
        if not self._link_state_refreshed:
            self._gym.refresh_rigid_body_state_tensor(self._sim)
            self._link_state_refreshed = True

        body.link_state = self._rigid_body_state.view(self._num_envs, -1, 13)[
            :, self._asset_rigid_body_slice[body.name]
        ].clone()

    def _reset_idx(self, bodies, env_ids):
        """ """
        if [body.name for body in bodies] != [body.name for body in self._bodies]:
            raise ValueError(
                "For Isaac Gym, the list of bodies cannot be altered after the first reset"
            )

        for b, body in enumerate(bodies):
            if body.initial_base_position is None:
                self._actor_root_state.view(self._num_envs, len(bodies), 13)[
                    :, b, :7
                ] = self._initial_actor_root_state.view(self._num_envs, len(bodies), 13)[:, b, :7]
            else:
                self._actor_root_state.view(self._num_envs, len(bodies), 13)[
                    :, b, :7
                ] = body.initial_base_position
            if body.initial_base_velocity is None:
                self._actor_root_state.view(self._num_envs, len(bodies), 13)[
                    :, b, 7:
                ] = self._initial_actor_root_state.view(self._num_envs, len(bodies), 13)[:, b, 7:]
            else:
                self._actor_root_state.view(self._num_envs, len(bodies), 13)[
                    :, b, 7:
                ] = body.initial_base_velocity

            if self._get_slice_length(self._asset_dof_slice[body.name]) == 0:
                for attr in ("initial_dof_position", "initial_dof_velocity"):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )
            else:
                self._reset_dof_state_buffer(body)

        # Reset base state.
        if self._actor_root_state is not None:
            actor_indices = self._actor_indices[env_ids].view(-1)
            self._gym.set_actor_root_state_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._actor_root_state),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )

        # Reset DoF state.
        if self._dof_state is not None:
            actor_indices = self._actor_indices[
                env_ids[:, None],
                [self._get_slice_length(self._asset_dof_slice[body.name]) > 0 for body in bodies],
            ].view(-1)
            self._gym.set_dof_state_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )

        self._check_and_update_props(bodies, env_ids=env_ids)

    def _reset_dof_state_buffer(self, body):
        """ """
        if body.initial_dof_position is None:
            self._dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name], 0
            ] = self._initial_dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name], 0
            ]
        else:
            self._dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name], 0
            ] = body.initial_dof_position
        if body.initial_dof_velocity is None:
            self._dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name], 1
            ] = self._initial_dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name], 1
            ]
        else:
            self._dof_state.view(self._num_envs, -1, 2)[
                :, self._asset_dof_slice[body.name], 1
            ] = body.initial_dof_velocity

    def _check_and_update_props(self, bodies, env_ids=None):
        """ """
        for body in bodies:
            for attr in ("link_color",):
                if body.attr_array_dirty_flag[attr]:
                    if env_ids is not None and not np.all(
                        np.isin(np.nonzero(body.attr_array_dirty_mask[attr])[0], env_ids.cpu())
                    ):
                        raise ValueError(
                            f"For Isaac Gym, to change '{attr}' for some env also requires the env "
                            f"indices to be in `env_ids`: '{body.name}'"
                        )
                    env_ids_masked = np.nonzero(body.attr_array_dirty_mask[attr])[0]
                    for idx in env_ids_masked:
                        if attr == "link_color":
                            self._set_link_color(body, idx)
                    body.attr_array_dirty_flag[attr] = False
                    body.attr_array_dirty_mask[attr][:] = False

            attr_rigid_shape_props = (
                "link_collision_filter",
                "link_lateral_friction",
                "link_spinning_friction",
                "link_rolling_friction",
                "link_restitution",
            )
            if any(body.attr_array_dirty_flag[x] for x in attr_rigid_shape_props):
                mask = np.zeros(self._num_envs, dtype=bool)
                for attr in attr_rigid_shape_props:
                    if body.attr_array_dirty_flag[attr]:
                        if env_ids is not None and not np.all(
                            np.isin(np.nonzero(body.attr_array_dirty_mask[attr])[0], env_ids.cpu())
                        ):
                            raise ValueError(
                                f"For Isaac Gym, to change '{attr}' for some env also requires the env "
                                f"indices to be in `env_ids`: '{body.name}'"
                            )
                        mask |= body.attr_array_dirty_mask[attr]
                        body.attr_array_dirty_flag[attr] = False
                        body.attr_array_dirty_mask[attr][:] = False
                env_ids_masked = np.nonzero(mask)[0]
                for idx in env_ids_masked:
                    self._set_rigid_shape_props(body, idx)

            for attr in ("link_linear_damping", "link_angular_damping"):
                if body.attr_array_dirty_flag[attr]:
                    raise ValueError(
                        f"For Isaac Gym, '{attr}' cannot be changed after the first reset: "
                        f"'{body.name}'"
                    )

            if self._get_slice_length(self._asset_dof_slice[body.name]) > 0:
                if body.attr_array_dirty_flag["dof_control_mode"]:
                    raise ValueError(
                        "For Isaac Gym, 'dof_control_mode' cannot be changed after the first "
                        f"reset: '{body.name}'"
                    )
                attr_dof_props = (
                    "dof_max_force",
                    "dof_max_velocity",
                    "dof_position_gain",
                    "dof_velocity_gain",
                    "dof_armature",
                )
                if any(body.attr_array_dirty_flag[x] for x in attr_dof_props):
                    mask = np.zeros(self._num_envs, dtype=bool)
                    for attr in attr_dof_props:
                        if body.attr_array_dirty_flag[attr]:
                            if env_ids is not None and not np.all(
                                np.isin(
                                    np.nonzero(body.attr_array_dirty_mask[attr])[0], env_ids.cpu()
                                )
                            ):
                                raise ValueError(
                                    f"For Isaac Gym, to change '{attr}' for certain env also requires "
                                    f"the env index to be in `env_ids`: '{body.name}'"
                                )
                            mask |= body.attr_array_dirty_mask[attr]
                            body.attr_array_dirty_flag[attr] = False
                            body.attr_array_dirty_mask[attr][:] = False
                    env_ids_masked = np.nonzero(mask)[0]
                    for idx in env_ids_masked:
                        self._set_dof_props(body, idx)
            else:
                for attr in (
                    "dof_control_mode",
                    "dof_max_force",
                    "dof_max_velocity",
                    "dof_position_gain",
                    "dof_velocity_gain",
                    "dof_armature",
                ):
                    if body.attr_array_dirty_flag[attr]:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )

    def _clear_state(self, bodies):
        """ """
        for body in bodies:
            body.dof_state = None
            body.link_state = None

        self._dof_state_refreshed = False
        self._link_state_refreshed = False

    def step(self, bodies):
        """ """
        if [body.name for body in bodies] != [body.name for body in self._bodies]:
            raise ValueError(
                "For Isaac Gym, the list of bodies cannot be altered after the first reset"
            )

        self._check_and_update_props(bodies)

        reset_dof_state = False
        actor_indices = []

        for b, body in enumerate(bodies):
            if self._get_slice_length(self._asset_dof_slice[body.name]) == 0:
                for attr in (
                    "dof_target_position",
                    "dof_target_velocity",
                    "dof_force",
                    "env_ids_reset_dof_state",
                ):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )
                continue

            if body.env_ids_reset_dof_state is not None:
                self._reset_dof_state_buffer(body)
                if not reset_dof_state:
                    reset_dof_state = True
                actor_indices.append(self._actor_indices[body.env_ids_reset_dof_state, b])
                body.env_ids_reset_dof_state = None

            if body.dof_target_position is not None and (
                body.dof_control_mode is None
                or body.dof_control_mode.ndim == 0
                and body.dof_control_mode != DoFControlMode.POSITION_CONTROL
                or body.dof_control_mode.ndim == 1
                and DoFControlMode.POSITION_CONTROL not in body.dof_control_mode
            ):
                raise ValueError(
                    "For Isaac Gym, 'dof_target_position' can only be set in the POSITION_CONTROL "
                    f"mode: '{body.name}'"
                )
            if body.dof_target_velocity is not None and (
                body.dof_control_mode is None
                or body.dof_control_mode.ndim == 0
                and body.dof_control_mode != DoFControlMode.VELOCITY_CONTROL
                or body.dof_control_mode.ndim == 1
                and DoFControlMode.VELOCITY_CONTROL not in body.dof_control_mode
            ):
                raise ValueError(
                    "For Isaac Gym, 'dof_target_velocity' can only be set in the VELOCITY_CONTROL "
                    f"mode: '{body.name}'"
                )
            if body.dof_force is not None and (
                body.dof_control_mode is None
                or body.dof_control_mode.ndim == 0
                and body.dof_control_mode != DoFControlMode.TORQUE_CONTROL
                or body.dof_control_mode.ndim == 1
                and DoFControlMode.TORQUE_CONTROL not in body.dof_control_mode
            ):
                raise ValueError(
                    "For Isaac Gym, 'dof_force' can only be set in the TORQUE_CONTROL mode: "
                    f"'{body.name}'"
                )

            # DriveMode is defaulted to DOF_MODE_NONE if dof_control_mode is None.
            if body.dof_control_mode is None:
                continue
            if body.dof_control_mode.ndim == 0:
                if body.dof_control_mode == DoFControlMode.POSITION_CONTROL:
                    self._dof_control_buffer[
                        :, self._asset_dof_slice[body.name]
                    ] = body.dof_target_position
                if body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL:
                    self._dof_control_buffer[
                        :, self._asset_dof_slice[body.name]
                    ] = body.dof_target_velocity
                if body.dof_control_mode == DoFControlMode.TORQUE_CONTROL:
                    self._dof_control_buffer[:, self._asset_dof_slice[body.name]] = body.dof_force
            if body.dof_control_mode.ndim == 1:
                if DoFControlMode.POSITION_CONTROL in body.dof_control_mode:
                    self._dof_control_buffer[
                        :,
                        [
                            x
                            for i, x in enumerate(
                                self._get_slice_range(self._asset_dof_slice[body.name])
                            )
                            if body.dof_control_mode[i] == DoFControlMode.POSITION_CONTROL
                        ],
                    ] = body.dof_target_position[
                        body.dof_control_mode == DoFControlMode.POSITION_CONTROL
                    ]
                if DoFControlMode.VELOCITY_CONTROL in body.dof_control_mode:
                    self._dof_control_buffer[
                        :,
                        [
                            x
                            for i, x in enumerate(
                                self._get_slice_range(self._asset_dof_slice[body.name])
                            )
                            if body.dof_control_mode[i] == DoFControlMode.VELOCITY_CONTROL
                        ],
                    ] = body.dof_target_velocity[
                        body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL
                    ]
                if DoFControlMode.TORQUE_CONTROL in body.dof_control_mode:
                    self._dof_control_buffer[
                        :,
                        [
                            x
                            for i, x in enumerate(
                                self._get_slice_range(self._asset_dof_slice[body.name])
                            )
                            if body.dof_control_mode[i] == DoFControlMode.TORQUE_CONTROL
                        ],
                    ] = body.dof_force[body.dof_control_mode == DoFControlMode.TORQUE_CONTROL]

        if reset_dof_state:
            actor_indices = torch.cat(actor_indices)
            self._gym.set_dof_state_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )

        if self._dof_control_buffer is not None:
            self._gym.set_dof_position_target_tensor(
                self._sim, gymtorch.unwrap_tensor(self._dof_control_buffer)
            )
            self._gym.set_dof_velocity_target_tensor(
                self._sim, gymtorch.unwrap_tensor(self._dof_control_buffer)
            )
            self._gym.set_dof_actuation_force_tensor(
                self._sim, gymtorch.unwrap_tensor(self._dof_control_buffer)
            )

        self._gym.simulate(self._sim)
        if self._device == "cpu" or self._viewer:
            self._gym.fetch_results(self._sim, True)

        if self._viewer:
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

            for evt in self._gym.query_viewer_action_events(self._viewer):
                if evt.action == "quit" and evt.value > 0:
                    sys.exit()
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self._enable_viewer_sync = not self._enable_viewer_sync

            if self._enable_viewer_sync:
                if (self._counter_render % self._render_steps) <= (
                    self._counter_render - 1
                ) % self._render_steps:
                    # Simulate real-time rendering with sleep if computation takes less than real time.
                    time_spent = time.time() - self._last_render_time
                    time_sleep = self._render_time_step - time_spent
                    if time_sleep > 0:
                        time.sleep(time_sleep)
                    self._last_render_time = time.time()

                    self._gym.step_graphics(self._sim)
                    self._gym.draw_viewer(self._viewer, self._sim)

                self._counter_render += 1
            else:
                self._gym.poll_viewer_events(self._viewer)

        self._clear_state(bodies)
        self._contact = None

    def _get_slice_range(self, slice_):
        """ """
        return range(slice_.start, slice_.stop)

    @property
    def contact(self):
        """ """
        if self._contact is None:
            self._contact = self._collect_contact()
        return self._contact

    def _collect_contact(self):
        """ """
        contact = []
        for env in self._envs:
            rigid_contacts = self._gym.get_env_rigid_contacts(env)
            if len(rigid_contacts) == 0:
                contact_array = create_contact_array(0)
            else:
                kwargs = {}
                kwargs["body_id_a"], kwargs["link_id_a"] = zip(
                    *[self._asset_rigid_body_mapping[x] for x in rigid_contacts["body0"]]
                )
                kwargs["body_id_b"], kwargs["link_id_b"] = zip(
                    *[self._asset_rigid_body_mapping[x] for x in rigid_contacts["body1"]]
                )
                kwargs["position_a_world"] = np.nan
                kwargs["position_b_world"] = np.nan
                kwargs["position_a_link"] = rigid_contacts["localPos0"]
                kwargs["position_b_link"] = rigid_contacts["localPos1"]
                kwargs["normal"] = rigid_contacts["normal"]
                kwargs["force"] = rigid_contacts["lambda"]
                contact_array = create_contact_array(len(rigid_contacts), **kwargs)
            contact.append(contact_array)
        return contact

    def close(self):
        """ """
        if self._created:
            self._gym.destroy_viewer(self._viewer)
            self._gym.destroy_sim(self._sim)
            self._created = False
