# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import sys
import os
import numpy as np
import torch
import time

try:
    from isaacgym import gymapi, gymtorch, gymutil
except ImportError:
    # A temporary workaround for requirement of importing `isaacgym` before `torch` in
    # `isaacgym/gymdeps.py`. Remove this exception handler if this requirement has been removed.
    torch_module = sys.modules.pop("torch")
    from isaacgym import gymapi

    sys.modules["torch"] = torch_module
    from isaacgym import gymtorch, gymutil

from easysim.simulators.simulator import Simulator
from easysim.constants import DescriptionType, DoFControlMode, MeshNormalMode
from easysim.contact import create_contact_array


class IsaacGym(Simulator):
    """Isaac Gym simulator."""

    _ATTR_RIGID_SHAPE_PROPS = (
        "link_collision_filter",
        "link_lateral_friction",
        "link_spinning_friction",
        "link_rolling_friction",
        "link_restitution",
    )
    _ATTR_DOF_PROPS = (
        "dof_has_limits",
        "dof_lower_limit",
        "dof_upper_limit",
        "dof_control_mode",
        "dof_max_force",
        "dof_max_velocity",
        "dof_position_gain",
        "dof_velocity_gain",
        "dof_armature",
    )
    _ATTR_PROJECTION_MATRIX = ("width", "height", "vertical_fov", "near", "far")
    _ATTR_VIEW_MATRIX = ("position", "target", "up_vector", "orientation")
    _MESH_NORMAL_MODE_MAP = {
        MeshNormalMode.FROM_ASSET: gymapi.FROM_ASSET,
        MeshNormalMode.COMPUTE_PER_VERTEX: gymapi.COMPUTE_PER_VERTEX,
        MeshNormalMode.COMPUTE_PER_FACE: gymapi.COMPUTE_PER_FACE,
    }
    _DOF_CONTROL_MODE_MAP = {
        DoFControlMode.NONE: gymapi.DOF_MODE_NONE,
        DoFControlMode.POSITION_CONTROL: gymapi.DOF_MODE_POS,
        DoFControlMode.VELOCITY_CONTROL: gymapi.DOF_MODE_VEL,
        DoFControlMode.TORQUE_CONTROL: gymapi.DOF_MODE_EFFORT,
    }

    def __init__(self, cfg, scene):
        """ """
        super().__init__(cfg, scene)

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

        if self._cfg.RENDER or self._cfg.ISAAC_GYM.ENABLE_CAMERA_SENSORS:
            self._graphics_device = "cuda:" + str(self._cfg.ISAAC_GYM.GRAPHICS_DEVICE_ID)
        elif self._cfg.ISAAC_GYM.GRAPHICS_DEVICE_ID != -1:
            self._cfg.ISAAC_GYM.GRAPHICS_DEVICE_ID = -1

        # Support only PhysX for now.
        self._physics_engine = gymapi.SIM_PHYSX
        self._sim_params = self._parse_sim_params(self._cfg, sim_device_type)

        self._num_envs = self._cfg.NUM_ENVS

        self._gym = gymapi.acquire_gym()

        self._created = False
        self._last_render_time = 0.0
        self._counter_render = 0
        self._render_time_step = max(
            1.0 / self._cfg.ISAAC_GYM.VIEWER.RENDER_FRAME_RATE, self._cfg.TIME_STEP
        )
        self._render_steps = self._render_time_step / self._cfg.TIME_STEP

    @property
    def device(self):
        """ """
        return self._device

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
        sim_params.use_gpu_pipeline = cfg.USE_GPU_PIPELINE

        sim_params.physx.use_gpu = sim_device_type == "cuda"
        sim_params.physx.max_depenetration_velocity = cfg.ISAAC_GYM.PHYSX.MAX_DEPENETRATION_VELOCITY
        sim_params.physx.contact_collection = gymapi.ContactCollection(
            cfg.ISAAC_GYM.PHYSX.CONTACT_COLLECTION
        )

        return sim_params

    def reset(self, env_ids):
        """ """
        if not self._created:
            self._sim = self._create_sim(
                self._sim_device_id,
                self._cfg.ISAAC_GYM.GRAPHICS_DEVICE_ID,
                self._physics_engine,
                self._sim_params,
            )

            if self._cfg.LOAD_GROUND_PLANE:
                self._load_ground_plane()
            self._load_assets()
            self._create_envs(
                self._num_envs, self._cfg.ISAAC_GYM.SPACING, int(np.sqrt(self._num_envs))
            )

            self._scene_cache = type(self._scene)()
            self._cache_body_and_set_props()
            self._set_body_device()
            self._set_body_callback()
            self._set_camera_props()
            self._set_camera_device()
            self._set_camera_callback()
            self._set_scene_callback()

            self._gym.prepare_sim(self._sim)
            self._acquire_physics_state_tensors()

            self._set_viewer()
            self._allocate_buffers()

            self._created = True

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self.device)

        self._reset_idx(env_ids)

        self._graphics_stepped = False
        self._clear_state()
        self._clear_image()
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
        plane_params.normal = gymapi.Vec3(x=0.0, y=0.0, z=1.0)
        plane_params.distance = self._cfg.GROUND_PLANE.DISTANCE
        self._gym.add_ground(self._sim, plane_params)

    def _load_assets(self):
        """ """
        self._assets = {}
        self._asset_num_dofs = {}
        self._asset_num_rigid_bodies = {}
        self._asset_num_rigid_shapes = {}

        for body in self._scene.bodies:
            asset_options = gymapi.AssetOptions()
            if body.simulator_config.isaac_gym.flip_visual_attachments is not None:
                asset_options.flip_visual_attachments = (
                    body.simulator_config.isaac_gym.flip_visual_attachments
                )
            if body.use_fixed_base is not None:
                asset_options.fix_base_link = body.use_fixed_base
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
            if body.simulator_config.isaac_gym.disable_gravity is not None:
                asset_options.disable_gravity = body.simulator_config.isaac_gym.disable_gravity
            asset_options.override_com = True
            asset_options.override_inertia = True
            if body.simulator_config.isaac_gym.vhacd_enabled is not None:
                asset_options.vhacd_enabled = body.simulator_config.isaac_gym.vhacd_enabled
            if body.simulator_config.isaac_gym.vhacd_params is not None:
                for attr in body.simulator_config.isaac_gym.vhacd_params:
                    setattr(
                        asset_options.vhacd_params,
                        attr,
                        body.simulator_config.isaac_gym.vhacd_params[attr],
                    )
            asset_options.use_mesh_materials = True
            if body.simulator_config.isaac_gym.mesh_normal_mode is not None:
                asset_options.mesh_normal_mode = self._MESH_NORMAL_MODE_MAP[
                    body.simulator_config.isaac_gym.mesh_normal_mode
                ]

            if body.description_type is None:
                raise ValueError(
                    f"For Isaac Gym, 'description_type' must not be None: '{body.name}'"
                )
            if body.description_type not in (
                DescriptionType.URDF,
                DescriptionType.SPHERE,
                DescriptionType.BOX,
            ):
                raise ValueError(
                    "For Isaac Gym, 'description_type' only supports URDF, SPHERE, and BOX: "
                    f"'{body.name}'"
                )
            if body.description_type == DescriptionType.URDF:
                for attr in ("sphere_radius", "box_half_extent"):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for geometry type URDF: '{body.name}'"
                        )
                asset_root, asset_file = os.path.split(body.urdf_file)
                self._assets[body.name] = self._gym.load_asset(
                    self._sim, asset_root, asset_file, options=asset_options
                )
            if body.description_type == DescriptionType.SPHERE:
                for attr in ("urdf_file", "box_half_extent"):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for geometry type SPHERE: '{body.name}'"
                        )
                if body.sphere_radius is None:
                    raise ValueError(
                        "For Isaac Gym, 'sphere_radius' must not be None if 'description_type' is "
                        f"set to SPHERE: '{body.name}'"
                    )
                self._assets[body.name] = self._gym.create_sphere(
                    self._sim, body.sphere_radius, options=asset_options
                )
            if body.description_type == DescriptionType.BOX:
                for attr in ("urdf_file", "sphere_radius"):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for geometry type BOX: '{body.name}'"
                        )
                if body.box_half_extent is None:
                    raise ValueError(
                        "For Isaac Gym, 'box_half_extent' must not be None if 'description_type' "
                        f"is set to BOX: '{body.name}'"
                    )
                self._assets[body.name] = self._gym.create_box(
                    self._sim, *[x * 2 for x in body.box_half_extent], options=asset_options
                )

            self._asset_num_dofs[body.name] = self._gym.get_asset_dof_count(self._assets[body.name])
            self._asset_num_rigid_bodies[body.name] = self._gym.get_asset_rigid_body_count(
                self._assets[body.name]
            )
            self._asset_num_rigid_shapes[body.name] = self._gym.get_asset_rigid_shape_count(
                self._assets[body.name]
            )

    def _create_envs(self, num_envs, spacing, num_per_row):
        """ """
        lower = gymapi.Vec3(x=-spacing, y=-spacing, z=0.0)
        upper = gymapi.Vec3(x=+spacing, y=+spacing, z=spacing)

        self._envs = []

        self._actor_handles = [{} for _ in range(num_envs)]
        self._actor_indices = [[] for _ in range(num_envs)]
        self._actor_indices_need_filter = False

        self._actor_root_indices = {body.name: [] for body in self._scene.bodies}
        self._dof_indices = {body.name: [] for body in self._scene.bodies}
        self._rigid_body_indices = {body.name: [] for body in self._scene.bodies}
        counter_actor = 0
        counter_dof = 0
        counter_rigid_body = 0

        self._actor_rigid_body_mapping = [{-1: [0, 0]} for _ in range(num_envs)]
        contact_id = {body.name: [] for body in self._scene.bodies}

        if len(self._scene.cameras) > 0 and not self._cfg.ISAAC_GYM.ENABLE_CAMERA_SENSORS:
            raise ValueError(
                "For Isaac Gym, ENABLE_CAMERA_SENSORS must be True if any cameras is used"
            )

        self._camera_handles = [{} for _ in range(num_envs)]
        self._camera_image_buffer = [{} for _ in range(num_envs)]

        for idx in range(num_envs):
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)

            counter_env_actor = 0
            counter_env_rigid_body = 0

            for body in self._scene.bodies:
                if body.env_ids_load is not None and idx not in body.env_ids_load:
                    self._actor_indices[idx].append(-1)
                    if not self._actor_indices_need_filter:
                        self._actor_indices_need_filter = True
                    contact_id[body.name].append(-2)
                    continue

                actor_handle = self._gym.create_actor(
                    env_ptr, self._assets[body.name], gymapi.Transform(), name=body.name, group=idx
                )
                actor_index = self._gym.get_actor_index(env_ptr, actor_handle, gymapi.DOMAIN_SIM)
                self._actor_handles[idx][body.name] = actor_handle
                self._actor_indices[idx].append(actor_index)

                self._actor_root_indices[body.name].append(counter_actor)
                self._dof_indices[body.name].append(counter_dof)
                self._rigid_body_indices[body.name].append(counter_rigid_body)
                counter_actor += 1
                counter_dof += self._asset_num_dofs[body.name]
                counter_rigid_body += self._asset_num_rigid_bodies[body.name]

                for i in range(self._asset_num_rigid_bodies[body.name]):
                    self._actor_rigid_body_mapping[idx][counter_env_rigid_body + i] = [
                        counter_env_actor,
                        i,
                    ]
                contact_id[body.name].append(counter_env_actor)
                counter_env_actor += 1
                counter_env_rigid_body += self._asset_num_rigid_bodies[body.name]

            for camera in self._scene.cameras:
                camera_props = self._make_camera_props(camera, idx)
                camera_handle = self._gym.create_camera_sensor(env_ptr, camera_props)

                color = self._gym.get_camera_image_gpu_tensor(
                    self._sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
                )
                depth = self._gym.get_camera_image_gpu_tensor(
                    self._sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH
                )
                segmentation = self._gym.get_camera_image_gpu_tensor(
                    self._sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION
                )

                self._camera_handles[idx][camera.name] = camera_handle
                self._camera_image_buffer[idx][camera.name] = {}
                self._camera_image_buffer[idx][camera.name]["color"] = gymtorch.wrap_tensor(color)
                self._camera_image_buffer[idx][camera.name]["depth"] = gymtorch.wrap_tensor(depth)
                self._camera_image_buffer[idx][camera.name]["segmentation"] = gymtorch.wrap_tensor(
                    segmentation
                )

            self._envs.append(env_ptr)

        self._actor_indices = torch.tensor(
            self._actor_indices, dtype=torch.int32, device=self.device
        )
        for body in self._scene.bodies:
            self._actor_root_indices[body.name] = torch.tensor(
                self._actor_root_indices[body.name], dtype=torch.int64, device=self.device
            )
            self._dof_indices[body.name] = torch.tensor(
                self._dof_indices[body.name], dtype=torch.int64, device=self.device
            )
            self._rigid_body_indices[body.name] = torch.tensor(
                self._rigid_body_indices[body.name], dtype=torch.int64, device=self.device
            )
            body.contact_id = contact_id[body.name]

    def _make_camera_props(self, camera, idx):
        """ """
        camera_props = gymapi.CameraProperties()

        if camera.width is not None:
            camera_props.width = camera.get_attr_array("width", idx)
        if camera.height is not None:
            camera_props.height = camera.get_attr_array("height", idx)
        if camera.vertical_fov is not None:
            if camera.width is None or camera.height is None:
                raise ValueError(
                    "For Isaac Gym, cannot set 'vertical_fov' without setting 'width' and "
                    f"'height': '{camera.name}'"
                )
            camera_props.horizontal_fov = (
                camera.get_attr_array("vertical_fov", idx)
                * camera.get_attr_array("width", idx)
                / camera.get_attr_array("height", idx)
            )
        if camera.near is not None:
            camera_props.near_plane = camera.get_attr_array("near", idx)
        if camera.far is not None:
            camera_props.far_plane = camera.get_attr_array("far", idx)
        camera_props.enable_tensors = True

        return camera_props

    def _cache_body_and_set_props(self):
        """ """
        for body in self._scene.bodies:
            x = type(body)()
            x.name = body.name
            self._scene_cache.add_body(x)

            if self._asset_num_dofs[body.name] == 0:
                for attr in self._ATTR_DOF_PROPS:
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )

            for idx in range(self._num_envs):
                if body.env_ids_load is not None and idx not in body.env_ids_load:
                    continue

                if body.scale is not None:
                    self._set_scale(body, idx)

                if any(getattr(body, x) is not None for x in self._ATTR_RIGID_SHAPE_PROPS):
                    self._set_rigid_shape_props(body, idx)

                if body.link_color is not None:
                    self._set_link_color(body, idx)

                if body.link_segmentation_id is not None:
                    self._set_link_segmentation_id(body, idx)

                if self._asset_num_dofs[body.name] > 0 and any(
                    getattr(body, x) is not None for x in self._ATTR_DOF_PROPS
                ):
                    self._set_dof_props(body, idx)

            if body.scale is None:
                scale = []
                for idx in range(self._num_envs):
                    if body.env_ids_load is not None and idx not in body.env_ids_load:
                        scale_idx = 1.0
                    else:
                        scale_idx = self._gym.get_actor_scale(
                            self._envs[idx], self._actor_handles[idx][body.name]
                        )
                    scale.append(scale_idx)
                body.scale = scale

            if any(getattr(body, x) is None for x in self._ATTR_RIGID_SHAPE_PROPS):
                rigid_shape_props = self._gym.get_asset_rigid_shape_properties(
                    self._assets[body.name]
                )
                if body.link_collision_filter is None:
                    body.link_collision_filter = [
                        [prop.filter for prop in rigid_shape_props]
                    ] * self._num_envs
                if body.link_lateral_friction is None:
                    body.link_lateral_friction = [
                        [prop.friction for prop in rigid_shape_props]
                    ] * self._num_envs
                if body.link_spinning_friction is None:
                    body.link_spinning_friction = [
                        [prop.torsion_friction for prop in rigid_shape_props]
                    ] * self._num_envs
                if body.link_rolling_friction is None:
                    body.link_rolling_friction = [
                        [prop.rolling_friction for prop in rigid_shape_props]
                    ] * self._num_envs
                if body.link_restitution is None:
                    body.link_restitution = [
                        [prop.restitution for prop in rigid_shape_props]
                    ] * self._num_envs

            if body.link_color is None:
                # Avoid error from `get_rigid_body_color()` when `graphics_device` is set to -1.
                if self._cfg.ISAAC_GYM.GRAPHICS_DEVICE_ID == -1:
                    body.link_color = [
                        [[1.0, 1.0, 1.0]] * self._asset_num_rigid_bodies[body.name]
                    ] * self._num_envs
                else:
                    link_color = []
                    for idx in range(self._num_envs):
                        if body.env_ids_load is not None and idx not in body.env_ids_load:
                            link_color_idx = [[1.0, 1.0, 1.0]] * self._asset_num_rigid_bodies[
                                body.name
                            ]
                        else:
                            link_color_idx = []
                            for i in range(self._asset_num_rigid_bodies[body.name]):
                                rigid_body_color = self._gym.get_rigid_body_color(
                                    self._envs[idx],
                                    self._actor_handles[idx][body.name],
                                    i,
                                    gymapi.MESH_VISUAL,
                                )
                                link_color_idx.append(
                                    [rigid_body_color.x, rigid_body_color.y, rigid_body_color.z]
                                )
                        link_color.append(link_color_idx)
                    body.link_color = link_color

            if self._asset_num_dofs[body.name] > 0 and any(
                getattr(body, x) is None for x in self._ATTR_DOF_PROPS
            ):
                dof_props = self._gym.get_asset_dof_properties(self._assets[body.name])
                if body.dof_has_limits is None:
                    body.dof_has_limits = np.tile(dof_props["hasLimits"], (self._num_envs, 1))
                if body.dof_lower_limit is None:
                    body.dof_lower_limit = np.tile(dof_props["lower"], (self._num_envs, 1))
                if body.dof_upper_limit is None:
                    body.dof_upper_limit = np.tile(dof_props["upper"], (self._num_envs, 1))
                if body.dof_control_mode is None:
                    body.dof_control_mode = [
                        k
                        for x in dof_props["driveMode"]
                        for k, v in self._DOF_CONTROL_MODE_MAP.items()
                        if x == v
                    ]
                if body.dof_max_velocity is None:
                    body.dof_max_velocity = np.tile(dof_props["velocity"], (self._num_envs, 1))
                if body.dof_max_force is None:
                    body.dof_max_force = np.tile(dof_props["effort"], (self._num_envs, 1))
                if body.dof_position_gain is None:
                    body.dof_position_gain = np.tile(dof_props["stiffness"], (self._num_envs, 1))
                if body.dof_velocity_gain is None:
                    body.dof_velocity_gain = np.tile(dof_props["damping"], (self._num_envs, 1))
                if body.dof_armature is None:
                    body.dof_armature = np.tile(dof_props["armature"], (self._num_envs, 1))

            body.lock_attr_array()

    def _set_scale(self, body, idx):
        """ """
        self._gym.set_actor_scale(
            self._envs[idx], self._actor_handles[idx][body.name], body.get_attr_array("scale", idx)
        )

    def _set_rigid_shape_props(self, body, idx):
        """ """
        for attr in self._ATTR_RIGID_SHAPE_PROPS:
            if (
                not body.attr_array_locked[attr]
                and getattr(body, attr) is not None
                and len(body.get_attr_array(attr, idx)) != self._asset_num_rigid_shapes[body.name]
            ):
                raise ValueError(
                    f"Size of '{attr}' in the link dimension "
                    f"({len(body.get_attr_array(attr, idx))}) should match the number of rigid "
                    f"shapes ({self._asset_num_rigid_shapes[body.name]}): '{body.name}'"
                )
        rigid_shape_props = self._gym.get_actor_rigid_shape_properties(
            self._envs[idx], self._actor_handles[idx][body.name]
        )
        if (
            not body.attr_array_locked["link_collision_filter"]
            and body.link_collision_filter is not None
            or body.attr_array_dirty_flag["link_collision_filter"]
        ):
            link_collision_filter = body.get_attr_array("link_collision_filter", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.filter = link_collision_filter[i]
        if (
            not body.attr_array_locked["link_lateral_friction"]
            and body.link_lateral_friction is not None
            or body.attr_array_dirty_flag["link_lateral_friction"]
        ):
            link_lateral_friction = body.get_attr_array("link_lateral_friction", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.friction = link_lateral_friction[i]
        if (
            not body.attr_array_locked["link_spinning_friction"]
            and body.link_spinning_friction is not None
            or body.attr_array_dirty_flag["link_spinning_friction"]
        ):
            link_spinning_friction = body.get_attr_array("link_spinning_friction", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.torsion_friction = link_spinning_friction[i]
        if (
            not body.attr_array_locked["link_rolling_friction"]
            and body.link_rolling_friction is not None
            or body.attr_array_dirty_flag["link_rolling_friction"]
        ):
            link_rolling_friction = body.get_attr_array("link_rolling_friction", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.rolling_friction = link_rolling_friction[i]
        if (
            not body.attr_array_locked["link_restitution"]
            and body.link_restitution is not None
            or body.attr_array_dirty_flag["link_restitution"]
        ):
            link_restitution = body.get_attr_array("link_restitution", idx)
            for i, prop in enumerate(rigid_shape_props):
                prop.restitution = link_restitution[i]
        self._gym.set_actor_rigid_shape_properties(
            self._envs[idx], self._actor_handles[idx][body.name], rigid_shape_props
        )

    def _set_link_color(self, body, idx):
        """ """
        link_color = body.get_attr_array("link_color", idx)
        if (
            not body.attr_array_locked["link_color"]
            and len(link_color) != self._asset_num_rigid_bodies[body.name]
        ):
            raise ValueError(
                f"Size of 'link_color' in the link dimension ({len(link_color)}) should match the "
                f"number of links ({self._asset_num_rigid_bodies[body.name]}): '{body.name}'"
            )
        for i in range(self._asset_num_rigid_bodies[body.name]):
            self._gym.set_rigid_body_color(
                self._envs[idx],
                self._actor_handles[idx][body.name],
                i,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*link_color[i]),
            )

    def _set_link_segmentation_id(self, body, idx):
        """ """
        link_segmentation_id = body.get_attr_array("link_segmentation_id", idx)
        if (
            not body.attr_array_locked["link_segmentation_id"]
            and len(link_segmentation_id) != self._asset_num_rigid_bodies[body.name]
        ):
            raise ValueError(
                "Size of 'link_segmentation_id' in the link dimension "
                f"({len(link_segmentation_id)}) should match the number of links "
                f"({self._asset_num_rigid_bodies[body.name]}): '{body.name}'"
            )
        for i in range(self._asset_num_rigid_bodies[body.name]):
            self._gym.set_rigid_body_segmentation_id(
                self._envs[idx], self._actor_handles[idx][body.name], i, link_segmentation_id[i]
            )

    def _set_dof_props(self, body, idx, set_drive_mode=True):
        """ """
        dof_props = self._gym.get_actor_dof_properties(
            self._envs[idx], self._actor_handles[idx][body.name]
        )
        if (
            not body.attr_array_locked["dof_has_limits"]
            and body.dof_has_limits is not None
            or body.attr_array_dirty_flag["dof_has_limits"]
        ):
            dof_props["hasLimits"] = body.get_attr_array("dof_has_limits", idx)
        if (
            not body.attr_array_locked["dof_lower_limit"]
            and body.dof_lower_limit is not None
            or body.attr_array_dirty_flag["dof_lower_limit"]
        ):
            dof_props["lower"] = body.get_attr_array("dof_lower_limit", idx)
        if (
            not body.attr_array_locked["dof_upper_limit"]
            and body.dof_upper_limit is not None
            or body.attr_array_dirty_flag["dof_upper_limit"]
        ):
            dof_props["upper"] = body.get_attr_array("dof_upper_limit", idx)
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
        if (
            not body.attr_array_locked["dof_max_velocity"]
            and body.dof_max_velocity is not None
            or body.attr_array_dirty_flag["dof_max_velocity"]
        ):
            dof_props["velocity"] = body.get_attr_array("dof_max_velocity", idx)
        if (
            not body.attr_array_locked["dof_max_force"]
            and body.dof_max_force is not None
            or body.attr_array_dirty_flag["dof_max_force"]
        ):
            dof_props["effort"] = body.get_attr_array("dof_max_force", idx)
        if (
            not body.attr_array_locked["dof_position_gain"]
            and body.dof_position_gain is not None
            or body.attr_array_dirty_flag["dof_position_gain"]
        ):
            dof_props["stiffness"] = body.get_attr_array("dof_position_gain", idx)
        if (
            not body.attr_array_locked["dof_velocity_gain"]
            and body.dof_velocity_gain is not None
            or body.attr_array_dirty_flag["dof_velocity_gain"]
        ):
            dof_props["damping"] = body.get_attr_array("dof_velocity_gain", idx)
        if (
            not body.attr_array_locked["dof_armature"]
            and body.dof_armature is not None
            or body.attr_array_dirty_flag["dof_armature"]
        ):
            dof_props["armature"] = body.get_attr_array("dof_armature", idx)
        self._gym.set_actor_dof_properties(
            self._envs[idx], self._actor_handles[idx][body.name], dof_props
        )

    def _set_body_device(self):
        """ """
        for body in self._scene.bodies:
            body.set_device(self.device)

    def _set_body_callback(self):
        """ """
        for body in self._scene.bodies:
            body.set_callback_collect_dof_state(self._collect_dof_state)
            body.set_callback_collect_link_state(self._collect_link_state)

    def _collect_dof_state(self, body):
        """ """
        if not self._dof_state_refreshed:
            self._gym.refresh_dof_state_tensor(self._sim)
            self._dof_state_refreshed = True

        if self._asset_num_dofs[body.name] > 0:
            if body.env_ids_load is None:
                body.dof_state = torch.as_strided(
                    self._dof_state,
                    (
                        len(self._dof_state) - self._asset_num_dofs[body.name] + 1,
                        self._asset_num_dofs[body.name],
                        2,
                    ),
                    (2, 2, 1),
                )[self._dof_indices[body.name]]
            else:
                body.dof_state = self._dof_state.new_zeros(
                    (self._num_envs, self._asset_num_dofs[body.name], 2)
                )
                body.dof_state[body.env_ids_load] = torch.as_strided(
                    self._dof_state,
                    (
                        len(self._dof_state) - self._asset_num_dofs[body.name] + 1,
                        self._asset_num_dofs[body.name],
                        2,
                    ),
                    (2, 2, 1),
                )[self._dof_indices[body.name]]

    def _collect_link_state(self, body):
        """ """
        if not self._link_state_refreshed:
            self._gym.refresh_rigid_body_state_tensor(self._sim)
            self._link_state_refreshed = True

        if body.env_ids_load is None:
            body.link_state = torch.as_strided(
                self._rigid_body_state,
                (
                    len(self._rigid_body_state) - self._asset_num_rigid_bodies[body.name] + 1,
                    self._asset_num_rigid_bodies[body.name],
                    13,
                ),
                (13, 13, 1),
            )[self._rigid_body_indices[body.name]]
        else:
            body.link_state = self._rigid_body_state.new_zeros(
                (self._num_envs, self._asset_num_rigid_bodies[body.name], 13)
            )
            body.link_state[body.env_ids_load] = torch.as_strided(
                self._rigid_body_state,
                (
                    len(self._rigid_body_state) - self._asset_num_rigid_bodies[body.name] + 1,
                    self._asset_num_rigid_bodies[body.name],
                    13,
                ),
                (13, 13, 1),
            )[self._rigid_body_indices[body.name]]

    def _set_camera_props(self):
        """ """
        for camera in self._scene.cameras:
            for idx in range(self._num_envs):
                self._set_camera_pose(camera, idx)

            if any(getattr(camera, x) is None for x in self._ATTR_PROJECTION_MATRIX):
                camera_props = gymapi.CameraProperties()
                if camera.width is None:
                    camera.width = [camera_props.width] * self._num_envs
                if camera.height is None:
                    camera.height = [camera_props.height] * self._num_envs
                if camera.vertical_fov is None:
                    camera.vertical_fov = (
                        [camera_props.horizontal_fov]
                        * self._num_envs
                        * camera.height
                        / camera.width
                    )
                if camera.near is None:
                    camera.near = [camera_props.near_plane] * self._num_envs
                if camera.far is None:
                    camera.far = [camera_props.far_plane] * self._num_envs

            camera.lock_attr_array()

    def _set_camera_pose(self, camera, idx):
        """ """
        for attr in ("up_vector",):
            if getattr(camera, attr) is not None:
                raise ValueError(f"'{attr}' is not supported in Isaac Gym: '{camera.name}'")
        if camera.target is not None:
            self._gym.set_camera_location(
                self._camera_handles[idx][camera.name],
                self._envs[idx],
                gymapi.Vec3(*camera.get_attr_array("position", idx)),
                gymapi.Vec3(*camera.get_attr_array("target", idx)),
            )
        elif camera.orientation is not None:
            # A rotation offset is required for gymapi.UP_AXIS_Z.
            self._gym.set_camera_transform(
                self._camera_handles[idx][camera.name],
                self._envs[idx],
                gymapi.Transform(
                    p=gymapi.Vec3(*camera.get_attr_array("position", idx)),
                    r=gymapi.Quat(*camera.get_attr_array("orientation", idx))
                    * gymapi.Quat(x=-0.5, y=+0.5, z=+0.5, w=+0.5),
                ),
            )
        else:
            raise ValueError(
                "For Isaac Gym, either 'target' or 'orientation' is required to be set: "
                f"{camera.name}"
            )

    def _set_camera_device(self):
        """ """
        for camera in self._scene.cameras:
            camera.set_device(self._graphics_device)

    def _set_camera_callback(self):
        """ """
        for camera in self._scene.cameras:
            camera.set_callback_render_color(self._render_color)
            camera.set_callback_render_depth(self._render_depth)
            camera.set_callback_render_segmentation(self._render_segmentation)

    def _render_color(self, camera):
        """ """
        for body in self._scene.bodies:
            if body.attr_array_dirty_flag["link_color"]:
                env_ids_masked = np.nonzero(body.attr_array_dirty_mask["link_color"])[0]
                for idx in env_ids_masked:
                    self._set_link_color(body, idx)
                body.attr_array_dirty_flag["link_color"] = False
                body.attr_array_dirty_mask["link_color"][:] = False
                if self._all_camera_rendered:
                    self._all_camera_rendered = False

        self._check_and_step_graphics()
        self._check_and_update_camera_props(camera)
        self._check_and_render_all_camera()

        color = self._camera_image_buffer[0][camera.name]["color"].new_zeros(
            (self._num_envs, np.max(camera.height), np.max(camera.width), 4)
        )

        self._gym.start_access_image_tensors(self._sim)
        for idx in range(self._num_envs):
            color[
                idx, : camera.get_attr_array("height", idx), : camera.get_attr_array("width", idx)
            ] = self._camera_image_buffer[idx][camera.name]["color"]
        self._gym.end_access_image_tensors(self._sim)

        camera.color = color

    def _render_depth(self, camera):
        """ """
        self._check_and_step_graphics()
        self._check_and_update_camera_props(camera)
        self._check_and_render_all_camera()

        depth = self._camera_image_buffer[0][camera.name]["depth"].new_zeros(
            (self._num_envs, np.max(camera.height), np.max(camera.width))
        )

        self._gym.start_access_image_tensors(self._sim)
        for idx in range(self._num_envs):
            depth[
                idx, : camera.get_attr_array("height", idx), : camera.get_attr_array("width", idx)
            ] = self._camera_image_buffer[idx][camera.name]["depth"]
        self._gym.end_access_image_tensors(self._sim)

        depth *= -1.0
        depth[depth == +np.inf] = 0.0

        camera.depth = depth

    def _render_segmentation(self, camera):
        """ """
        for body in self._scene.bodies:
            if body.attr_array_dirty_flag["link_segmentation_id"]:
                env_ids_masked = np.nonzero(body.attr_array_dirty_mask["link_segmentation_id"])[0]
                for idx in env_ids_masked:
                    self._set_link_segmentation_id(body, idx)
                body.attr_array_dirty_flag["link_segmentation_id"] = False
                body.attr_array_dirty_mask["link_segmentation_id"][:] = False
                if self._all_camera_rendered:
                    self._all_camera_rendered = False

        self._check_and_step_graphics()
        self._check_and_update_camera_props(camera)
        self._check_and_render_all_camera()

        segmentation = self._camera_image_buffer[0][camera.name]["segmentation"].new_zeros(
            (self._num_envs, np.max(camera.height), np.max(camera.width))
        )

        self._gym.start_access_image_tensors(self._sim)
        for idx in range(self._num_envs):
            segmentation[
                idx, : camera.get_attr_array("height", idx), : camera.get_attr_array("width", idx)
            ] = self._camera_image_buffer[idx][camera.name]["segmentation"]
        self._gym.end_access_image_tensors(self._sim)

        camera.segmentation = segmentation

    def _check_and_step_graphics(self):
        """ """
        if not self._graphics_stepped:
            self._gym.step_graphics(self._sim)
            self._graphics_stepped = True

    def _check_and_update_camera_props(self, camera):
        """ """
        if any(camera.attr_array_dirty_flag[x] for x in self._ATTR_VIEW_MATRIX):
            mask = np.zeros(self._num_envs, dtype=bool)
            for attr in self._ATTR_VIEW_MATRIX:
                if camera.attr_array_dirty_flag[attr]:
                    mask |= camera.attr_array_dirty_mask[attr]
            env_ids_masked = np.nonzero(mask)[0]
            for idx in env_ids_masked:
                self._set_camera_pose(camera, idx)
            for attr in self._ATTR_VIEW_MATRIX:
                if camera.attr_array_dirty_flag[attr]:
                    camera.attr_array_dirty_flag[attr] = False
                    camera.attr_array_dirty_mask[attr][:] = False
            if self._all_camera_rendered:
                self._all_camera_rendered = False

    def _check_and_render_all_camera(self):
        """ """
        if not self._all_camera_rendered:
            self._gym.render_all_camera_sensors(self._sim)
            self._all_camera_rendered = True

    def _set_scene_callback(self):
        """ """
        self._scene.set_callback_add_camera(self._add_camera)
        self._scene.set_callback_remove_camera(self._remove_camera)

    def _add_camera(self, camera):
        """ """
        raise ValueError(
            "For Isaac Gym, the set of cameras cannot be altered after the first reset"
        )

    def _remove_camera(self, camera):
        """ """
        raise ValueError(
            "For Isaac Gym, the set of cameras cannot be altered after the first reset"
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

            if self._cfg.VIEWER.INIT_CAMERA_POSITION != (
                None,
                None,
                None,
            ) and self._cfg.VIEWER.INIT_CAMERA_TARGET != (None, None, None):
                self._gym.viewer_camera_look_at(
                    self._viewer,
                    None,
                    gymapi.Vec3(*self._cfg.VIEWER.INIT_CAMERA_POSITION),
                    gymapi.Vec3(*self._cfg.VIEWER.INIT_CAMERA_TARGET),
                )

            if self._cfg.ISAAC_GYM.VIEWER.DRAW_AXES:
                if len(self._scene.cameras) > 0:
                    raise ValueError(
                        "For Isaac Gym, set DRAW_AXES to False if opening the viewer with cameras, "
                        "since the drawn axes cannot be hidden in camera's rendered images."
                    )
                axes_geom = gymutil.AxesGeometry(1.0)
                for env_ptr in self._envs:
                    gymutil.draw_lines(
                        axes_geom, self._gym, self._viewer, env_ptr, gymapi.Transform()
                    )

    def _allocate_buffers(self):
        """ """
        if self._dof_state is None:
            self._dof_position_target_buffer = None
            self._dof_velocity_target_buffer = None
            self._dof_actuation_force_buffer = None
        else:
            self._dof_position_target_buffer = torch.zeros(
                len(self._dof_state), dtype=torch.float32, device=self.device
            )
            self._dof_velocity_target_buffer = torch.zeros(
                len(self._dof_state), dtype=torch.float32, device=self.device
            )
            self._dof_actuation_force_buffer = torch.zeros(
                len(self._dof_state), dtype=torch.float32, device=self.device
            )

    def _reset_idx(self, env_ids):
        """ """
        if [body.name for body in self._scene.bodies] != [
            body.name for body in self._scene_cache.bodies
        ]:
            raise ValueError(
                "For Isaac Gym, the list of bodies cannot be altered after the first reset"
            )

        for body in self._scene.bodies:
            self._reset_base_state_buffer(body)

            if self._asset_num_dofs[body.name] == 0:
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
            if self._actor_indices_need_filter:
                actor_indices = actor_indices[actor_indices != -1]
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
                [self._asset_num_dofs[body.name] > 0 for body in self._scene.bodies],
            ].view(-1)
            if self._actor_indices_need_filter:
                actor_indices = actor_indices[actor_indices != -1]
            self._gym.set_dof_state_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )

        self._check_and_update_body_props(env_ids=env_ids)

    def _reset_base_state_buffer(self, body):
        """ """
        if body.initial_base_position is None:
            initial_base_position = self._initial_actor_root_state[
                self._actor_root_indices[body.name], :7
            ]
        else:
            if body.env_ids_load is None or body.initial_base_position.ndim == 1:
                initial_base_position = body.initial_base_position
            else:
                initial_base_position = body.initial_base_position[body.env_ids_load]
        self._actor_root_state[self._actor_root_indices[body.name], :7] = initial_base_position
        if body.initial_base_velocity is None:
            initial_base_velocity = self._initial_actor_root_state[
                self._actor_root_indices[body.name], 7:
            ]
        else:
            if body.env_ids_load is None or body.initial_base_velocity.ndim == 1:
                initial_base_velocity = body.initial_base_velocity
            else:
                initial_base_velocity = body.initial_base_velocity[body.env_ids_load]
        self._actor_root_state[self._actor_root_indices[body.name], 7:] = initial_base_velocity

    def _reset_dof_state_buffer(self, body):
        """ """
        if body.initial_dof_position is None:
            initial_dof_position = torch.as_strided(
                self._initial_dof_state[:, 0],
                (
                    len(self._initial_dof_state) - self._asset_num_dofs[body.name] + 1,
                    self._asset_num_dofs[body.name],
                ),
                (2, 2),
            )[self._dof_indices[body.name]]
        else:
            if body.env_ids_load is None or body.initial_dof_position.ndim == 1:
                initial_dof_position = body.initial_dof_position
            else:
                initial_dof_position = body.initial_dof_position[body.env_ids_load]
        torch.as_strided(
            self._dof_state[:, 0],
            (
                len(self._dof_state) - self._asset_num_dofs[body.name] + 1,
                self._asset_num_dofs[body.name],
            ),
            (2, 2),
        )[self._dof_indices[body.name]] = initial_dof_position
        if body.initial_dof_velocity is None:
            initial_dof_velocity = torch.as_strided(
                self._initial_dof_state[:, 1],
                (
                    len(self._initial_dof_state) - self._asset_num_dofs[body.name] + 1,
                    self._asset_num_dofs[body.name],
                ),
                (2, 2),
            )[self._dof_indices[body.name]]
        else:
            if body.env_ids_load is None or body.initial_dof_velocity.ndim == 1:
                initial_dof_velocity = body.initial_dof_velocity
            else:
                initial_dof_velocity = body.initial_dof_velocity[body.env_ids_load]
        torch.as_strided(
            self._dof_state[:, 1],
            (
                len(self._dof_state) - self._asset_num_dofs[body.name] + 1,
                self._asset_num_dofs[body.name],
            ),
            (2, 2),
        )[self._dof_indices[body.name]] = initial_dof_velocity

    def _check_and_update_body_props(self, env_ids=None):
        """ """
        for body in self._scene.bodies:
            for attr in ("scale", "link_color"):
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
                        if attr == "scale":
                            self._set_scale(body, idx)
                        if attr == "link_color":
                            self._set_link_color(body, idx)
                    body.attr_array_dirty_flag[attr] = False
                    body.attr_array_dirty_mask[attr][:] = False

            if any(body.attr_array_dirty_flag[x] for x in self._ATTR_RIGID_SHAPE_PROPS):
                mask = np.zeros(self._num_envs, dtype=bool)
                for attr in self._ATTR_RIGID_SHAPE_PROPS:
                    if body.attr_array_dirty_flag[attr]:
                        if env_ids is not None and not np.all(
                            np.isin(np.nonzero(body.attr_array_dirty_mask[attr])[0], env_ids.cpu())
                        ):
                            raise ValueError(
                                f"For Isaac Gym, to change '{attr}' for some env also requires the "
                                f"env indices to be in `env_ids`: '{body.name}'"
                            )
                        mask |= body.attr_array_dirty_mask[attr]
                env_ids_masked = np.nonzero(mask)[0]
                for idx in env_ids_masked:
                    self._set_rigid_shape_props(body, idx)
                for attr in self._ATTR_RIGID_SHAPE_PROPS:
                    if body.attr_array_dirty_flag[attr]:
                        body.attr_array_dirty_flag[attr] = False
                        body.attr_array_dirty_mask[attr][:] = False

            for attr in ("link_linear_damping", "link_angular_damping"):
                if body.attr_array_dirty_flag[attr]:
                    raise ValueError(
                        f"For Isaac Gym, '{attr}' cannot be changed after the first reset: "
                        f"'{body.name}'"
                    )

            if self._asset_num_dofs[body.name] > 0:
                if body.attr_array_dirty_flag["dof_control_mode"]:
                    raise ValueError(
                        "For Isaac Gym, 'dof_control_mode' cannot be changed after the first "
                        f"reset: '{body.name}'"
                    )
                if any(
                    body.attr_array_dirty_flag[x]
                    for x in self._ATTR_DOF_PROPS
                    if x != "dof_control_mode"
                ):
                    mask = np.zeros(self._num_envs, dtype=bool)
                    for attr in self._ATTR_DOF_PROPS:
                        if body.attr_array_dirty_flag[attr]:
                            if env_ids is not None and not np.all(
                                np.isin(
                                    np.nonzero(body.attr_array_dirty_mask[attr])[0], env_ids.cpu()
                                )
                            ):
                                raise ValueError(
                                    f"For Isaac Gym, to change '{attr}' for certain env also "
                                    f"requires the env index to be in `env_ids`: '{body.name}'"
                                )
                            mask |= body.attr_array_dirty_mask[attr]
                    env_ids_masked = np.nonzero(mask)[0]
                    for idx in env_ids_masked:
                        self._set_dof_props(body, idx, set_drive_mode=False)
                    for attr in self._ATTR_DOF_PROPS:
                        if body.attr_array_dirty_flag[attr]:
                            body.attr_array_dirty_flag[attr] = False
                            body.attr_array_dirty_mask[attr][:] = False
            else:
                for attr in self._ATTR_DOF_PROPS:
                    if body.attr_array_dirty_flag[attr]:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )

    def _clear_state(self):
        """ """
        for body in self._scene.bodies:
            body.dof_state = None
            body.link_state = None

        self._dof_state_refreshed = False
        self._link_state_refreshed = False

    def _clear_image(self):
        """ """
        for camera in self._scene.cameras:
            camera.color = None
            camera.depth = None
            camera.segmentation = None

        self._all_camera_rendered = False

    def step(self):
        """ """
        if [body.name for body in self._scene.bodies] != [
            body.name for body in self._scene_cache.bodies
        ]:
            raise ValueError(
                "For Isaac Gym, the list of bodies cannot be altered after the first reset"
            )

        self._check_and_update_body_props()

        reset_base_state = False
        reset_dof_state = False
        actor_indices_base = []
        actor_indices_dof = []

        for b, body in enumerate(self._scene.bodies):
            if body.env_ids_reset_base_state is not None:
                if body.env_ids_load is not None and not torch.all(
                    torch.isin(body.env_ids_reset_base_state, body.env_ids_load)
                ):
                    raise ValueError(
                        "'env_ids_reset_base_state' must be a subset of 'env_ids_load' for "
                        f"non-None 'env_ids_load': '{body.name}'"
                    )
                self._reset_base_state_buffer(body)
                if not reset_base_state:
                    reset_base_state = True
                actor_indices_base.append(self._actor_indices[body.env_ids_reset_base_state, b])
                body.env_ids_reset_base_state = None

            if self._asset_num_dofs[body.name] == 0:
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
                if body.env_ids_load is not None and not torch.all(
                    torch.isin(body.env_ids_reset_dof_state, body.env_ids_load)
                ):
                    raise ValueError(
                        "'env_ids_reset_dof_state' must be a subset of 'env_ids_load' for non-None "
                        f"'env_ids_load': '{body.name}'"
                    )
                self._reset_dof_state_buffer(body)
                if not reset_dof_state:
                    reset_dof_state = True
                actor_indices_dof.append(self._actor_indices[body.env_ids_reset_dof_state, b])
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
                    if body.env_ids_load is None or body.dof_target_position.ndim == 1:
                        dof_target_position = body.dof_target_position
                    else:
                        dof_target_position = body.dof_target_position[body.env_ids_load]
                    torch.as_strided(
                        self._dof_position_target_buffer,
                        (
                            len(self._dof_position_target_buffer)
                            - self._asset_num_dofs[body.name]
                            + 1,
                            self._asset_num_dofs[body.name],
                        ),
                        (1, 1),
                    )[self._dof_indices[body.name]] = dof_target_position
                if body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL:
                    if body.env_ids_load is None or body.dof_target_velocity.ndim == 1:
                        dof_target_velocity = body.dof_target_velocity
                    else:
                        dof_target_velocity = body.dof_target_velocity[body.env_ids_load]
                    torch.as_strided(
                        self._dof_velocity_target_buffer,
                        (
                            len(self._dof_velocity_target_buffer)
                            - self._asset_num_dofs[body.name]
                            + 1,
                            self._asset_num_dofs[body.name],
                        ),
                        (1, 1),
                    )[self._dof_indices[body.name]] = dof_target_velocity
                if body.dof_control_mode == DoFControlMode.TORQUE_CONTROL:
                    if body.env_ids_load is None or body.dof_force.ndim == 1:
                        dof_force = body.dof_force
                    else:
                        dof_force = body.dof_force[body.env_ids_load]
                    torch.as_strided(
                        self._dof_actuation_force_buffer,
                        (
                            len(self._dof_actuation_force_buffer)
                            - self._asset_num_dofs[body.name]
                            + 1,
                            self._asset_num_dofs[body.name],
                        ),
                        (1, 1),
                    )[self._dof_indices[body.name]] = dof_force
            if body.dof_control_mode.ndim == 1:
                if DoFControlMode.POSITION_CONTROL in body.dof_control_mode:
                    if body.env_ids_load is None or body.dof_target_position.ndim == 1:
                        dof_target_position = body.dof_target_position[
                            ..., body.dof_control_mode == DoFControlMode.POSITION_CONTROL
                        ]
                    else:
                        dof_target_position = body.dof_target_position[
                            body.env_ids_load[:, None],
                            body.dof_control_mode == DoFControlMode.POSITION_CONTROL,
                        ]
                    torch.as_strided(
                        self._dof_position_target_buffer,
                        (
                            len(self._dof_position_target_buffer)
                            - self._asset_num_dofs[body.name]
                            + 1,
                            self._asset_num_dofs[body.name],
                        ),
                        (1, 1),
                    )[
                        self._dof_indices[body.name][:, None],
                        body.dof_control_mode == DoFControlMode.POSITION_CONTROL,
                    ] = dof_target_position
                if DoFControlMode.VELOCITY_CONTROL in body.dof_control_mode:
                    if body.env_ids_load is None or body.dof_target_position.ndim == 1:
                        dof_target_velocity = body.dof_target_velocity[
                            ..., body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL
                        ]
                    else:
                        dof_target_velocity = body.dof_target_velocity[
                            body.env_ids_load[:, None],
                            body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL,
                        ]
                    torch.as_strided(
                        self._dof_velocity_target_buffer,
                        (
                            len(self._dof_velocity_target_buffer)
                            - self._asset_num_dofs[body.name]
                            + 1,
                            self._asset_num_dofs[body.name],
                        ),
                        (1, 1),
                    )[
                        self._dof_indices[body.name][:, None],
                        body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL,
                    ] = dof_target_velocity
                if DoFControlMode.TORQUE_CONTROL in body.dof_control_mode:
                    if body.env_ids_load is None or body.dof_force.ndim == 1:
                        dof_force = body.dof_force[
                            ..., body.dof_control_mode == DoFControlMode.TORQUE_CONTROL
                        ]
                    else:
                        dof_force = body.dof_force[
                            body.env_ids_load[:, None],
                            body.dof_control_mode == DoFControlMode.TORQUE_CONTROL,
                        ]
                    torch.as_strided(
                        self._dof_actuation_force_buffer,
                        (
                            len(self._dof_actuation_force_buffer)
                            - self._asset_num_dofs[body.name]
                            + 1,
                            self._asset_num_dofs[body.name],
                        ),
                        (1, 1),
                    )[
                        self._dof_indices[body.name][:, None],
                        body.dof_control_mode == DoFControlMode.TORQUE_CONTROL,
                    ] = dof_force

        if reset_base_state:
            actor_indices = torch.cat(actor_indices_base)
            self._gym.set_actor_root_state_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._actor_root_state),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )

        if reset_dof_state:
            actor_indices = torch.cat(actor_indices_dof)
            self._gym.set_dof_state_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )

        if self._dof_state is not None:
            self._gym.set_dof_position_target_tensor(
                self._sim, gymtorch.unwrap_tensor(self._dof_position_target_buffer)
            )
            self._gym.set_dof_velocity_target_tensor(
                self._sim, gymtorch.unwrap_tensor(self._dof_velocity_target_buffer)
            )
            self._gym.set_dof_actuation_force_tensor(
                self._sim, gymtorch.unwrap_tensor(self._dof_actuation_force_buffer)
            )

        self._gym.simulate(self._sim)
        if self.device == "cpu" or self._cfg.RENDER or self._cfg.ISAAC_GYM.ENABLE_CAMERA_SENSORS:
            self._gym.fetch_results(self._sim, True)

        self._graphics_stepped = False
        self._clear_state()
        self._clear_image()
        self._contact = None

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

                    self._check_and_step_graphics()

                    self._gym.draw_viewer(self._viewer, self._sim)

                self._counter_render += 1
            else:
                self._gym.poll_viewer_events(self._viewer)

                if self._counter_render != 0:
                    self._counter_render = 0

    @property
    def contact(self):
        """ """
        if self._contact is None:
            self._contact = self._collect_contact()
        return self._contact

    def _collect_contact(self):
        """ """
        contact = []
        for idx, env in enumerate(self._envs):
            rigid_contacts = self._gym.get_env_rigid_contacts(env)
            if len(rigid_contacts) == 0:
                contact_array = create_contact_array(0)
            else:
                kwargs = {}
                kwargs["body_id_a"], kwargs["link_id_a"] = zip(
                    *[self._actor_rigid_body_mapping[idx][x] for x in rigid_contacts["body0"]]
                )
                kwargs["body_id_b"], kwargs["link_id_b"] = zip(
                    *[self._actor_rigid_body_mapping[idx][x] for x in rigid_contacts["body1"]]
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
            for idx in range(self._num_envs):
                for name in self._camera_handles[idx]:
                    self._gym.destroy_camera_sensor(
                        self._sim, self._envs[idx], self._camera_handles[idx][name]
                    )
            self._gym.destroy_viewer(self._viewer)
            self._gym.destroy_sim(self._sim)
            self._created = False
