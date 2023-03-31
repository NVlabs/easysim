# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data
import pkgutil
import numpy as np
import time

from contextlib import contextmanager

from easysim.simulators.simulator import Simulator
from easysim.constants import DescriptionType, DoFControlMode
from easysim.contact import create_contact_array


class Bullet(Simulator):
    """Bullet simulator."""

    _ATTR_LINK_DYNAMICS = (
        "link_lateral_friction",
        "link_spinning_friction",
        "link_rolling_friction",
        "link_restitution",
        "link_linear_damping",
        "link_angular_damping",
    )
    _ATTR_DOF_DYNAMICS = (
        "dof_lower_limit",
        "dof_upper_limit",
    )
    _ATTR_PROJECTION_MATRIX = ("width", "height", "vertical_fov", "near", "far")
    _ATTR_VIEW_MATRIX = ("position", "target", "up_vector", "orientation")
    _DOF_CONTROL_MODE_MAP = {
        DoFControlMode.POSITION_CONTROL: pybullet.POSITION_CONTROL,
        DoFControlMode.VELOCITY_CONTROL: pybullet.VELOCITY_CONTROL,
        DoFControlMode.TORQUE_CONTROL: pybullet.TORQUE_CONTROL,
    }

    def __init__(self, cfg, scene):
        """ """
        super().__init__(cfg, scene)

        if self._cfg.NUM_ENVS != 1:
            raise ValueError("NUM_ENVS must be 1 for Bullet")
        if self._cfg.SIM_DEVICE != "cpu":
            raise ValueError("SIM_DEVICE must be 'cpu' for Bullet")
        if self._cfg.USE_GPU_PIPELINE:
            raise ValueError("USE_GPU_PIPELINE must be False for Bullet")

        self._device = "cpu"
        self._graphics_device = "cpu"

        self._connected = False
        self._last_frame_time = 0.0

    @property
    def device(self):
        """ """
        return self._device

    def reset(self, env_ids):
        """ """
        if not self._connected:
            if self._cfg.RENDER:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

            self._plugins = {}

            if self._cfg.BULLET.USE_EGL:
                if self._cfg.RENDER:
                    raise ValueError("USE_EGL can only be True when RENDER is set to False")
                egl = pkgutil.get_loader("eglRenderer")
                self._plugins["egl_renderer"] = self._p.loadPlugin(
                    egl.get_filename(), "_eglRendererPlugin"
                )

        with self._disable_cov_rendering():
            self._p.resetSimulation()
            self._p.setGravity(*self._cfg.GRAVITY)
            if self._cfg.USE_DEFAULT_STEP_PARAMS:
                sim_params = self._p.getPhysicsEngineParameters()
                self._cfg.TIME_STEP = sim_params["fixedTimeStep"]
                self._cfg.SUBSTEPS = max(sim_params["numSubSteps"], 1)
            else:
                self._p.setPhysicsEngineParameter(
                    fixedTimeStep=self._cfg.TIME_STEP, numSubSteps=self._cfg.SUBSTEPS
                )
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

            if self._cfg.LOAD_GROUND_PLANE:
                self._body_id_ground_plane = self._load_ground_plane()

            self._scene_cache = type(self._scene)()

            self._body_ids = {}
            self._dof_indices = {}
            self._num_links = {}

            for body in self._scene.bodies:
                self._load_body(body)
                self._cache_body_and_set_control_and_props(body)
                self._set_body_device(body)
                self._set_body_callback(body)

            self._projection_matrix = {}
            self._view_matrix = {}
            self._image_cache = {}

            for camera in self._scene.cameras:
                self._load_camera(camera)
                self._set_camera_device(camera)
                self._set_camera_callback(camera)

            if not self._connected:
                self._set_scene_callback()

            if (
                self._cfg.RENDER
                and self._cfg.VIEWER.INIT_CAMERA_POSITION != (None, None, None)
                and self._cfg.VIEWER.INIT_CAMERA_TARGET != (None, None, None)
            ):
                self._set_viewer_camera_pose(
                    self._cfg.VIEWER.INIT_CAMERA_POSITION, self._cfg.VIEWER.INIT_CAMERA_TARGET
                )

        if not self._connected:
            self._connected = True

        self._clear_state()
        self._clear_image()
        self._contact = None

    @contextmanager
    def _disable_cov_rendering(self):
        """ """
        try:
            if self._cfg.RENDER:
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            yield
        finally:
            if self._cfg.RENDER:
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def _load_ground_plane(self):
        """ """
        kwargs = {}
        kwargs["basePosition"] = (0.0, 0.0, -self._cfg.GROUND_PLANE.DISTANCE)
        return self._p.loadURDF("plane_implicit.urdf", **kwargs)

    def _load_body(self, body):
        """ """
        if body.env_ids_load is not None:
            if np.array_equal(body.env_ids_load.cpu(), []):
                return
            elif not np.array_equal(body.env_ids_load.cpu(), [0]):
                raise ValueError(
                    f"For Bullet, 'env_ids_load' must be either None, [] or [0]: '{body.name}'"
                )

        kwargs = {}
        if (
            body.simulator_config.bullet.use_self_collision is not None
            and body.simulator_config.bullet.use_self_collision
        ):
            kwargs["flags"] = pybullet.URDF_USE_SELF_COLLISION
        if body.description_type is None:
            raise ValueError(f"For Bullet, 'description_type' must not be None: '{body.name}'")
        if body.description_type not in (
            DescriptionType.URDF,
            DescriptionType.SPHERE,
            DescriptionType.BOX,
        ):
            raise ValueError(
                f"For Bullet, 'description_type' only supports URDF, SPHERE, and BOX: '{body.name}'"
            )
        if body.description_type == DescriptionType.URDF:
            if body.use_fixed_base is not None:
                kwargs["useFixedBase"] = body.use_fixed_base
            if body.scale is not None:
                kwargs["globalScaling"] = body.get_attr_array("scale", 0)
                if body.attr_array_dirty_flag["scale"]:
                    body.attr_array_dirty_flag["scale"] = False
            self._body_ids[body.name] = self._p.loadURDF(
                body.description_config.urdf.path, **kwargs
            )
        else:
            for attr in ("scale",):
                if getattr(body, attr) is not None:
                    description_type = [
                        x
                        for x in dir(DescriptionType)
                        if not x.startswith("__")
                        and getattr(DescriptionType, x) == body.description_type
                    ][0]
                    raise ValueError(
                        f"For Bullet, '{attr}' is not supported for geometry type "
                        f"{description_type}: '{body.name}'"
                    )
            kwargs_visual = {}
            kwargs_collision = {}
            if body.description_type == DescriptionType.SPHERE:
                if body.description_config.sphere.radius is not None:
                    kwargs_visual["radius"] = body.description_config.sphere.radius
                    kwargs_collision["radius"] = body.description_config.sphere.radius
                kwargs["baseVisualShapeIndex"] = self._p.createVisualShape(
                    pybullet.GEOM_SPHERE, **kwargs_visual
                )
                kwargs["baseCollisionShapeIndex"] = self._p.createCollisionShape(
                    pybullet.GEOM_SPHERE, **kwargs_collision
                )
            if body.description_type == DescriptionType.BOX:
                if body.description_config.box.half_extent is not None:
                    kwargs_visual["halfExtents"] = body.description_config.box.half_extent
                    kwargs_collision["halfExtents"] = body.description_config.box.half_extent
                kwargs["baseVisualShapeIndex"] = self._p.createVisualShape(
                    pybullet.GEOM_BOX, **kwargs_visual
                )
                kwargs["baseCollisionShapeIndex"] = self._p.createCollisionShape(
                    pybullet.GEOM_BOX, **kwargs_collision
                )
            if body.use_fixed_base is not None and body.use_fixed_base:
                kwargs["baseMass"] = 0.0
            self._body_ids[body.name] = self._p.createMultiBody(**kwargs)

        dof_indices = []
        for j in range(self._p.getNumJoints(self._body_ids[body.name])):
            joint_info = self._p.getJointInfo(self._body_ids[body.name], j)
            if joint_info[2] != pybullet.JOINT_FIXED:
                dof_indices.append(j)
        self._dof_indices[body.name] = np.asanyarray(dof_indices, dtype=np.int64)

        self._num_links[body.name] = self._p.getNumJoints(self._body_ids[body.name]) + 1

        # Reset base state.
        if body.initial_base_position is not None:
            self._reset_base_position(body)
        if body.initial_base_velocity is not None:
            self._reset_base_velocity(body)

        # Reset DoF state.
        if len(self._dof_indices[body.name]) == 0:
            for attr in ("initial_dof_position", "initial_dof_velocity"):
                if getattr(body, attr) is not None:
                    raise ValueError(f"'{attr}' must be None for body with 0 DoF: '{body.name}'")
        if body.initial_dof_position is not None:
            self._reset_dof_state(body)
        elif body.initial_dof_velocity is not None:
            raise ValueError(
                "For Bullet, cannot reset 'initial_dof_velocity' without resetting "
                f"'initial_dof_position': '{body.name}'"
            )

        body.contact_id = [self._body_ids[body.name]]

    def _reset_base_position(self, body):
        """ """
        if body.initial_base_position.ndim == 1:
            self._p.resetBasePositionAndOrientation(
                self._body_ids[body.name],
                body.initial_base_position[:3],
                body.initial_base_position[3:],
            )
        if body.initial_base_position.ndim == 2:
            self._p.resetBasePositionAndOrientation(
                self._body_ids[body.name],
                body.initial_base_position[0, :3],
                body.initial_base_position[0, 3:],
            )

    def _reset_base_velocity(self, body):
        """ """
        kwargs = {}
        if body.initial_base_velocity.ndim == 1:
            kwargs["linearVelocity"] = body.initial_base_velocity[:3]
            kwargs["angularVelocity"] = body.initial_base_velocity[3:]
        if body.initial_base_velocity.ndim == 2:
            kwargs["linearVelocity"] = body.initial_base_velocity[0, :3]
            kwargs["angularVelocity"] = body.initial_base_velocity[0, 3:]
        self._p.resetBaseVelocity(self._body_ids[body.name], **kwargs)

    def _reset_dof_state(self, body):
        """ """
        for i, j in enumerate(self._dof_indices[body.name]):
            kwargs = {}
            if body.initial_dof_velocity is not None:
                if body.initial_dof_velocity.ndim == 1:
                    kwargs["targetVelocity"] = body.initial_dof_velocity[i]
                if body.initial_dof_velocity.ndim == 2:
                    kwargs["targetVelocity"] = body.initial_dof_velocity[0, i]
            if body.initial_dof_position.ndim == 1:
                self._p.resetJointState(
                    self._body_ids[body.name], j, body.initial_dof_position[i], **kwargs
                )
            if body.initial_dof_position.ndim == 2:
                self._p.resetJointState(
                    self._body_ids[body.name], j, body.initial_dof_position[0, i], **kwargs
                )

    def _cache_body_and_set_control_and_props(self, body):
        """ """
        x = type(body)()
        x.name = body.name
        self._scene_cache.add_body(x)

        if body.env_ids_load is not None and len(body.env_ids_load) == 0:
            body.lock_attr_array()
            return

        for attr in ("link_segmentation_id", "dof_has_limits", "dof_armature"):
            if getattr(body, attr) is not None:
                raise ValueError(f"'{attr}' is not supported in Bullet: '{body.name}'")

        if len(self._dof_indices[body.name]) == 0:
            for attr in ("dof_lower_limit", "dof_upper_limit", "dof_control_mode"):
                if getattr(body, attr) is not None:
                    raise ValueError(f"'{attr}' must be None for body with 0 DoF: '{body.name}'")

        if body.dof_control_mode is not None:
            if (
                body.dof_control_mode.ndim == 0
                and body.dof_control_mode
                not in (
                    DoFControlMode.POSITION_CONTROL,
                    DoFControlMode.VELOCITY_CONTROL,
                    DoFControlMode.TORQUE_CONTROL,
                )
                or body.dof_control_mode.ndim == 1
                and any(
                    y
                    not in (
                        DoFControlMode.POSITION_CONTROL,
                        DoFControlMode.VELOCITY_CONTROL,
                        DoFControlMode.TORQUE_CONTROL,
                    )
                    for y in body.dof_control_mode
                )
            ):
                raise ValueError(
                    "For Bullet, 'dof_control_mode' only supports POSITION_CONTROL, "
                    f"VELOCITY_CONTROL, and TORQUE_CONTROL: '{body.name}'"
                )

            if (
                body.dof_control_mode.ndim == 0
                and body.dof_control_mode == DoFControlMode.TORQUE_CONTROL
            ):
                self._p.setJointMotorControlArray(
                    self._body_ids[body.name],
                    self._dof_indices[body.name],
                    pybullet.VELOCITY_CONTROL,
                    forces=[0] * len(self._dof_indices[body.name]),
                )
            if (
                body.dof_control_mode.ndim == 1
                and DoFControlMode.TORQUE_CONTROL in body.dof_control_mode
            ):
                self._p.setJointMotorControlArray(
                    self._body_ids[body.name],
                    self._dof_indices[body.name][
                        body.dof_control_mode == DoFControlMode.TORQUE_CONTROL
                    ],
                    pybullet.VELOCITY_CONTROL,
                    forces=[0]
                    * len(
                        self._dof_indices[body.name][
                            body.dof_control_mode == DoFControlMode.TORQUE_CONTROL
                        ]
                    ),
                )
        elif len(self._dof_indices[body.name]) > 0:
            raise ValueError(
                f"For Bullet, 'dof_control_mode' is required for body with DoF > 0: '{body.name}'"
            )

        if body.link_collision_filter is not None:
            self._set_link_collision_filter(body)

        if any(
            not body.attr_array_default_flag[x] and getattr(body, x) is not None
            for x in self._ATTR_LINK_DYNAMICS
        ):
            self._set_link_dynamics(body)

        if not body.attr_array_default_flag["link_color"] and body.link_color is not None:
            self._set_link_color(body)

        if len(self._dof_indices[body.name]) > 0 and any(
            not body.attr_array_default_flag[x] and getattr(body, x) is not None
            for x in self._ATTR_DOF_DYNAMICS
        ):
            self._set_dof_dynamics(body)

        if any(
            getattr(body, x) is None
            for x in self._ATTR_LINK_DYNAMICS
            if x not in ("link_linear_damping", "link_angular_damping")
        ):
            dynamics_info = [
                self._p.getDynamicsInfo(self._body_ids[body.name], i)
                for i in range(-1, self._num_links[body.name] - 1)
            ]
            if body.link_lateral_friction is None:
                body.link_lateral_friction = [[x[1] for x in dynamics_info]]
                body.attr_array_default_flag["link_lateral_friction"] = True
            if body.link_spinning_friction is None:
                body.link_spinning_friction = [[x[7] for x in dynamics_info]]
                body.attr_array_default_flag["link_spinning_friction"] = True
            if body.link_rolling_friction is None:
                body.link_rolling_friction = [[x[6] for x in dynamics_info]]
                body.attr_array_default_flag["link_rolling_friction"] = True
            if body.link_restitution is None:
                body.link_restitution = [[x[5] for x in dynamics_info]]
                body.attr_array_default_flag["link_restitution"] = True

        if body.link_color is None:
            visual_data = self._p.getVisualShapeData(self._body_ids[body.name])
            body.link_color = [[x[7] for x in visual_data]]
            body.attr_array_default_flag["link_color"] = True

        if len(self._dof_indices[body.name]) > 0 and any(
            getattr(body, x) is None for x in self._ATTR_DOF_DYNAMICS
        ):
            joint_info = [
                self._p.getJointInfo(self._body_ids[body.name], j)
                for j in self._dof_indices[body.name]
            ]
            if body.dof_lower_limit is None:
                body.dof_lower_limit = [[x[8] for x in joint_info]]
                body.attr_array_default_flag["dof_lower_limit"] = True
            if body.dof_upper_limit is None:
                body.dof_upper_limit = [[x[9] for x in joint_info]]
                body.attr_array_default_flag["dof_upper_limit"] = True
            if body.dof_max_velocity is None:
                body.dof_max_velocity = [[x[11] for x in joint_info]]
                body.attr_array_default_flag["dof_max_velocity"] = True

        body.lock_attr_array()

        for attr in (
            ("link_color", "link_collision_filter")
            + self._ATTR_LINK_DYNAMICS
            + self._ATTR_DOF_DYNAMICS
        ):
            if body.attr_array_dirty_flag[attr]:
                body.attr_array_dirty_flag[attr] = False

    def _set_link_collision_filter(self, body):
        """ """
        link_collision_filter = body.get_attr_array("link_collision_filter", 0)
        if (
            not body.attr_array_locked["link_collision_filter"]
            and len(link_collision_filter) != self._num_links[body.name]
        ):
            raise ValueError(
                "Size of 'link_collision_filter' in the link dimension "
                f"({len(link_collision_filter)}) should match the number of links "
                f"({self._num_links[body.name]}): '{body.name}'"
            )
        for i in range(-1, self._num_links[body.name] - 1):
            self._p.setCollisionFilterGroupMask(
                self._body_ids[body.name],
                i,
                link_collision_filter[i + 1],
                link_collision_filter[i + 1],
            )

    def _set_link_dynamics(self, body, dirty_only=False):
        """ """
        for attr in self._ATTR_LINK_DYNAMICS:
            if attr in ("link_linear_damping", "link_angular_damping"):
                continue
            if (
                not body.attr_array_locked[attr]
                and getattr(body, attr) is not None
                and len(body.get_attr_array(attr, 0)) != self._num_links[body.name]
            ):
                raise ValueError(
                    f"Size of '{attr}' in the link dimension ({len(body.get_attr_array(attr, 0))}) "
                    f"should match the number of links ({self._num_links[body.name]}): "
                    f"'{body.name}'"
                )
        kwargs = {}
        if (
            not dirty_only
            and not body.attr_array_default_flag["link_lateral_friction"]
            and body.link_lateral_friction is not None
            or body.attr_array_dirty_flag["link_lateral_friction"]
        ):
            kwargs["lateralFriction"] = body.get_attr_array("link_lateral_friction", 0)
        if (
            not dirty_only
            and not body.attr_array_default_flag["link_spinning_friction"]
            and body.link_spinning_friction is not None
            or body.attr_array_dirty_flag["link_spinning_friction"]
        ):
            kwargs["spinningFriction"] = body.get_attr_array("link_spinning_friction", 0)
        if (
            not dirty_only
            and not body.attr_array_default_flag["link_rolling_friction"]
            and body.link_rolling_friction is not None
            or body.attr_array_dirty_flag["link_rolling_friction"]
        ):
            kwargs["rollingFriction"] = body.get_attr_array("link_rolling_friction", 0)
        if (
            not dirty_only
            and not body.attr_array_default_flag["link_restitution"]
            and body.link_restitution is not None
            or body.attr_array_dirty_flag["link_restitution"]
        ):
            kwargs["restitution"] = body.get_attr_array("link_restitution", 0)
        if len(kwargs) > 0:
            for i in range(-1, self._num_links[body.name] - 1):
                self._p.changeDynamics(
                    self._body_ids[body.name], i, **{k: v[i + 1] for k, v in kwargs.items()}
                )
        # Bullet only sets `linearDamping` and `angularDamping` for link index -1. See:
        #     https://github.com/bulletphysics/bullet3/blob/740d2b978352b16943b24594572586d95d476466/examples/SharedMemory/PhysicsClientC_API.cpp#L3419
        #     https://github.com/bulletphysics/bullet3/blob/740d2b978352b16943b24594572586d95d476466/examples/SharedMemory/PhysicsClientC_API.cpp#L3430
        kwargs = {}
        if (
            not dirty_only
            and not body.attr_array_default_flag["link_linear_damping"]
            and body.link_linear_damping is not None
            or body.attr_array_dirty_flag["link_linear_damping"]
        ):
            kwargs["linearDamping"] = body.get_attr_array("link_linear_damping", 0)
        if (
            not dirty_only
            and not body.attr_array_default_flag["link_angular_damping"]
            and body.link_angular_damping is not None
            or body.attr_array_dirty_flag["link_angular_damping"]
        ):
            kwargs["angularDamping"] = body.get_attr_array("link_angular_damping", 0)
        if len(kwargs) > 0:
            self._p.changeDynamics(
                self._body_ids[body.name], -1, **{k: v for k, v in kwargs.items()}
            )

    def _set_link_color(self, body):
        """ """
        link_color = body.get_attr_array("link_color", 0)
        if (
            not body.attr_array_locked["link_color"]
            and len(link_color) != self._num_links[body.name]
        ):
            raise ValueError(
                f"Size of 'link_color' in the link dimension ({len(link_color)}) should match the "
                f"number of links: '{body.name}' ({self._num_links[body.name]})"
            )
        for i in range(-1, self._num_links[body.name] - 1):
            self._p.changeVisualShape(self._body_ids[body.name], i, rgbaColor=link_color[i + 1])

    def _set_dof_dynamics(self, body, dirty_only=False):
        """ """
        for attr in self._ATTR_DOF_DYNAMICS:
            if (
                not body.attr_array_locked[attr]
                and getattr(body, attr) is not None
                and len(body.get_attr_array(attr, 0)) != len(self._dof_indices[body.name])
            ):
                raise ValueError(
                    f"Size of '{attr}' in the DoF dimension ({len(body.get_attr_array(attr, 0))}) "
                    f"should match the number of DoFs ({len(self._dof_indices[body.name])}): "
                    f"'{body.name}'"
                )
        kwargs = {}
        if (
            not dirty_only
            and not body.attr_array_default_flag["dof_lower_limit"]
            and body.dof_lower_limit is not None
            or body.attr_array_dirty_flag["dof_lower_limit"]
        ):
            kwargs["jointLowerLimit"] = body.get_attr_array("dof_lower_limit", 0)
        if (
            not dirty_only
            and not body.attr_array_default_flag["dof_upper_limit"]
            and body.dof_upper_limit is not None
            or body.attr_array_dirty_flag["dof_upper_limit"]
        ):
            kwargs["jointUpperLimit"] = body.get_attr_array("dof_upper_limit", 0)
        if len(kwargs) > 0:
            for i, j in enumerate(self._dof_indices[body.name]):
                self._p.changeDynamics(
                    self._body_ids[body.name], j, **{k: v[i] for k, v in kwargs.items()}
                )

    def _set_body_device(self, body):
        """ """
        body.set_device(self.device)

    def _set_body_callback(self, body):
        """ """
        body.set_callback_collect_dof_state(self._collect_dof_state)
        body.set_callback_collect_link_state(self._collect_link_state)

    def _collect_dof_state(self, body):
        """ """
        if len(self._dof_indices[body.name]) > 0:
            joint_states = self._p.getJointStates(
                self._body_ids[body.name], self._dof_indices[body.name]
            )
            dof_state = [x[0:2] for x in joint_states]
            body.dof_state = [dof_state]

    def _collect_link_state(self, body):
        """ """
        pos, orn = self._p.getBasePositionAndOrientation(self._body_ids[body.name])
        lin, ang = self._p.getBaseVelocity(self._body_ids[body.name])
        link_state = [pos + orn + lin + ang]
        if self._num_links[body.name] > 1:
            link_indices = [*range(0, self._num_links[body.name] - 1)]
            # Need to set computeForwardKinematics=1. See:
            #     https://github.com/bulletphysics/bullet3/issues/2806
            link_states = self._p.getLinkStates(
                self._body_ids[body.name],
                link_indices,
                computeLinkVelocity=1,
                computeForwardKinematics=1,
            )
            link_state += [x[4] + x[5] + x[6] + x[7] for x in link_states]
        body.link_state = [link_state]

    def _load_camera(self, camera):
        """ """
        self._set_projection_matrix(camera)
        self._set_view_matrix(camera)

        self._image_cache[camera.name] = {}
        self._clear_image_cache(camera)

        camera.lock_attr_array()

    def _set_projection_matrix(self, camera):
        """ """
        self._projection_matrix[camera.name] = self._p.computeProjectionMatrixFOV(
            camera.get_attr_array("vertical_fov", 0),
            camera.get_attr_array("width", 0) / camera.get_attr_array("height", 0),
            camera.get_attr_array("near", 0),
            camera.get_attr_array("far", 0),
        )

    def _set_view_matrix(self, camera):
        """ """
        if camera.target is not None and camera.up_vector is not None:
            self._view_matrix[camera.name] = self._p.computeViewMatrix(
                camera.get_attr_array("position", 0),
                camera.get_attr_array("target", 0),
                camera.get_attr_array("up_vector", 0),
            )
        elif camera.orientation is not None:
            R = np.array(
                self._p.getMatrixFromQuaternion(camera.get_attr_array("orientation", 0)),
                dtype=np.float32,
            ).reshape(3, 3)
            view_matrix = np.eye(4, dtype=np.float32)
            view_matrix[:3, :3] = R
            view_matrix[3, :3] = -1 * camera.get_attr_array("position", 0).dot(R)
            self._view_matrix[camera.name] = tuple(view_matrix.flatten())
        else:
            raise ValueError(
                "For Bullet, either ('target', 'up_vector') or 'orientation' is required to be "
                f"set: {camera.name}"
            )

    def _clear_image_cache(self, camera):
        """ """
        self._image_cache[camera.name]["color"] = None
        self._image_cache[camera.name]["depth"] = None
        self._image_cache[camera.name]["segmentation"] = None

    def _set_camera_device(self, camera):
        """ """
        camera.set_device(self._graphics_device)

    def _set_camera_callback(self, camera):
        """ """
        camera.set_callback_render_color(self._render_color)
        camera.set_callback_render_depth(self._render_depth)
        camera.set_callback_render_segmentation(self._render_segmentation)

    def _render_color(self, camera):
        """ """
        for body in self._scene.bodies:
            if body.attr_array_dirty_flag["link_color"]:
                self._set_link_color(body)
                body.attr_array_dirty_flag["link_color"] = False
                if self._image_cache[camera.name]["color"] is not None:
                    self._image_cache[camera.name]["color"] = None

        self._check_and_update_camera(camera)
        if self._image_cache[camera.name]["color"] is None:
            self._render(camera)
        camera.color = self._image_cache[camera.name]["color"][None]

    def _render_depth(self, camera):
        """ """
        self._check_and_update_camera(camera)
        if self._image_cache[camera.name]["depth"] is None:
            self._render(camera)
        depth = (
            camera.get_attr_array("far", 0)
            * camera.get_attr_array("near", 0)
            / (
                camera.get_attr_array("far", 0)
                - (camera.get_attr_array("far", 0) - camera.get_attr_array("near", 0))
                * self._image_cache[camera.name]["depth"]
            )
        )
        depth[self._image_cache[camera.name]["depth"] == 1.0] = 0.0
        camera.depth = depth[None]

    def _render_segmentation(self, camera):
        """ """
        for body in self._scene.bodies:
            for attr in ("link_segmentation_id",):
                if getattr(body, attr) is not None:
                    raise ValueError(f"'{attr}' is not supported in Bullet: '{body.name}'")

        self._check_and_update_camera(camera)
        if self._image_cache[camera.name]["segmentation"] is None:
            self._render(camera)
        camera.segmentation = self._image_cache[camera.name]["segmentation"][None]

    def _check_and_update_camera(self, camera):
        """ """
        if any(camera.attr_array_dirty_flag[x] for x in self._ATTR_PROJECTION_MATRIX):
            self._set_projection_matrix(camera)
            for attr in self._ATTR_PROJECTION_MATRIX:
                if camera.attr_array_dirty_flag[attr]:
                    camera.attr_array_dirty_flag[attr] = False
            self._clear_image_cache(camera)
        if any(camera.attr_array_dirty_flag[x] for x in self._ATTR_VIEW_MATRIX):
            self._set_view_matrix(camera)
            for attr in self._ATTR_VIEW_MATRIX:
                if camera.attr_array_dirty_flag[attr]:
                    camera.attr_array_dirty_flag[attr] = False
            self._clear_image_cache(camera)

    def _render(self, camera):
        """ """
        (
            _,
            _,
            self._image_cache[camera.name]["color"],
            self._image_cache[camera.name]["depth"],
            self._image_cache[camera.name]["segmentation"],
        ) = self._p.getCameraImage(
            camera.get_attr_array("width", 0),
            camera.get_attr_array("height", 0),
            viewMatrix=self._view_matrix[camera.name],
            projectionMatrix=self._projection_matrix[camera.name],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

    def _set_scene_callback(self):
        """ """
        self._scene.set_callback_add_camera(self._add_camera)
        self._scene.set_callback_remove_camera(self._remove_camera)

    def _add_camera(self, camera):
        """ """
        self._load_camera(camera)
        self._set_camera_callback(camera)

    def _remove_camera(self, camera):
        """ """
        del self._projection_matrix[camera.name]
        del self._view_matrix[camera.name]
        del self._image_cache[camera.name]

    def _set_viewer_camera_pose(self, position, target):
        """ """
        disp = [x - y for x, y in zip(position, target)]
        dist = np.linalg.norm(disp)
        yaw = np.arctan2(disp[0], -disp[1])
        yaw = np.rad2deg(yaw)
        pitch = np.arctan2(-disp[2], np.linalg.norm((disp[0], disp[1])))
        pitch = np.rad2deg(pitch)

        self._p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    def _clear_state(self):
        """ """
        for body in self._scene.bodies:
            body.dof_state = None
            body.link_state = None

    def _clear_image(self):
        """ """
        for camera in self._scene.cameras:
            camera.color = None
            camera.depth = None
            camera.segmentation = None

            self._clear_image_cache(camera)

    def step(self):
        """ """
        if self._cfg.RENDER:
            # Simulate real-time rendering with sleep if computation takes less than real time.
            time_spent = time.time() - self._last_frame_time
            time_sleep = self._cfg.TIME_STEP - time_spent
            if time_sleep > 0:
                time.sleep(time_sleep)
            self._last_frame_time = time.time()

        for body in reversed(self._scene_cache.bodies):
            if body.name not in [x.name for x in self._scene.bodies]:
                # Remove body.
                with self._disable_cov_rendering():
                    self._p.removeBody(self._body_ids[body.name])
                    del self._body_ids[body.name]
                    del self._dof_indices[body.name]
                    del self._num_links[body.name]
                    self._scene_cache.remove_body(body)
        for body in self._scene.bodies:
            if body.name not in [x.name for x in self._scene_cache.bodies]:
                # Add body.
                with self._disable_cov_rendering():
                    self._load_body(body)
                    self._cache_body_and_set_control_and_props(body)
                    self._set_body_device(body)
                    self._set_body_callback(body)

        for body in self._scene.bodies:
            for attr in (
                "link_segmentation_id",
                "dof_has_limits",
                "dof_armature",
            ):
                if getattr(body, attr) is not None:
                    raise ValueError(f"'{attr}' is not supported in Bullet: '{body.name}'")

            if body.env_ids_load is not None and len(body.env_ids_load) == 0:
                for attr in ("env_ids_reset_base_state", "env_ids_reset_dof_state"):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"For Bullet, '{attr}' should be None if 'env_ids_load' is set to []: "
                            f"'{body.name}'"
                        )
                continue

            if body.env_ids_reset_base_state is not None:
                if not np.array_equal(body.env_ids_reset_base_state.cpu(), [0]):
                    raise ValueError(
                        "For Bullet, 'env_ids_reset_base_state' must be either None or [0]: "
                        f"'{body.name}'"
                    )
                if body.initial_base_position is None and body.initial_base_velocity is None:
                    raise ValueError(
                        "'initial_base_position' and 'initial_base_velocity' cannot be both None "
                        f"when 'env_ids_reset_base_state' is used: {body.name}"
                    )
                if body.initial_base_position is not None:
                    self._reset_base_position(body)
                if body.initial_base_velocity is not None:
                    self._reset_base_velocity(body)
                body.env_ids_reset_base_state = None

            if body.attr_array_dirty_flag["scale"]:
                raise ValueError(
                    f"For Bullet, 'scale' can only be changed before each reset: '{body.name}'"
                )
            if body.attr_array_dirty_flag["link_collision_filter"]:
                self._set_link_collision_filter(body)
                body.attr_array_dirty_flag["link_collision_filter"] = False
            if any(body.attr_array_dirty_flag[x] for x in self._ATTR_LINK_DYNAMICS):
                self._set_link_dynamics(body, dirty_only=True)
                for attr in self._ATTR_LINK_DYNAMICS:
                    if body.attr_array_dirty_flag[attr]:
                        body.attr_array_dirty_flag[attr] = False
            if body.attr_array_dirty_flag["link_color"]:
                self._set_link_color(body)
                body.attr_array_dirty_flag["link_color"] = False

            if len(self._dof_indices[body.name]) == 0:
                for attr in (
                    "dof_lower_limit",
                    "dof_upper_limit",
                    "dof_control_mode",
                    "dof_max_force",
                    "dof_max_velocity",
                    "dof_position_gain",
                    "dof_velocity_gain",
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
                if not np.array_equal(body.env_ids_reset_dof_state.cpu(), [0]):
                    raise ValueError(
                        "For Bullet, 'env_ids_reset_dof_state' must be either None or [0]: "
                        f"'{body.name}'"
                    )
                self._reset_dof_state(body)
                body.env_ids_reset_dof_state = None

            if any(body.attr_array_dirty_flag[x] for x in self._ATTR_DOF_DYNAMICS):
                self._set_dof_dynamics(body, dirty_only=True)
                for attr in self._ATTR_DOF_DYNAMICS:
                    if body.attr_array_dirty_flag[attr]:
                        body.attr_array_dirty_flag[attr] = False

            if body.attr_array_dirty_flag["dof_control_mode"]:
                raise ValueError(
                    "For Bullet, 'dof_control_mode' cannot be changed after each reset: "
                    f"'{body.name}'"
                )
            # The redundant if-else block below is an artifact due to `setJointMotorControlArray()`
            # not supporting `maxVelocity`. `setJointMotorControlArray()` is still preferred when
            # `maxVelocity` is not needed due to better speed performance.
            if body.attr_array_default_flag["dof_max_velocity"]:
                kwargs = {}
                if body.dof_target_position is not None:
                    kwargs["targetPositions"] = body.get_attr_tensor("dof_target_position", 0)
                if body.dof_target_velocity is not None:
                    kwargs["targetVelocities"] = body.get_attr_tensor("dof_target_velocity", 0)
                if body.dof_position_gain is not None:
                    kwargs["positionGains"] = body.get_attr_array("dof_position_gain", 0)
                if body.dof_velocity_gain is not None:
                    kwargs["velocityGains"] = body.get_attr_array("dof_velocity_gain", 0)
                if body.dof_control_mode.ndim == 0:
                    if body.dof_max_force is not None:
                        if body.dof_control_mode not in (
                            DoFControlMode.POSITION_CONTROL,
                            DoFControlMode.VELOCITY_CONTROL,
                        ):
                            raise ValueError(
                                "For Bullet, 'dof_max_force' can only be set in POSITION_CONTROL "
                                f"and VELOCITY_CONTROL modes: '{body.name}'"
                            )
                        kwargs["forces"] = body.get_attr_array("dof_max_force", 0)
                    if body.dof_force is not None:
                        if body.dof_control_mode != DoFControlMode.TORQUE_CONTROL:
                            raise ValueError(
                                "For Bullet, 'dof_force' can only be set in the TORQUE_CONTROL "
                                f"mode: '{body.name}'"
                            )
                        kwargs["forces"] = body.get_attr_tensor("dof_force", 0)
                    self._p.setJointMotorControlArray(
                        self._body_ids[body.name],
                        self._dof_indices[body.name],
                        self._DOF_CONTROL_MODE_MAP[body.dof_control_mode.item()],
                        **kwargs,
                    )
                if body.dof_control_mode.ndim == 1:
                    if body.dof_max_force is not None:
                        if (
                            DoFControlMode.POSITION_CONTROL not in body.dof_control_mode
                            and DoFControlMode.VELOCITY_CONTROL not in body.dof_control_mode
                        ):
                            raise ValueError(
                                "For Bullet, 'dof_max_force' can only be set in POSITION_CONTROL "
                                f"and VELOCITY_CONTROL modes: '{body.name}'"
                            )
                        kwargs["forces"] = body.get_attr_array("dof_max_force", 0)
                    if DoFControlMode.POSITION_CONTROL in body.dof_control_mode:
                        self._p.setJointMotorControlArray(
                            self._body_ids[body.name],
                            self._dof_indices[body.name][
                                body.dof_control_mode == DoFControlMode.POSITION_CONTROL
                            ],
                            self._DOF_CONTROL_MODE_MAP[DoFControlMode.POSITION_CONTROL],
                            **{
                                k: v[body.dof_control_mode == DoFControlMode.POSITION_CONTROL]
                                for k, v in kwargs.items()
                            },
                        )
                    if DoFControlMode.VELOCITY_CONTROL in body.dof_control_mode:
                        self._p.setJointMotorControlArray(
                            self._body_ids[body.name],
                            self._dof_indices[body.name][
                                body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL
                            ],
                            self._DOF_CONTROL_MODE_MAP[DoFControlMode.VELOCITY_CONTROL],
                            **{
                                k: v[body.dof_control_mode == DoFControlMode.VELOCITY_CONTROL]
                                for k, v in kwargs.items()
                            },
                        )
                    if "forces" in kwargs:
                        del kwargs["forces"]
                    if body.dof_force is not None:
                        if DoFControlMode.TORQUE_CONTROL not in body.dof_control_mode:
                            raise ValueError(
                                "For Bullet, 'dof_force' can only be set in the TORQUE_CONTROL "
                                f"mode: '{body.name}'"
                            )
                        kwargs["forces"] = body.get_attr_tensor("dof_force", 0)
                    if DoFControlMode.TORQUE_CONTROL in body.dof_control_mode:
                        self._p.setJointMotorControlArray(
                            self._body_ids[body.name],
                            self._dof_indices[body.name][
                                body.dof_control_mode == DoFControlMode.TORQUE_CONTROL
                            ],
                            self._DOF_CONTROL_MODE_MAP[DoFControlMode.TORQUE_CONTROL],
                            **{
                                k: v[body.dof_control_mode == DoFControlMode.TORQUE_CONTROL]
                                for k, v in kwargs.items()
                            },
                        )
            else:
                kwargs = {}
                if body.dof_target_position is not None:
                    kwargs["targetPosition"] = body.get_attr_tensor("dof_target_position", 0)
                if body.dof_target_velocity is not None:
                    kwargs["targetVelocity"] = body.get_attr_tensor("dof_target_velocity", 0)
                if body.dof_position_gain is not None:
                    kwargs["positionGain"] = body.get_attr_array("dof_position_gain", 0)
                if body.dof_velocity_gain is not None:
                    kwargs["velocityGain"] = body.get_attr_array("dof_velocity_gain", 0)
                # For Bullet, 'dof_max_velocity' has no effect when not in the POSITION_CONROL mode.
                kwargs["maxVelocity"] = body.get_attr_array("dof_max_velocity", 0)
                if body.dof_control_mode.ndim == 0:
                    if body.dof_max_force is not None:
                        if body.dof_control_mode not in (
                            DoFControlMode.POSITION_CONTROL,
                            DoFControlMode.VELOCITY_CONTROL,
                        ):
                            raise ValueError(
                                "For Bullet, 'dof_max_force' can only be set in POSITION_CONTROL "
                                f"and VELOCITY_CONTROL modes: '{body.name}'"
                            )
                        kwargs["force"] = body.get_attr_array("dof_max_force", 0)
                    if body.dof_force is not None:
                        if body.dof_control_mode != DoFControlMode.TORQUE_CONTROL:
                            raise ValueError(
                                "For Bullet, 'dof_force' can only be set in the TORQUE_CONTROL "
                                f"mode: '{body.name}'"
                            )
                        kwargs["force"] = body.get_attr_tensor("dof_force", 0)
                    for i, j in enumerate(self._dof_indices[body.name]):
                        self._p.setJointMotorControl2(
                            self._body_ids[body.name],
                            j,
                            self._DOF_CONTROL_MODE_MAP[body.dof_control_mode.item()],
                            **{k: v[i] for k, v in kwargs.items()},
                        )
                if body.dof_control_mode.ndim == 1:
                    if (
                        body.dof_max_force is not None
                        and DoFControlMode.POSITION_CONTROL not in body.dof_control_mode
                        and DoFControlMode.VELOCITY_CONTROL not in body.dof_control_mode
                    ):
                        raise ValueError(
                            "For Bullet, 'dof_max_force' can only be set in POSITION_CONTROL and "
                            f"VELOCITY_CONTROL modes: '{body.name}'"
                        )
                    if (
                        body.dof_force is not None
                        and DoFControlMode.TORQUE_CONTROL not in body.dof_control_mode
                    ):
                        raise ValueError(
                            "For Bullet, 'dof_force' can only be set in the TORQUE_CONTROL mode: "
                            f"'{body.name}'"
                        )
                    for i, j in enumerate(self._dof_indices[body.name]):
                        if body.dof_control_mode[i] in (
                            DoFControlMode.POSITION_CONTROL,
                            DoFControlMode.VELOCITY_CONTROL,
                        ):
                            if "force" in kwargs:
                                del kwargs["force"]
                            if body.dof_max_force is not None:
                                kwargs["force"] = body.get_attr_array("dof_max_force", 0)
                            if body.dof_force is not None and not np.isnan(
                                body.get_attr_tensor("dof_force", 0)[i]
                            ):
                                raise ValueError(
                                    "For Bullet, 'dof_force' is required to be np.nan for DoF "
                                    f"({i}) in POSITION_CONTROL and VELOCITY modes: {body.name}"
                                )
                            self._p.setJointMotorControl2(
                                self._body_ids[body.name],
                                j,
                                self._DOF_CONTROL_MODE_MAP[body.dof_control_mode[i].item()],
                                **{k: v[i] for k, v in kwargs.items()},
                            )
                        if body.dof_control_mode[i] == DoFControlMode.TORQUE_CONTROL:
                            if "force" in kwargs:
                                del kwargs["force"]
                            if body.dof_force is not None:
                                kwargs["force"] = body.get_attr_tensor("dof_force", 0)
                            if body.dof_max_force is not None and not np.isnan(
                                body.get_attr_array("dof_max_force", 0)[i]
                            ):
                                raise ValueError(
                                    "For Bullet, 'dof_max_force' is required to be np.nan for DoF "
                                    f"({i}) in the TORQUE_CONTROL mode: {body.name}"
                                )
                            self._p.setJointMotorControl2(
                                self._body_ids[body.name],
                                j,
                                self._DOF_CONTROL_MODE_MAP[body.dof_control_mode[i].item()],
                                **{k: v[i] for k, v in kwargs.items()},
                            )

        self._p.stepSimulation()

        self._clear_state()
        self._clear_image()
        self._contact = None

    @property
    def contact(self):
        """ """
        if self._contact is None:
            self._contact = self._collect_contact()
        return self._contact

    def _collect_contact(self):
        """ """
        pts = self._p.getContactPoints()
        if len(pts) == 0:
            contact_array = create_contact_array(0)
        else:
            kwargs = {}
            kwargs["body_id_a"] = [x[1] if x[1] != self._body_id_ground_plane else -1 for x in pts]
            kwargs["body_id_b"] = [x[2] if x[2] != self._body_id_ground_plane else -1 for x in pts]
            kwargs["link_id_a"] = [x[3] + 1 for x in pts]
            kwargs["link_id_b"] = [x[4] + 1 for x in pts]
            kwargs["position_a_world"] = [x[5] for x in pts]
            kwargs["position_b_world"] = [x[6] for x in pts]
            kwargs["position_a_link"] = np.nan
            kwargs["position_b_link"] = np.nan
            kwargs["normal"] = [x[7] for x in pts]
            kwargs["force"] = [x[9] for x in pts]
            contact_array = create_contact_array(len(pts), **kwargs)
        return [contact_array]

    def close(self):
        """ """
        if self._connected:
            for name in self._plugins:
                self._p.unloadPlugin(self._plugins[name])
            self._p.disconnect()
            self._connected = False
