# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data
import numpy as np
import time

from contextlib import contextmanager

from easysim.simulators.simulator import Simulator
from easysim.constants import DoFControlMode
from easysim.contact import create_contact_array


class Bullet(Simulator):
    """Bullet simulator."""

    _DOF_CONTROL_MODE_MAP = {
        DoFControlMode.POSITION_CONTROL: pybullet.POSITION_CONTROL,
        DoFControlMode.VELOCITY_CONTROL: pybullet.VELOCITY_CONTROL,
        DoFControlMode.TORQUE_CONTROL: pybullet.TORQUE_CONTROL,
    }

    def __init__(self, cfg):
        """ """
        super().__init__(cfg)

        if self._cfg.NUM_ENVS != 1:
            raise ValueError("Number of envs must be 1 for Bullet")

        self._connected = False
        self._last_frame_time = 0.0

    def reset(self, bodies, env_ids):
        """ """
        if not self._connected:
            if self._cfg.RENDER:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._connected = True

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

            self._body_id_ground_plane = self._load_ground_plane()

            self._body_ids = {}
            self._dof_indices = {}
            self._num_links = {}
            self._bodies = type(bodies)()

            for body in bodies:
                self._load_body(body)
                self._cache_and_set_control(body)

            if (
                self._cfg.RENDER
                and self._cfg.INIT_VIEWER_CAMERA_POSITION != (None, None, None)
                and self._cfg.INIT_VIEWER_CAMERA_TARGET != (None, None, None)
            ):
                self._set_viewer_camera_pose(
                    self._cfg.INIT_VIEWER_CAMERA_POSITION, self._cfg.INIT_VIEWER_CAMERA_TARGET
                )

        self._contact = None

    @contextmanager
    def _disable_cov_rendering(self):
        """ """
        try:
            if self._cfg.RENDER:
                self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
            yield
        finally:
            if self._cfg.RENDER:
                self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    def _load_ground_plane(self):
        """ """
        return self._p.loadURDF("plane_implicit.urdf")

    def _load_body(self, body):
        """ """
        kwargs = {}
        if body.use_fixed_base is not None:
            kwargs["useFixedBase"] = body.use_fixed_base
        self._body_ids[body.name] = self._p.loadURDF(body.urdf_file, **kwargs)

        dof_indices = []
        for j in range(self._p.getNumJoints(self._body_ids[body.name])):
            joint_info = self._p.getJointInfo(self._body_ids[body.name], j)
            if joint_info[2] != self._p.JOINT_FIXED:
                dof_indices.append(j)
        self._dof_indices[body.name] = np.asanyarray(dof_indices, dtype=np.int64)

        self._num_links[body.name] = self._p.getNumJoints(self._body_ids[body.name]) + 1

        # Reset base state.
        if body.initial_base_position is not None:
            if body.initial_base_position.dim() == 1:
                self._p.resetBasePositionAndOrientation(
                    self._body_ids[body.name],
                    body.initial_base_position[:3],
                    body.initial_base_position[3:],
                )
            if body.initial_base_position.dim() == 2:
                self._p.resetBasePositionAndOrientation(
                    self._body_ids[body.name],
                    body.initial_base_position[0, :3],
                    body.initial_base_position[0, 3:],
                )
        if body.initial_base_velocity is not None:
            kwargs = {}
            if body.initial_base_velocity.dim() == 1:
                kwargs["linearVelocity"] = body.initial_base_velocity[:3]
                kwargs["angularVelocity"] = body.initial_base_velocity[3:]
            if body.initial_base_velocity.dim() == 2:
                kwargs["linearVelocity"] = body.initial_base_velocity[0, :3]
                kwargs["angularVelocity"] = body.initial_base_velocity[0, 3:]
            self._p.resetBaseVelocity(self._body_ids[body.name], **kwargs)

        # Reset DoF state.
        if len(self._dof_indices[body.name]) == 0:
            if body.initial_dof_position is not None and body.initial_dof_position.shape[-1] != 0:
                raise ValueError(
                    f"Body '{body.name}' has 0 DoF but is given a non-empty 'initial_dof_position'"
                )
            if body.initial_dof_velocity is not None and body.initial_dof_velocity.shape[-1] != 0:
                raise ValueError(
                    f"Body '{body.name}' has 0 DoF but is given a non-empty 'initial_dof_velocity'"
                )
        if body.initial_dof_position is not None:
            for i, j in enumerate(self._dof_indices[body.name]):
                kwargs = {}
                if body.initial_dof_velocity is not None:
                    if body.initial_dof_velocity.dim() == 1:
                        kwargs["targetVelocity"] = body.initial_dof_velocity[i]
                    if body.initial_dof_velocity.dim() == 2:
                        kwargs["targetVelocity"] = body.initial_dof_velocity[0, i]
                if body.initial_dof_position.dim() == 1:
                    self._p.resetJointState(
                        self._body_ids[body.name], j, body.initial_dof_position[i], **kwargs
                    )
                if body.initial_dof_position.dim() == 2:
                    self._p.resetJointState(
                        self._body_ids[body.name], j, body.initial_dof_position[0, i], **kwargs
                    )
        elif body.initial_dof_velocity is not None:
            raise ValueError(
                "For Bullet, cannot reset 'initial_dof_velocity' without resetting "
                f"'initial_dof_position': '{body.name}'"
            )

        body.contact_id = self._body_ids[body.name]

    def _cache_and_set_control(self, body):
        """ """
        x = type(body)()
        x.name = body.name
        for attr in (
            "link_color",
            "link_collision_filter",
            "link_lateral_friction",
            "link_spinning_friction",
            "link_rolling_friction",
            "link_restitution",
            "link_linear_damping",
            "link_angular_damping",
        ):
            if getattr(body, attr) is not None:
                setattr(x, attr, getattr(body, attr).copy())

        if body.dof_control_mode is not None:
            if len(self._dof_indices[x.name]) == 0:
                raise ValueError(
                    f"Body '{x.name}' has 0 DoF but is given a non-empty 'dof_control_mode'"
                )
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
                    [
                        y
                        not in (
                            DoFControlMode.POSITION_CONTROL,
                            DoFControlMode.VELOCITY_CONTROL,
                            DoFControlMode.TORQUE_CONTROL,
                        )
                        for y in body.dof_control_mode
                    ]
                )
            ):
                raise ValueError(
                    "Bullet only supports POSITION_CONTROL, VELOCITY_CONTROL, and TORQUE_CONTROL "
                    f"modes: '{x.name}'"
                )
            x.dof_control_mode = body.dof_control_mode.copy()

            if x.dof_control_mode.ndim == 0 and x.dof_control_mode == DoFControlMode.TORQUE_CONTROL:
                self._p.setJointMotorControlArray(
                    self._body_ids[x.name],
                    self._dof_indices[x.name],
                    self._p.VELOCITY_CONTROL,
                    forces=[0] * len(self._dof_indices[x.name]),
                )
            if x.dof_control_mode.ndim == 1 and DoFControlMode.TORQUE_CONTROL in x.dof_control_mode:
                self._p.setJointMotorControlArray(
                    self._body_ids[x.name],
                    self._dof_indices[x.name][x.dof_control_mode == DoFControlMode.TORQUE_CONTROL],
                    self._p.VELOCITY_CONTROL,
                    forces=[0]
                    * len(
                        self._dof_indices[x.name][
                            x.dof_control_mode == DoFControlMode.TORQUE_CONTROL
                        ]
                    ),
                )
        elif len(self._dof_indices[x.name]) != 0:
            raise ValueError(
                f"For Bullet, 'dof_control_mode' is required for body with DoF > 0: '{x.name}'"
            )

        self._bodies.append(x)

        if self._bodies[-1].link_color is not None:
            self._set_link_color(self._bodies[-1])
        if self._bodies[-1].link_collision_filter is not None:
            self._set_link_collision_filter(self._bodies[-1])
        if (
            self._bodies[-1].link_lateral_friction is not None
            or self._bodies[-1].link_spinning_friction is not None
            or self._bodies[-1].link_rolling_friction is not None
            or self._bodies[-1].link_restitution is not None
            or self._bodies[-1].link_linear_damping is not None
            or self._bodies[-1].link_angular_damping is not None
        ):
            self._set_link_dynamics(self._bodies[-1])

    def _set_link_color(self, body):
        """ """
        link_color = self._get_body_attr_array(body, "link_color", 2)
        if len(link_color) != self._num_links[body.name]:
            raise ValueError(
                f"Size of 'link_color' in the link dimension ({len(link_color)}) should match the "
                f"number of links: '{body.name}' ({self._num_links[body.name]})"
            )
        for i in range(-1, self._num_links[body.name] - 1):
            self._p.changeVisualShape(self._body_ids[body.name], i, rgbaColor=link_color[i + 1])

    def _get_body_attr_array(self, body, attr, ndim):
        """ """
        array = getattr(body, attr)
        if array.ndim == ndim:
            return array
        if array.ndim == ndim + 1:
            if len(array) != 1:
                raise ValueError(
                    f"For Bullet, a per-env specification for '{attr}' must have the first "
                    f"dimension (number of envs) of size 1: '{body.name}'"
                )
            return array[0]

    def _set_link_collision_filter(self, body):
        """ """
        link_collision_filter = self._get_body_attr_array(body, "link_collision_filter", 1)
        if len(link_collision_filter) != self._num_links[body.name]:
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

    def _set_link_dynamics(self, body):
        """ """
        for attr in (
            "link_lateral_friction",
            "link_spinning_friction",
            "link_rolling_friction",
            "link_restitution",
        ):
            if getattr(body, attr) is not None:
                if len(self._get_body_attr_array(body, attr, 1)) != self._num_links[body.name]:
                    raise ValueError(
                        f"Size of '{attr}' in the link dimension "
                        f"({len(self._get_body_attr_array(body, attr, 1))}) should match the "
                        f"number of links ({self._num_links[body.name]}): '{body.name}'"
                    )
        kwargs = {}
        if body.link_lateral_friction is not None:
            kwargs["lateralFriction"] = self._get_body_attr_array(body, "link_lateral_friction", 1)
        if body.link_spinning_friction is not None:
            kwargs["spinningFriction"] = self._get_body_attr_array(
                body, "link_spinning_friction", 1
            )
        if body.link_rolling_friction is not None:
            kwargs["rollingFriction"] = self._get_body_attr_array(body, "link_rolling_friction", 1)
        if body.link_restitution is not None:
            kwargs["restitution"] = self._get_body_attr_array(body, "link_restitution", 1)
        if len(kwargs) > 0:
            for i in range(-1, self._num_links[body.name] - 1):
                self._p.changeDynamics(
                    self._body_ids[body.name], i, **{k: v[i + 1] for k, v in kwargs.items()}
                )
        # Bullet only sets `linearDamping` and `angularDamping` for link index -1. See:
        #     https://github.com/bulletphysics/bullet3/blob/740d2b978352b16943b24594572586d95d476466/examples/SharedMemory/PhysicsClientC_API.cpp#L3419
        #     https://github.com/bulletphysics/bullet3/blob/740d2b978352b16943b24594572586d95d476466/examples/SharedMemory/PhysicsClientC_API.cpp#L3430
        kwargs = {}
        if body.link_linear_damping is not None:
            kwargs["linearDamping"] = self._get_body_attr_array(body, "link_linear_damping", 0)
        if body.link_angular_damping is not None:
            kwargs["angularDamping"] = self._get_body_attr_array(body, "link_angular_damping", 0)
        if len(kwargs) > 0:
            self._p.changeDynamics(
                self._body_ids[body.name], -1, **{k: v for k, v in kwargs.items()}
            )

    def _set_viewer_camera_pose(self, position, target):
        """ """
        disp = [x - y for x, y in zip(position, target)]
        dist = np.linalg.norm(disp)
        yaw = np.arctan2(disp[0], -disp[1])
        yaw = np.rad2deg(yaw)
        pitch = np.arctan2(-disp[2], np.linalg.norm((disp[0], disp[1])))
        pitch = np.rad2deg(pitch)

        self._p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    def step(self, bodies):
        """ """
        if self._cfg.RENDER:
            # Simulate real-time rendering with sleep if computation takes less than real time.
            time_spent = time.time() - self._last_frame_time
            time_sleep = self._cfg.TIME_STEP - time_spent
            if time_sleep > 0:
                time.sleep(time_sleep)
            self._last_frame_time = time.time()

        for body in reversed(self._bodies):
            if body.name not in [x.name for x in bodies]:
                # Remove body.
                with self._disable_cov_rendering():
                    self._p.removeBody(self._body_ids[body.name])
                    del self._body_ids[body.name]
                    self._bodies.remove(body)
        for body in bodies:
            if body.name not in [x.name for x in self._bodies]:
                # Add body.
                with self._disable_cov_rendering():
                    self._load_body(body)
                    self._cache_and_set_control(body)

        assert [body.name for body in bodies] == [
            body.name for body in self._bodies
        ], "Mismatched input and cached bodies"

        for b, body in enumerate(bodies):
            set_link_dynamics = False
            for attr in (
                "link_color",
                "link_collision_filter",
                "link_lateral_friction",
                "link_spinning_friction",
                "link_rolling_friction",
                "link_restitution",
                "link_linear_damping",
                "link_angular_damping",
            ):
                if not np.array_equal(getattr(body, attr), getattr(self._bodies[b], attr)):
                    if getattr(body, attr) is None:
                        raise ValueError(f"'{attr}' cannot be changed to None: '{body.name}'")
                    setattr(self._bodies[b], attr, getattr(body, attr).copy())
                    if attr == "link_color":
                        self._set_link_color(self._bodies[b])
                    if attr == "link_collision_filter":
                        self._set_link_collision_filter(self._bodies[b])
                    if (
                        attr
                        in (
                            "link_lateral_friction",
                            "link_spinning_friction",
                            "link_rolling_friction",
                            "link_restitution",
                            "link_linear_damping",
                            "link_angular_damping",
                        )
                        and not set_link_dynamics
                    ):
                        set_link_dynamics = True
            if set_link_dynamics:
                self._set_link_dynamics(self._bodies[b])

            if len(self._dof_indices[body.name]) == 0:
                for attr in (
                    "dof_control_mode",
                    "dof_max_force",
                    "dof_max_velocity",
                    "dof_position_gain",
                    "dof_velocity_gain",
                    "dof_target_position",
                    "dof_target_velocity",
                    "dof_force",
                ):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"Body '{body.name}' has 0 DoF but is given a non-empty '{attr}'"
                        )
                continue

            if not np.array_equal(body.dof_control_mode, self._bodies[b].dof_control_mode):
                raise ValueError(
                    "For Bullet, 'dof_control_mode' cannot be changed after each reset: "
                    f"'{body.name}'"
                )
            # The redundant if-else block below is an artifact due to `setJointMotorControlArray()`
            # not supporting `maxVelocity`. `setJointMotorControlArray()` is still preferred when
            # `maxVelocity` is not needed due to better speed performance.
            if body.dof_max_velocity is None:
                kwargs = {}
                if body.dof_target_position is not None:
                    kwargs["targetPositions"] = self._get_body_attr_array(
                        body, "dof_target_position", 1
                    )
                if body.dof_target_velocity is not None:
                    kwargs["targetVelocities"] = self._get_body_attr_array(
                        body, "dof_target_velocity", 1
                    )
                if body.dof_position_gain is not None:
                    kwargs["positionGains"] = self._get_body_attr_array(
                        body, "dof_position_gain", 1
                    )
                if body.dof_velocity_gain is not None:
                    kwargs["velocityGains"] = self._get_body_attr_array(
                        body, "dof_velocity_gain", 1
                    )
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
                        kwargs["forces"] = self._get_body_attr_array(body, "dof_max_force", 1)
                    if body.dof_force is not None:
                        if body.dof_control_mode != DoFControlMode.TORQUE_CONTROL:
                            raise ValueError(
                                "For Bullet, 'dof_force' can only be set in the TORQUE_CONTROL "
                                f"mode: '{body.name}'"
                            )
                        kwargs["forces"] = self._get_body_attr_array(body, "dof_force", 1)
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
                        kwargs["forces"] = self._get_body_attr_array(body, "dof_force", 1)
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
                        kwargs["forces"] = self._get_body_attr_array(body, "dof_force", 1)
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
                    kwargs["targetPosition"] = self._get_body_attr_array(
                        body, "dof_target_position", 1
                    )
                if body.dof_target_velocity is not None:
                    kwargs["targetVelocity"] = self._get_body_attr_array(
                        body, "dof_target_velocity", 1
                    )
                if body.dof_max_velocity is not None:
                    # For Bullet, 'dof_max_velocity' has no effect when not in the POSITION_CONROL
                    # mode.
                    kwargs["maxVelocity"] = self._get_body_attr_array(body, "dof_max_velocity", 1)
                if body.dof_position_gain is not None:
                    kwargs["positionGain"] = self._get_body_attr_array(body, "dof_position_gain", 1)
                if body.dof_velocity_gain is not None:
                    kwargs["velocityGain"] = self._get_body_attr_array(body, "dof_velocity_gain", 1)
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
                        kwargs["force"] = self._get_body_attr_array(body, "dof_max_force", 1)
                    if body.dof_force is not None:
                        if body.dof_control_mode != DoFControlMode.TORQUE_CONTROL:
                            raise ValueError(
                                "For Bullet, 'dof_force' can only be set in the TORQUE_CONTROL "
                                f"mode: '{body.name}'"
                            )
                        kwargs["force"] = self._get_body_attr_array(body, "dof_force", 1)
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
                                kwargs["force"] = self._get_body_attr_array(
                                    body, "dof_max_force", 1
                                )
                            if body.dof_force is not None and not np.isnan(
                                self._get_body_attr_array(body, "dof_force", 1)[i]
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
                                kwargs["force"] = self._get_body_attr_array(body, "dof_force", 1)
                            if body.dof_max_force is not None and not np.isnan(
                                self._get_body_attr_array(body, "dof_max_force", 1)[i]
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

        for body in bodies:
            if self._num_links[body.name] > 1:
                joint_states = self._p.getJointStates(
                    self._body_ids[body.name], self._dof_indices[body.name]
                )
                dof_state = [x[0:2] for x in joint_states]
                body.dof_state = [dof_state]

            pos, orn = self._p.getBasePositionAndOrientation(self._body_ids[body.name])
            lin, ang = self._p.getBaseVelocity(self._body_ids[body.name])
            link_state = [pos + orn + lin + ang]
            if self._num_links[body.name] > 1:
                link_indices = [*range(0, self._num_links[body.name] - 1)]
                link_states = self._p.getLinkStates(
                    self._body_ids[body.name], link_indices, computeLinkVelocity=1
                )
                link_state += [x[4] + x[5] + x[6] + x[7] for x in link_states]
            body.link_state = [link_state]

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
            self._p.disconnect()
            self._connected = False
