# Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import os
import pkgutil
import site
import re
import sys
import carb
import torch
import time

from contextlib import contextmanager
from omni.isaac.kit import SimulationApp

from easysim.simulators.simulator import Simulator
from easysim.constants import DescriptionType


class IsaacSim(Simulator):
    """Isaac Sim simulator."""

    _DEFAULT_BASE_ENV_PATH = "/World/envs"
    _DEFAULT_ZERO_ENV_NAME = "env"
    _DEFAULT_ZERO_ENV_PATH = f"{_DEFAULT_BASE_ENV_PATH}/{_DEFAULT_ZERO_ENV_NAME}_0"
    _COLLISION_ROOT_PATH = "/World/collisions"
    _DEFAULT_DISTANT_LIGHT_PATH = "/World/defaultDistantLight"
    _DEFAULT_DISTANT_LIGHT_INTENSITY = 5000

    def _init(self):
        """ """
        if self._cfg.SUBSTEPS != 1:
            raise ValueError("SUBSTEPS must be 1 for Isaac Sim")

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

        self._sim_params = self._parse_sim_params(sim_device_type)

        self._num_envs = self._cfg.NUM_ENVS

        self._created = False
        self._last_frame_time = 0.0

    @property
    def device(self):
        """ """
        return self._device

    @property
    def graphics_device(self):
        """ """
        raise NotImplementedError

    def _parse_sim_params(self, sim_device_type):
        """ """
        sim_params = {}

        if not self._cfg.USE_DEFAULT_STEP_PARAMS:
            sim_params["dt"] = self._cfg.TIME_STEP

        sim_params["gravity"] = self._cfg.GRAVITY
        sim_params["use_gpu_pipeline"] = self._cfg.USE_GPU_PIPELINE

        if self._cfg.RENDER:
            sim_params["use_flatcache"] = True
        else:
            sim_params["use_flatcache"] = False

        sim_params["use_gpu"] = sim_device_type == "cuda"

        return sim_params

    def reset(self, env_ids):
        """ """
        if not self._created:
            self._create_simulation_app()
            self._import_isaac_sim_modules()
            self._create_world()
            self._set_cloner()

            if self._cfg.LOAD_GROUND_PLANE:
                self._load_ground_plane()
            self._load_bodies()
            self._set_viewer()
            self._set_light()

            self._world.reset()
            self._create_views()

            self._scene_cache = type(self._scene)()
            self._cache_body_and_set_props()
            self._set_body_callback()

            self._created = True

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self.device)

        self._reset_idx(env_ids)

        self._clear_state()

    def _create_simulation_app(self):
        """ """
        launch_config = {}
        launch_config["headless"] = not self._cfg.RENDER
        launch_config["physics_gpu"] = self._sim_device_id

        experience = ""
        if not self._cfg.RENDER:
            experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        with self._disable_modules_from_sys_path():
            self._simulation_app = SimulationApp(launch_config=launch_config, experience=experience)

        if self._cfg.RENDER:
            carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

    @contextmanager
    def _disable_modules_from_sys_path(self):
        """ """
        try:
            modules = self._cfg.ISAAC_SIM.MODULES_DISABLED_AT_LAUNCH
            packages = []
            for module in modules:
                package = pkgutil.get_loader(module)
                if package:
                    packages.append(package)
            if packages:
                paths = []
                for x in site.getsitepackages():
                    if any(re.match(f"{x}", package.get_filename()) for package in packages):
                        paths.append(x)
                indices = []
                for x in paths:
                    indices.append(sys.path.index(x))
                indices, paths = zip(*sorted(zip(indices, paths)))
                for i in indices[::-1]:
                    sys.path.pop(i)
            yield
        finally:
            if packages:
                for i, x in zip(indices, paths):
                    sys.path.insert(i, x)

    def _import_isaac_sim_modules(self):
        """ """
        global GridCloner
        global define_prim
        global add_reference_to_stage, get_current_stage
        global World
        global Gf, UsdLux

        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.utils.prims import define_prim
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
        from omni.isaac.core.world import World
        from pxr import Gf, UsdLux

        if self._cfg.RENDER:
            global get_viewport_from_window_name
            global ViewportCameraState

            from omni.kit.viewport.utility import get_viewport_from_window_name
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

    def _create_world(self):
        """ """
        rendering_dt = 1.0 / self._cfg.ISAAC_SIM.VIEWER.RENDER_FRAME_RATE

        if self._cfg.USE_GPU_PIPELINE:
            device = "cuda"
        else:
            device = "cpu"

        self._world = World(
            rendering_dt=rendering_dt, sim_params=self._sim_params, backend="torch", device=device
        )

        if self._cfg.USE_DEFAULT_STEP_PARAMS:
            self._cfg.TIME_STEP = self._world.get_physics_dt()

    def _set_cloner(self):
        """ """
        self._cloner = GridCloner(self._cfg.ISAAC_SIM.SPACING)
        self._cloner.define_base_env(self._DEFAULT_BASE_ENV_PATH)
        define_prim(self._DEFAULT_ZERO_ENV_PATH)

        self._cloner_prim_paths = self._cloner.generate_paths(
            f"{self._DEFAULT_BASE_ENV_PATH}/{self._DEFAULT_ZERO_ENV_NAME}", self._num_envs
        )
        self._env_pos = self._cloner.clone(
            self._DEFAULT_ZERO_ENV_PATH, self._cloner_prim_paths, replicate_physics=True
        )
        self._env_pos = torch.tensor(self._env_pos, dtype=torch.float32, device=self.device)

        self._cloner.filter_collisions(
            self._world.get_physics_context().prim_path,
            self._COLLISION_ROOT_PATH,
            self._cloner_prim_paths,
        )

    def _load_ground_plane(self):
        """ """
        self._world.scene.add_default_ground_plane()

    def _load_bodies(self):
        """ """
        for body in self._scene.bodies:
            if body.description_type is None:
                raise ValueError(f"'description_type' must not be None: '{body.name}'")
            if body.description_type not in (DescriptionType.USD,):
                raise ValueError(
                    f"For Isaac Sim, 'description_type' only supports USD: '{body.name}'"
                )
            add_reference_to_stage(body.usd.path, f"{self._DEFAULT_ZERO_ENV_PATH}/{body.name}")

    def _set_viewer(self):
        """ """
        if self._cfg.RENDER:
            if bool(self._cfg.VIEWER.INIT_CAMERA_POSITION == (None, None, None)) != bool(
                self._cfg.VIEWER.INIT_CAMERA_TARGET == (None, None, None)
            ):
                raise ValueError(
                    "INIT_CAMERA_POSITION and INIT_CAMERA_TARGET need to be set together in "
                    f"order to take effect: {self._cfg.VIEWER.INIT_CAMERA_POSITION}, "
                    f"{self._cfg.VIEWER.INIT_CAMERA_TARGET}"
                )
            elif self._cfg.VIEWER.INIT_CAMERA_POSITION != (
                None,
                None,
                None,
            ) and self._cfg.VIEWER.INIT_CAMERA_TARGET != (None, None, None):
                viewport_api = get_viewport_from_window_name(window_name="Viewport")
                viewport_api.set_active_camera("/OmniverseKit_Persp")
                camera_state = ViewportCameraState(
                    camera_path="/OmniverseKit_Persp", viewport=viewport_api
                )
                camera_state.set_position_world(
                    Gf.Vec3d(*self._cfg.VIEWER.INIT_CAMERA_POSITION), True
                )
                camera_state.set_target_world(Gf.Vec3d(*self._cfg.VIEWER.INIT_CAMERA_TARGET), True)

    def _set_light(self):
        """ """
        if self._cfg.ISAAC_SIM.ADD_DISTANT_LIGHT:
            stage = get_current_stage()
            light = UsdLux.DistantLight.Define(stage, self._DEFAULT_DISTANT_LIGHT_PATH)
            light.GetPrim().GetAttribute("intensity").Set(self._DEFAULT_DISTANT_LIGHT_INTENSITY)

    def _create_views(self):
        """ """
        self._articulation_views = {}
        self._num_dofs = {}

        self._initial_base_position = {}
        self._initial_base_velocity = {}
        self._initial_dof_position = {}
        self._initial_dof_velocity = {}

        for body in self._scene.bodies:
            self._articulation_views[
                body.name
            ] = self._world.physics_sim_view.create_articulation_view(
                f"{self._DEFAULT_BASE_ENV_PATH}/*/{body.name}"
            )
            self._num_dofs[body.name] = self._articulation_views[body.name].max_dofs

            self._initial_base_position[body.name] = (
                self._articulation_views[body.name].get_root_transforms().clone()
            )
            self._initial_base_velocity[body.name] = (
                self._articulation_views[body.name].get_root_velocities().clone()
            )
            self._initial_dof_position[body.name] = (
                self._articulation_views[body.name].get_dof_positions().clone()
            )
            self._initial_dof_velocity[body.name] = (
                self._articulation_views[body.name].get_dof_velocities().clone()
            )

    def _cache_body_and_set_props(self):
        """ """
        indices = torch.arange(self._num_envs, device="cpu")

        for body in self._scene.bodies:
            x = type(body)()
            x.name = body.name
            self._scene_cache.add_body(x)

            for attr in ("dof_control_mode",):
                if getattr(body, attr) is not None:
                    raise ValueError(f"'{attr}' is not supported in Isaac Sim: '{body.name}'")

            if self._num_dofs[body.name] == 0:
                for attr in (
                    "dof_has_limits",
                    "dof_lower_limit",
                    "dof_upper_limit",
                    "dof_control_mode",
                    "dof_max_force",
                    "dof_max_velocity",
                    "dof_position_gain",
                    "dof_velocity_gain",
                    "dof_armature",
                ):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )

            if body.dof_max_force is not None:
                data = torch.from_numpy(body.dof_max_force).expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_max_forces(data, indices)

            if body.dof_max_velocity is not None:
                data = torch.from_numpy(body.dof_max_velocity).expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_max_velocities(data, indices)

            if body.dof_position_gain is not None:
                data = torch.from_numpy(body.dof_position_gain).expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_stiffnesses(data, indices)

            if body.dof_velocity_gain is not None:
                data = torch.from_numpy(body.dof_velocity_gain).expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_dampings(data, indices)

    def _set_body_callback(self):
        """ """
        for body in self._scene.bodies:
            body.set_callback_collect_dof_state(self._collect_dof_state)
            body.set_callback_collect_link_state(self._collect_link_state)

    def _collect_dof_state(self, body):
        """ """
        if self._num_dofs[body.name] > 0:
            body.dof_state = torch.stack(
                (
                    self._articulation_views[body.name].get_dof_positions(),
                    self._articulation_views[body.name].get_dof_velocities(),
                ),
                dim=2,
            )

    def _collect_link_state(self, body):
        """ """
        link_state = torch.cat(
            (
                self._articulation_views[body.name].get_root_transforms(),
                self._articulation_views[body.name].get_root_velocities(),
            ),
            dim=1,
        )
        link_state[:, 0:3] -= self._env_pos
        body.link_state = link_state

    def _reset_idx(self, env_ids):
        """ """
        if [body.name for body in self._scene.bodies] != [
            body.name for body in self._scene_cache.bodies
        ]:
            raise ValueError(
                "For Isaac Sim, the list of bodies cannot be altered after the first reset"
            )

        for body in self._scene.bodies:
            self._reset_base_state(body, env_ids)

            if self._num_dofs[body.name] == 0:
                for attr in ("initial_dof_position", "initial_dof_velocity"):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )
            else:
                self._reset_dof_state(body, env_ids)

    def _reset_base_state(self, body, env_ids):
        """ """
        if body.initial_base_position is None:
            data = self._initial_base_position[body.name]
        else:
            data = body.initial_base_position.expand((self._num_envs, -1)).contiguous()
            data[env_ids, 0:3] += self._env_pos[env_ids]
        self._articulation_views[body.name].set_root_transforms(data, env_ids)

        if body.initial_base_velocity is None:
            data = self._initial_base_velocity[body.name]
        else:
            data = body.initial_base_velocity.expand((self._num_envs, -1))
        self._articulation_views[body.name].set_root_velocities(data, env_ids)

    def _reset_dof_state(self, body, env_ids):
        """ """
        if body.initial_dof_position is None:
            data = self._initial_dof_position[body.name]
        else:
            data = body.initial_dof_position.expand((self._num_envs, -1))
        self._articulation_views[body.name].set_dof_positions(data, env_ids)

        if body.initial_dof_velocity is None:
            data = self._initial_dof_velocity[body.name]
        else:
            data = body.initial_dof_velocity.expand((self._num_envs, -1))
        self._articulation_views[body.name].set_dof_velocities(data, env_ids)

    def _clear_state(self):
        """ """
        for body in self._scene.bodies:
            body.dof_state = None
            body.link_state = None

    def step(self):
        """ """
        if self._cfg.RENDER and self._cfg.SIMULATE_REAL_TIME_RENDER:
            # Simulate real-time rendering with sleep if computation takes less than real time.
            time_spent = time.time() - self._last_frame_time
            time_sleep = self._cfg.TIME_STEP - time_spent
            if time_sleep > 0:
                time.sleep(time_sleep)
            self._last_frame_time = time.time()

        if [body.name for body in self._scene.bodies] != [
            body.name for body in self._scene_cache.bodies
        ]:
            raise ValueError(
                "For Isaac Sim, the list of bodies cannot be altered after the first reset"
            )

        for body in self._scene.bodies:
            if self._num_dofs[body.name] == 0:
                for attr in (
                    "dof_target_position",
                    "dof_target_velocity",
                    "dof_actuation_force",
                    "env_ids_reset_dof_state",
                ):
                    if getattr(body, attr) is not None:
                        raise ValueError(
                            f"'{attr}' must be None for body with 0 DoF: '{body.name}'"
                        )
                continue

            indices = torch.arange(self._num_envs, device=self.device)
            if body.dof_target_position is not None:
                data = body.dof_target_position.expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_position_targets(data, indices)
            if body.dof_target_velocity is not None:
                data = body.dof_target_velocity.expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_velocity_targets(data, indices)
            if body.dof_actuation_force is not None:
                data = body.dof_actuation_force.expand((self._num_envs, -1))
                self._articulation_views[body.name].set_dof_actuation_forces(data, indices)

        self._world.step(render=self._cfg.RENDER)

        self._clear_state()

    @property
    def contact(self):
        """ """
        raise NotImplementedError

    def close(self):
        """ """
        if self._created:
            self._simulation_app.close()
