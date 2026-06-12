# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based MARL environment for Xie et al. formation swarm control.

Delegates observations, rewards, terminations, events, and curriculum to managers
while keeping formation-specific physical updates (ball dynamics, column sampling)
in the environment class.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from environments.envs import ManagerBasedMarlEnv

from .formation_marl_env_cfg import FormationSwarmMarlEnvCfg


def _laplacian(points: torch.Tensor, normalize: bool) -> torch.Tensor:
    r"""Compute the pairwise-distance graph Laplacian for formation points.

    For pairwise distance matrix :math:`D_{ij}=\lVert p_i-p_j\rVert_2` and degree
    :math:`d_i=\sum_j D_{ij}`, the unnormalized Laplacian is

    .. math::

        L = \operatorname{diag}(d) - D.

    With ``normalize=True``, the function returns

    .. math::

        L_\text{norm} = I - \operatorname{diag}(d)^{-1/2}D\operatorname{diag}(d)^{-1/2}.
    """
    distances = torch.cdist(points, points)
    degree = distances.sum(dim=-1)
    if normalize:
        scale = degree.clamp_min(1.0e-6).pow(-0.5)
        adj = scale.unsqueeze(-1) * distances * scale.unsqueeze(-2)
        eye = torch.eye(points.shape[-2], device=points.device, dtype=points.dtype)
        return eye.expand(points.shape[:-2] + eye.shape) - adj
    return degree.unsqueeze(-1) - distances


class FormationSwarmMarlEnv(ManagerBasedMarlEnv):
    """Manager-based formation flight with static columns and dynamic balls.

    Extends :class:`ManagerBasedMarlEnv` to add formation-specific physical
    updates (ball launching, ball physics, obstacle visuals) while delegating
    observations, rewards, terminations, events, and curriculum to managers.
    """

    cfg: FormationSwarmMarlEnvCfg

    def __init__(self, cfg: FormationSwarmMarlEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize the formation swarm MARL environment."""
        self._formation_curriculum_applied = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._formation_asset_names = tuple(
            self.ma_spec.agents[agent_id].asset_name for agent_id in self.possible_agents
        )

        formation = torch.tensor(self.cfg.formation, device=self.device, dtype=torch.float32)
        formation = formation * self.cfg.formation_size
        self._formation_offsets = formation
        self._formation_target_pos = torch.tensor(self.cfg.target_pos, device=self.device, dtype=torch.float32)
        self._formation_target_vel = torch.tensor(self.cfg.target_vel, device=self.device, dtype=torch.float32)
        self._formation_target_heading = torch.tensor(
            self.cfg.target_heading, device=self.device, dtype=torch.float32
        )
        max_steps = getattr(self, "max_episode_length", 450)
        self._formation_final_pos = (
            self._formation_target_pos + self._formation_target_vel * (max_steps * self.step_dt * 0.9)
        )
        self._formation_l = _laplacian(formation, normalize=True)
        self._formation_l_unnormalized = _laplacian(formation, normalize=False)
        self._standard_formation_size = torch.cdist(formation, formation).max()

        curriculum_stage = getattr(self.cfg, "curriculum_stage", 0)
        if curriculum_stage == 1:
            active_static = 0
            active_balls = 0
            throw_threshold = self.cfg.curriculum_delayed_throw_threshold_steps
            throw_range = self.cfg.curriculum_delayed_throw_time_range_steps
        elif curriculum_stage == 2:
            active_static = self.cfg.static_obstacles
            active_balls = 0
            throw_threshold = self.cfg.curriculum_delayed_throw_threshold_steps
            throw_range = self.cfg.curriculum_delayed_throw_time_range_steps
        elif curriculum_stage == 3:
            active_static = self.cfg.static_obstacles
            active_balls = self.cfg.num_balls
            throw_threshold = self.cfg.throw_threshold_steps
            throw_range = self.cfg.throw_time_range_steps
        else:
            active_static = getattr(self.cfg, "active_static_obstacles", None) or self.cfg.static_obstacles
            active_balls = getattr(self.cfg, "active_balls", None) or self.cfg.num_balls
            throw_threshold = self.cfg.throw_threshold_steps
            throw_range = self.cfg.throw_time_range_steps

        self._formation_active_static_obstacles = max(0, min(int(active_static), self.cfg.static_obstacles))
        self._formation_active_balls = max(0, min(int(active_balls), self.cfg.num_balls))
        self._formation_throw_threshold_steps = int(throw_threshold)
        self._formation_throw_time_range_steps = int(throw_range)

        self._formation_grid_offsets = torch.tensor(
            [
                [-0.1, -0.1], [-0.1, 0.0], [-0.1, 0.1],
                [0.0, -0.1],  [0.0, 0.0],  [0.0, 0.1],
                [0.1, -0.1],  [0.1, 0.0],  [0.1, 0.1],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        num_agents = len(self.possible_agents)
        self._formation_last_actions = torch.zeros(self.num_envs, num_agents, 4, device=self.device)
        self._formation_current_action_features = torch.zeros(
            self.num_envs, num_agents, 4, device=self.device
        )
        self._formation_previous_action_features = torch.zeros_like(
            self._formation_current_action_features
        )
        self._formation_previous_actions = self._formation_last_actions.clone()

        self._formation_column_positions = torch.zeros(
            self.num_envs, self.cfg.static_obstacles, 3, device=self.device
        )

        self._formation_ball_positions = torch.zeros(
            self.num_envs, self.cfg.num_balls, 3, device=self.device
        )
        self._formation_ball_velocities = torch.zeros_like(self._formation_ball_positions)
        self._formation_ball_start_pos = torch.zeros_like(self._formation_ball_positions)
        self._formation_ball_start_vel = torch.zeros_like(self._formation_ball_positions)
        self._formation_ball_launch_step = torch.zeros(
            self.num_envs, self.cfg.num_balls, device=self.device, dtype=torch.long
        )
        self._formation_ball_active = torch.zeros(
            self.num_envs, self.cfg.num_balls, device=self.device, dtype=torch.bool
        )
        self._formation_ball_launched = torch.zeros_like(self._formation_ball_active)

    # ---------------------------------------------------------------------
    # DirectMARLEnv hooks
    # ---------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Create ground plane, observer camera, and obstacle visual prims."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(80.0, 80.0)))

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)
        self._spawn_view_camera()
        if getattr(self.cfg, "spawn_obstacle_visuals", True):
            self._spawn_obstacle_visuals()
        if getattr(self.cfg, "video_drone_markers", False):
            self._spawn_video_drone_markers()

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Update ball dynamics and store action features for reward computation."""
        self._update_balls()

        for i, agent_id in enumerate(self.possible_agents):
            action = actions.get(agent_id)
            if action is not None:
                clean_action = torch.nan_to_num(action.to(self.device), nan=0.0, posinf=1.0, neginf=-1.0)
                self._formation_previous_action_features[:, i] = (
                    self._formation_current_action_features[:, i].clone()
                )
                self._formation_current_action_features[:, i] = clean_action.to(self.device)

        super()._pre_physics_step(actions)

        for i in range(len(self.possible_agents)):
            self._formation_previous_actions[:, i] = self._formation_last_actions[:, i].clone()
            self._formation_last_actions[:, i] = self._formation_current_action_features[:, i]

    def _reset_idx(self, env_ids) -> None:
        """Reset internal state for the given environment indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        self._formation_current_action_features[env_ids] = 0.0
        self._formation_previous_action_features[env_ids] = 0.0
        self._formation_last_actions[env_ids] = 0.0
        self._formation_previous_actions[env_ids] = 0.0
        self._sync_obstacle_visuals(env_ids)

    def render(self):
        """Render the environment, optionally following the first swarm for video demos."""
        if getattr(self.cfg, "video_drone_markers", False):
            self._sync_video_drone_markers()
        if self.render_mode == "rgb_array" and getattr(self.cfg, "video_follow_camera", False):
            self._update_video_follow_camera()
        return super().render()

    def _update_video_follow_camera(self) -> None:
        """Move the Kit perspective recording camera with the first environment's swarm center."""
        if self.num_envs <= 0 or getattr(self, "video_recorder", None) is None:
            return
        try:
            positions = torch.stack(
                [self.scene[asset_name].data.root_pos_w.torch[0] for asset_name in self._formation_asset_names],
                dim=0,
            )
        except Exception:
            return

        center = positions.mean(dim=0)
        eye_offset = torch.tensor(
            getattr(self.cfg, "video_follow_eye_offset", (3.0, -4.5, 2.2)),
            device=center.device,
            dtype=center.dtype,
        )
        lookahead = torch.tensor(
            getattr(self.cfg, "video_follow_lookahead", (0.0, 1.5, 0.0)),
            device=center.device,
            dtype=center.dtype,
        )
        eye = tuple(float(v) for v in (center + eye_offset).detach().cpu())
        lookat = tuple(float(v) for v in (center + lookahead).detach().cpu())

        self.video_recorder.cfg.eye = eye
        self.video_recorder.cfg.lookat = lookat

        capture = getattr(self.video_recorder, "_capture", None)
        if getattr(capture, "cfg", None) is not None:
            capture.cfg.eye = eye
            capture.cfg.lookat = lookat
        camera_prim_path = getattr(getattr(capture, "cfg", None), "camera_prim_path", "/OmniverseKit_Persp")
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            ViewportManager.set_camera_view(camera_prim_path, eye=list(eye), target=list(lookat))
        except Exception:
            pass

        focal_length = getattr(self.cfg, "video_focal_length", None)
        if focal_length is not None:
            try:
                import omni.usd
                from pxr import UsdGeom

                stage = omni.usd.get_context().get_stage()
                camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
                if camera:
                    camera.GetFocalLengthAttr().Set(float(focal_length))
            except Exception:
                return

    # ---------------------------------------------------------------------
    # Ball dynamics
    # ---------------------------------------------------------------------

    def _update_balls(self) -> None:
        """Launch and simulate ball positions using simple kinematic physics."""
        if self.cfg.num_balls == 0:
            return

        active_slots = torch.arange(self.cfg.num_balls, device=self.device)
        active_slots = active_slots < self._formation_active_balls
        should_launch = (
            self.episode_length_buf.unsqueeze(-1) >= self._formation_ball_launch_step
        ) & ~self._formation_ball_active
        should_launch = should_launch & active_slots.view(1, -1)

        if should_launch.any():
            positions = torch.stack(
                [
                    self.scene[asset_name].data.root_pos_w.torch - self.scene.env_origins
                    for asset_name in self._formation_asset_names
                ],
                dim=1,
            )
            center = positions.mean(dim=1)
            env_ids, ball_ids = should_launch.nonzero(as_tuple=True)

            if self.cfg.random_ball_speed:
                speed = (
                    torch.rand(len(env_ids), device=self.device)
                    * (self.cfg.max_ball_speed - self.cfg.min_ball_speed)
                    + self.cfg.min_ball_speed
                )
            else:
                speed = torch.full((len(env_ids),), float(self.cfg.ball_speed), device=self.device)

            direction_xy = torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0
            direction_xy = direction_xy / (
                torch.linalg.norm(direction_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
            )
            t_hit = torch.rand(len(env_ids), device=self.device) * 0.8 + 0.8
            start_z = (
                torch.rand(len(env_ids), device=self.device) * center[env_ids, 2]
                + 0.5 * center[env_ids, 2]
            )
            vel = torch.zeros(len(env_ids), 3, device=self.device)
            vel[:, :2] = direction_xy * speed.unsqueeze(-1)
            vel[:, 2] = (center[env_ids, 2] - start_z) / t_hit + 0.5 * 9.81 * t_hit
            start = center[env_ids].clone()
            start[:, :2] = center[env_ids, :2] - vel[:, :2] * t_hit.unsqueeze(-1)
            start[:, 2] = start_z

            self._formation_ball_start_pos[env_ids, ball_ids] = start
            self._formation_ball_start_vel[env_ids, ball_ids] = vel
            self._formation_ball_positions[env_ids, ball_ids] = start
            self._formation_ball_velocities[env_ids, ball_ids] = vel
            self._formation_ball_launch_step[env_ids, ball_ids] = self.episode_length_buf[env_ids]
            self._formation_ball_active[env_ids, ball_ids] = True
            self._formation_ball_launched[env_ids, ball_ids] = True

        active_envs, active_balls = self._formation_ball_active.nonzero(as_tuple=True)
        if len(active_envs) > 0:
            elapsed = (
                self.episode_length_buf[active_envs]
                - self._formation_ball_launch_step[active_envs, active_balls]
            ).float() * self.step_dt
            start = self._formation_ball_start_pos[active_envs, active_balls]
            vel_initial = self._formation_ball_start_vel[active_envs, active_balls]
            pos = start + vel_initial * elapsed.unsqueeze(-1)
            pos[:, 2] = start[:, 2] + vel_initial[:, 2] * elapsed - 0.5 * 9.81 * elapsed.square()
            self._formation_ball_positions[active_envs, active_balls] = pos
            self._formation_ball_velocities[active_envs, active_balls] = vel_initial.clone()
            self._formation_ball_velocities[active_envs, active_balls, 2] = (
                vel_initial[:, 2] - 9.81 * elapsed
            )
            landed = pos[:, 2] < 0.2
            self._formation_ball_active[active_envs[landed], active_balls[landed]] = False

        if getattr(self.cfg, "spawn_obstacle_visuals", True):
            self._sync_obstacle_visuals(torch.arange(self.num_envs, device=self.device))

    # ---------------------------------------------------------------------
    # Camera and obstacle visuals
    # ---------------------------------------------------------------------

    def _spawn_view_camera(self) -> None:
        """Create a custom observer camera if the viewer path is set."""
        camera_path = self.cfg.viewer.cam_prim_path
        if (
            not camera_path
            or camera_path == "/OmniverseKit_Persp"
            or self.sim.stage.GetPrimAtPath(camera_path).IsValid()
        ):
            return
        eye = torch.tensor([self.cfg.viewer.eye], dtype=torch.float32)
        target = torch.tensor([self.cfg.viewer.lookat], dtype=torch.float32)
        orientation = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eye, target, up_axis="Z")
        )[0]
        camera_cfg = sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=20.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        )
        camera_cfg.func(
            camera_path, camera_cfg,
            translation=tuple(float(v) for v in eye[0]),
            orientation=tuple(float(v) for v in orientation),
        )

    def _spawn_obstacle_visuals(self) -> None:
        """Create prims for columns and balls under the first environment."""
        self._formation_column_prims = []
        self._formation_ball_prims = []
        env_index = 0
        column_cfg = sim_utils.CylinderCfg(
            radius=self.cfg.column_radius,
            height=getattr(self.cfg, "static_height", 5.0),
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.8)),
        )
        ball_cfg = sim_utils.SphereCfg(
            radius=self.cfg.ball_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
        for i in range(self.cfg.static_obstacles):
            prim_path = f"/World/envs/env_{env_index}/columns/column_{i:02d}"
            prim = column_cfg.func(prim_path, column_cfg, translation=(0.0, 0.0, -10.0))
            self._formation_column_prims.append(prim)
        for i in range(self.cfg.num_balls):
            prim_path = f"/World/envs/env_{env_index}/balls/ball_{i:02d}"
            prim = ball_cfg.func(prim_path, ball_cfg, translation=(0.0, 0.0, -10.0))
            self._formation_ball_prims.append(prim)

    def _spawn_video_drone_markers(self) -> None:
        """Create video-only colored markers that make each drone legible in thesis captures."""
        self._formation_video_marker_prims = []
        colors = getattr(
            self.cfg,
            "video_marker_colors",
            ((0.1, 1.0, 0.15), (1.0, 0.85, 0.05), (0.15, 0.55, 1.0)),
        )
        radius = float(getattr(self.cfg, "video_marker_radius", 0.08))
        for i in range(int(getattr(self.cfg, "num_drones", len(self.possible_agents)))):
            color = colors[i % len(colors)]
            marker_cfg = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in color)),
            )
            prim_path = f"/World/envs/env_0/video_markers/drone_{i:02d}"
            prim = marker_cfg.func(prim_path, marker_cfg, translation=(0.0, 0.0, -10.0))
            self._formation_video_marker_prims.append(prim)

    def _sync_video_drone_markers(self) -> None:
        """Move video-only markers to the env-0 drone positions."""
        markers = getattr(self, "_formation_video_marker_prims", None)
        if not markers:
            return
        offset = torch.tensor(
            getattr(self.cfg, "video_marker_offset", (0.0, 0.0, 0.08)),
            device=self.device,
            dtype=torch.float32,
        )
        for i, prim in enumerate(markers):
            if i >= len(self._formation_asset_names):
                continue
            position = self.scene[self._formation_asset_names[i]].data.root_pos_w.torch[0] + offset
            sim_utils.standardize_xform_ops(prim, translation=tuple(float(v) for v in position.detach().cpu()))

    def _sync_obstacle_visuals(self, env_ids: torch.Tensor) -> None:
        """Move and show/hide column and ball prims to match internal state."""
        columns = getattr(self, "_formation_column_prims", None)
        balls = getattr(self, "_formation_ball_prims", None)
        if not columns and not balls:
            return
        if not (env_ids == 0).any():
            return

        origin = self.scene.env_origins[0].detach().cpu()
        static_height = getattr(self.cfg, "static_height", 5.0)

        for i, prim in enumerate(self._formation_column_prims):
            if i < self._formation_active_static_obstacles:
                position = self._formation_column_positions[0, i].detach().cpu() + origin
                position[2] = static_height / 2.0
            else:
                position = torch.tensor((0.0, 0.0, -10.0))
            sim_utils.standardize_xform_ops(prim, translation=tuple(float(v) for v in position))

        for i, prim in enumerate(self._formation_ball_prims):
            position = self._formation_ball_positions[0, i].detach().cpu() + origin
            sim_utils.set_prim_visibility(prim, bool(self._formation_ball_active[0, i]))
            sim_utils.standardize_xform_ops(prim, translation=tuple(float(v) for v in position))

    # ---------------------------------------------------------------------
    # State helpers (called by event terms)
    # ---------------------------------------------------------------------

    def _write_swarm_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        orientations: torch.Tensor | None,
        linear_velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
    ) -> None:
        """Write root pose and velocity to all drone articulations in the swarm.

        Args:
            env_ids: Environment indices to write.
            positions: Target positions [m],
                shape ``(len(env_ids), num_agents, 3)``.
            orientations: Target orientations as quaternions [x, y, z, w],
                shape ``(len(env_ids), num_agents, 4)``, or ``None``.
            linear_velocities: Target linear velocities [m/s],
                shape ``(len(env_ids), num_agents, 3)``.
            angular_velocities: Target angular velocities [rad/s],
                shape ``(len(env_ids), num_agents, 3)``.
        """
        for drone_index, asset_name in enumerate(self._formation_asset_names):
            drone = self.scene[asset_name]
            root_pose = drone.data.default_root_pose.torch[env_ids].clone()
            root_vel = drone.data.default_root_vel.torch[env_ids].clone()
            joint_pos = drone.data.default_joint_pos.torch[env_ids].clone()
            joint_vel = drone.data.default_joint_vel.torch[env_ids].clone()

            root_pose[:, :3] = positions[:, drone_index] + self.scene.env_origins[env_ids]
            if orientations is not None:
                root_pose[:, 3:7] = orientations[:, drone_index]
            root_vel[:, :3] = linear_velocities[:, drone_index]
            root_vel[:, 3:6] = angular_velocities[:, drone_index]

            drone.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            drone.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
            drone.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            drone.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _set_hover_actions(self, env_ids: torch.Tensor) -> None:
        """Apply hover thrust directly to drone motor actuators on reset."""
        if len(env_ids) == 0:
            return

        hover_collective = getattr(self, "_hover_collective", None)
        if hover_collective is None:
            from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import (
                CRAZYFLIE_CFG as _CFG,
            )
            from cpsquare_lab.embodiments.multirotor.common.ctbr import (
                hover_collective_thrust_from_multirotor_cfg,
            )
            self._hover_collective = hover_collective_thrust_from_multirotor_cfg(_CFG)

        for asset_name in self._formation_asset_names:
            asset = self.scene[asset_name]
            thruster_actuator = asset.actuators.get("thrusters")
            if thruster_actuator is None:
                continue
            num_thrusters = thruster_actuator.num_thrusters
            motor_thrust = torch.full(
                (len(env_ids), num_thrusters),
                self._hover_collective / num_thrusters,
                device=self.device,
            )
            asset.set_thrust_target(motor_thrust, env_ids=env_ids)
