"""Thin Isaac Lab DirectMARLEnv composition for the quad swarm paper task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp
from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_PARAMS
from cpsquare_lab.embodiments.multirotor.common.action_mapping import normalized_rotor_actions_to_thrust
from cpsquare_lab.tasks.common.metrics import (
    flush_episode_metrics,
    initialize_episode_metrics,
    record_metric,
    reset_episode_metrics,
)
from cpsquare_lab.tasks.swarm.collision_replay import CollisionReplayManager, SwarmReplaySnapshot
from cpsquare_lab.tasks.swarm.events import (
    floor_crash_events,
    robot_obstacle_collision_events,
    robot_robot_collision_events,
)
from cpsquare_lab.tasks.swarm.observations import paper_swarm_observation
from cpsquare_lab.tasks.swarm.rewards import (
    angular_velocity_penalty,
    binary_event_penalty,
    close_proximity_penalty,
    control_effort_penalty,
    inverse_squared_goal_reward,
    tilt_penalty,
)

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from . import paper_spec as spec
from .env_cfg import QuadSwarmPaperEnvCfg
from .obstacle_room import sample_obstacle_field, sample_start_goal_pairs

METRIC_NAMES = (
    "success",
    "collision",
    "final_distance",
    "robot_robot_collisions",
    "robot_obstacle_collisions",
    "replay_reset",
    "timeout",
    "crash",
)


class QuadSwarmPaperEnv(DirectMARLEnv):
    """Paper-style decentralized quadrotor swarm obstacle-navigation task."""

    cfg: QuadSwarmPaperEnvCfg

    def __init__(self, cfg: QuadSwarmPaperEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self._agent_ids = list(cfg.possible_agents)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._max_thrusts = torch.tensor(CRAZYFLIE_PARAMS.max_thrusts, device=self.device, dtype=torch.float32)
        self._hover_thrust = torch.full(
            (self.num_envs, len(CRAZYFLIE_PARAMS.max_thrusts)),
            CRAZYFLIE_PARAMS.hover_thrust,
            device=self.device,
            dtype=torch.float32,
        )
        self._thrust_targets = {agent: self._hover_thrust.clone() for agent in self._agent_ids}
        self._last_actions = torch.zeros(
            self.num_envs, self.cfg.num_drones, spec.ACTION_SIZE, device=self.device, dtype=torch.float32
        )
        self._goals = torch.zeros(self.num_envs, self.cfg.num_drones, 3, device=self.device, dtype=torch.float32)
        self._obstacle_positions = torch.zeros(
            self.num_envs,
            spec.OBSTACLE_GRID_SHAPE[0] * spec.OBSTACLE_GRID_SHAPE[1],
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._obstacle_mask = torch.zeros(self._obstacle_positions.shape[:2], device=self.device, dtype=torch.bool)
        self._previous_robot_collisions = torch.zeros(
            self.num_envs, self.cfg.num_drones, self.cfg.num_drones, device=self.device, dtype=torch.bool
        )
        self._previous_obstacle_collisions = torch.zeros(
            self.num_envs, self.cfg.num_drones, self._obstacle_positions.shape[1], device=self.device, dtype=torch.bool
        )
        self._last_robot_collision_events = torch.zeros(
            self.num_envs, self.cfg.num_drones, device=self.device, dtype=torch.bool
        )
        self._last_obstacle_collision_events = torch.zeros_like(self._last_robot_collision_events)
        self._last_floor_events = torch.zeros_like(self._last_robot_collision_events)
        self._last_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        replay_steps = max(1, int(round(float(self.cfg.replay_lag_s) / self.step_dt)))
        self._replay = CollisionReplayManager(
            num_envs=self.num_envs,
            history_steps=replay_steps,
            replay_probability=self.cfg.replay_probability,
        )
        initialize_episode_metrics(self, METRIC_NAMES)

    def _setup_scene(self) -> None:
        self.drones = {}
        for agent in self._agent_ids:
            robot_cfg = self.cfg.robot_cfg.replace(prim_path=f"/World/envs/env_.*/{agent}")
            self.drones[agent] = robot_cfg.class_type(robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(100.0, 100.0)))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        for agent, drone in self.drones.items():
            self.scene.articulations[agent] = drone

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)
        self._spawn_view_camera()
        self._spawn_obstacle_visuals()

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for drone_index, agent in enumerate(self._agent_ids):
            action = torch.nan_to_num(actions[agent].to(self.device), nan=0.0, posinf=1.0, neginf=-1.0).clamp(
                -1.0, 1.0
            )
            self._last_actions[:, drone_index] = action
            self._thrust_targets[agent] = normalized_rotor_actions_to_thrust(action, self._max_thrusts)

    def _apply_action(self) -> None:
        for agent, drone in self.drones.items():
            drone.set_thrust_target(self._thrust_targets[agent])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self._collect_swarm_state()
        obstacle_positions = self._obstacle_positions
        obstacle_mask = self._obstacle_mask if self.cfg.enable_obstacles else torch.zeros_like(self._obstacle_mask)
        observations, self._last_neighbor_indices = paper_swarm_observation(
            positions=state["positions"],
            goals=self._goals,
            linear_velocities=state["linear_velocities"],
            rotation_matrices=state["rotation_matrices"],
            angular_velocities=state["angular_velocities"],
            obstacle_positions=obstacle_positions,
            obstacle_mask=obstacle_mask,
            k_neighbors=self.cfg.visible_neighbors,
            obstacle_radius=self.cfg.obstacle_radius,
            sdf_resolution=self.cfg.local_sdf_resolution,
        )
        return {agent: observations[:, index] for index, agent in enumerate(self._agent_ids)}

    def _get_states(self) -> torch.Tensor | None:
        return None

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        state = getattr(self, "_last_state", None)
        if state is None:
            state = self._collect_swarm_state()

        goal_reward = self.cfg.goal_reward_scale * inverse_squared_goal_reward(
            state["positions"],
            self._goals,
            distance_scale=self.cfg.goal_reward_distance_scale,
        )
        reward = goal_reward
        reward = reward + close_proximity_penalty(
            state["positions"],
            safety_radius=self.cfg.robot_proximity_radius,
            collision_radius=self.cfg.robot_collision_radius,
            penalty=self.cfg.proximity_penalty,
        )
        reward = reward + binary_event_penalty(
            self._last_robot_collision_events,
            penalty=self.cfg.robot_collision_penalty,
        )
        reward = reward + binary_event_penalty(
            self._last_obstacle_collision_events,
            penalty=self.cfg.obstacle_collision_penalty,
        )
        reward = reward + binary_event_penalty(self._last_floor_events, penalty=self.cfg.floor_crash_penalty)
        reward = reward + angular_velocity_penalty(
            state["angular_velocities"],
            scale=self.cfg.angular_velocity_penalty_scale,
        )
        reward = reward + control_effort_penalty(self._last_actions, scale=self.cfg.control_effort_penalty_scale)
        reward = reward + tilt_penalty(state["rotation_matrices"], scale=self.cfg.tilt_penalty_scale)

        record_metric(self, "success", self._last_success.float())
        collision = self._last_robot_collision_events.any(dim=1) | self._last_obstacle_collision_events.any(dim=1)
        record_metric(self, "collision", collision.float())
        final_distance = torch.linalg.norm(self._goals - state["positions"], dim=-1).mean(dim=1)
        record_metric(self, "final_distance", final_distance)
        record_metric(self, "crash", self._last_floor_events.any(dim=1).float())

        return {agent: reward[:, index] for index, agent in enumerate(self._agent_ids)}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        state = self._collect_swarm_state()
        self._last_state = state

        robot_events, active_robot_collisions, robot_pair_counts = robot_robot_collision_events(
            state["positions"],
            collision_radius=self.cfg.robot_collision_radius,
            previous_collision_matrix=self._previous_robot_collisions,
        )
        obstacle_events, active_obstacle_collisions, obstacle_counts = robot_obstacle_collision_events(
            state["positions"],
            self._obstacle_positions,
            self._obstacle_mask if self.cfg.enable_obstacles else torch.zeros_like(self._obstacle_mask),
            obstacle_radius=self.cfg.obstacle_radius,
            previous_collision_matrix=self._previous_obstacle_collisions,
        )
        floor_events = floor_crash_events(state["positions"], minimum_height=self.cfg.floor_crash_height)
        goal_distances = torch.linalg.norm(self._goals - state["positions"], dim=-1)
        success = torch.all(goal_distances <= self.cfg.goal_reached_radius, dim=1)

        self._last_robot_collision_events = robot_events
        self._last_obstacle_collision_events = obstacle_events
        self._last_floor_events = floor_events
        self._last_success = success
        self._previous_robot_collisions = active_robot_collisions
        self._previous_obstacle_collisions = active_obstacle_collisions

        collision_envs = robot_events.any(dim=1) | obstacle_events.any(dim=1)
        if self.cfg.enable_replay and collision_envs.any():
            self._replay.record_collisions(collision_envs.nonzero(as_tuple=False).squeeze(-1))
        if self.cfg.enable_replay:
            self._replay.push_state(self._make_snapshot(state))

        record_metric(self, "robot_robot_collisions", robot_pair_counts)
        record_metric(self, "robot_obstacle_collisions", obstacle_counts)

        crashed = floor_events.any(dim=1)
        terminated_env = collision_envs | crashed | success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        record_metric(self, "timeout", time_out.float())

        terminated = {agent: terminated_env for agent in self._agent_ids}
        time_outs = {agent: time_out for agent in self._agent_ids}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        flush_episode_metrics(self, env_ids_tensor)
        super()._reset_idx(env_ids_tensor)

        replay_samples = self._replay.sample(env_ids_tensor) if self.cfg.enable_replay else {}
        fresh_env_ids = [
            env_id for env_id in env_ids_tensor.detach().cpu().tolist() if int(env_id) not in replay_samples
        ]
        if fresh_env_ids:
            fresh_tensor = torch.tensor(fresh_env_ids, device=self.device, dtype=torch.long)
            self._reset_fresh(fresh_tensor)
        for env_id, snapshot in replay_samples.items():
            self._restore_snapshot(env_id, snapshot)

        if replay_samples:
            replay_metric = torch.zeros(self.num_envs, device=self.device)
            replay_metric[list(replay_samples.keys())] = 1.0
            record_metric(self, "replay_reset", replay_metric)

        self._previous_robot_collisions[env_ids_tensor] = False
        self._previous_obstacle_collisions[env_ids_tensor] = False
        self._last_actions[env_ids_tensor] = 0.0
        self._replay.reset_history(env_ids_tensor)
        reset_episode_metrics(self, env_ids_tensor)
        self._last_state = None
        self._sync_obstacle_visuals(env_ids_tensor)

    def _reset_fresh(self, env_ids: torch.Tensor) -> None:
        positions, goals, orientations = sample_start_goal_pairs(
            len(env_ids),
            self.cfg.num_drones,
            room_size=self.cfg.room_size,
            device=self.device,
        )
        obstacle_positions, obstacle_mask = sample_obstacle_field(
            len(env_ids),
            density=self.cfg.obstacle_density if self.cfg.enable_obstacles else 0.0,
            device=self.device,
        )
        self._goals[env_ids] = goals
        self._obstacle_positions[env_ids] = obstacle_positions
        self._obstacle_mask[env_ids] = obstacle_mask
        self._write_swarm_state(
            env_ids,
            positions=positions,
            orientations=orientations,
            linear_velocities=torch.zeros_like(positions),
            angular_velocities=torch.zeros_like(positions),
        )

    def _restore_snapshot(self, env_id: int, snapshot: SwarmReplaySnapshot) -> None:
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self._goals[env_ids] = snapshot.goals.to(self.device)
        self._obstacle_positions[env_ids] = snapshot.obstacle_positions.to(self.device)
        self._obstacle_mask[env_ids] = snapshot.obstacle_mask.to(self.device)
        self.episode_length_buf[env_ids] = snapshot.episode_lengths.to(self.device)
        self._write_swarm_state(
            env_ids,
            positions=snapshot.positions.to(self.device),
            orientations=snapshot.orientations.to(self.device),
            linear_velocities=snapshot.linear_velocities.to(self.device),
            angular_velocities=snapshot.angular_velocities.to(self.device),
        )

    def _write_swarm_state(
        self,
        env_ids: torch.Tensor,
        *,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        linear_velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
    ) -> None:
        for drone_index, agent in enumerate(self._agent_ids):
            drone = self.drones[agent]
            root_pose = wp.to_torch(drone.data.default_root_pose)[env_ids].clone()
            root_vel = wp.to_torch(drone.data.default_root_vel)[env_ids].clone()
            joint_pos = wp.to_torch(drone.data.default_joint_pos)[env_ids].clone()
            joint_vel = wp.to_torch(drone.data.default_joint_vel)[env_ids].clone()

            root_pose[:, :3] = positions[:, drone_index] + self.scene.env_origins[env_ids]
            root_pose[:, 3:7] = orientations[:, drone_index]
            root_vel[:, :3] = linear_velocities[:, drone_index]
            root_vel[:, 3:6] = angular_velocities[:, drone_index]

            drone.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            drone.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
            drone.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            drone.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
            drone.set_thrust_target(self._hover_thrust[env_ids], env_ids=env_ids)
            self._thrust_targets[agent][env_ids] = self._hover_thrust[env_ids]

    def _collect_swarm_state(self) -> dict[str, torch.Tensor]:
        positions = []
        orientations = []
        linear_velocities = []
        angular_velocities = []
        rotation_matrices = []
        for agent in self._agent_ids:
            drone = self.drones[agent]
            root_quat = wp.to_torch(drone.data.root_quat_w)
            positions.append(wp.to_torch(drone.data.root_pos_w) - self.scene.env_origins)
            orientations.append(root_quat)
            linear_velocities.append(wp.to_torch(drone.data.root_lin_vel_w))
            angular_velocity = getattr(drone.data, "root_ang_vel_b", drone.data.root_ang_vel_w)
            angular_velocities.append(wp.to_torch(angular_velocity))
            rotation_matrices.append(math_utils.matrix_from_quat(root_quat))

        return {
            "positions": torch.stack(positions, dim=1),
            "orientations": torch.stack(orientations, dim=1),
            "linear_velocities": torch.stack(linear_velocities, dim=1),
            "angular_velocities": torch.stack(angular_velocities, dim=1),
            "rotation_matrices": torch.stack(rotation_matrices, dim=1),
        }

    def _make_snapshot(self, state: dict[str, torch.Tensor]) -> SwarmReplaySnapshot:
        return SwarmReplaySnapshot(
            positions=state["positions"],
            orientations=state["orientations"],
            linear_velocities=state["linear_velocities"],
            angular_velocities=state["angular_velocities"],
            goals=self._goals,
            obstacle_positions=self._obstacle_positions,
            obstacle_mask=self._obstacle_mask,
            episode_lengths=self.episode_length_buf,
        )

    def _spawn_view_camera(self) -> None:
        camera_path = self.cfg.viewer.cam_prim_path
        if not camera_path or camera_path == "/OmniverseKit_Persp":
            return
        if self.sim.stage.GetPrimAtPath(camera_path).IsValid():
            return

        eye = torch.tensor([self.cfg.viewer.eye], dtype=torch.float32)
        target = torch.tensor([self.cfg.viewer.lookat], dtype=torch.float32)
        orientation = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eye, target, up_axis="Z")
        )[0]
        camera_cfg = sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=20.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        )
        camera_cfg.func(
            camera_path,
            camera_cfg,
            translation=tuple(float(value) for value in eye[0]),
            orientation=tuple(float(value) for value in orientation),
        )

    def _spawn_obstacle_visuals(self) -> None:
        self._obstacle_visual_prims = []
        if not self.cfg.spawn_obstacle_actors:
            return

        env_index = min(max(int(self.cfg.viewer.env_index), 0), int(self.cfg.scene.num_envs) - 1)
        self._obstacle_visual_env_index = env_index
        obstacle_cfg = sim_utils.CylinderCfg(
            radius=self.cfg.obstacle_radius,
            height=spec.ROOM_HEIGHT,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.55, 0.18), roughness=0.8),
        )
        num_slots = spec.OBSTACLE_GRID_SHAPE[0] * spec.OBSTACLE_GRID_SHAPE[1]
        for obstacle_index in range(num_slots):
            prim_path = f"/World/envs/env_{env_index}/obstacles/obstacle_{obstacle_index:02d}"
            prim = obstacle_cfg.func(prim_path, obstacle_cfg, translation=(0.0, 0.0, -spec.ROOM_HEIGHT))
            sim_utils.set_prim_visibility(prim, False)
            self._obstacle_visual_prims.append(prim)

    def _sync_obstacle_visuals(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_obstacle_visual_prims", None):
            return

        env_index = getattr(self, "_obstacle_visual_env_index", 0)
        if not (env_ids == env_index).any():
            return

        origin = self.scene.env_origins[env_index].detach().cpu()
        positions = self._obstacle_positions[env_index].detach().cpu()
        mask = self._obstacle_mask[env_index].detach().cpu()
        for obstacle_index, prim in enumerate(self._obstacle_visual_prims):
            visible = bool(mask[obstacle_index])
            sim_utils.set_prim_visibility(prim, visible)
            if visible:
                position = positions[obstacle_index] + origin
                sim_utils.standardize_xform_ops(
                    prim,
                    translation=tuple(float(value) for value in position),
                )
