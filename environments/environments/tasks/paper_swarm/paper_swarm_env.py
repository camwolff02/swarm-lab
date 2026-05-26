# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paper swarm manager-based MARL environment with collision replay."""

from __future__ import annotations

import torch
from cpsquare_lab.tasks.swarm.collision_replay import CollisionReplayManager, SwarmReplayBatch, SwarmReplaySnapshot

from environments.envs import ManagerBasedMarlEnv

from .mdp.commands import _sync_swarm_pose_commands


class PaperSwarmMarlEnv(ManagerBasedMarlEnv):
    """Task-specific manager MARL environment for paper swarm replay training."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._paper_swarm_replay: CollisionReplayManager | None = None
        self._paper_swarm_robot_collision_now: torch.Tensor | None = None
        self._paper_swarm_obstacle_collision_now: torch.Tensor | None = None
        self._paper_swarm_robot_collision_events: torch.Tensor | None = None
        self._paper_swarm_obstacle_collision_events: torch.Tensor | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._setup_replay()

    def _setup_replay(self) -> None:
        agent_count = len(self.possible_agents)
        self._paper_swarm_robot_collision_now = torch.zeros(
            self.num_envs, agent_count, device=self.device, dtype=torch.bool
        )
        self._paper_swarm_obstacle_collision_now = torch.zeros_like(self._paper_swarm_robot_collision_now)
        self._paper_swarm_robot_collision_events = torch.zeros_like(self._paper_swarm_robot_collision_now)
        self._paper_swarm_obstacle_collision_events = torch.zeros_like(self._paper_swarm_robot_collision_now)
        if not getattr(self.cfg, "replay_enabled", False):
            return
        history_steps = max(1, round(float(self.cfg.replay_lag_s) / self.step_dt))
        self._paper_swarm_replay = CollisionReplayManager(
            num_envs=self.num_envs,
            history_steps=history_steps,
            capacity=int(self.cfg.replay_capacity),
            replay_probability=float(self.cfg.replay_probability),
            max_uses=int(self.cfg.replay_max_uses),
        )

    def close(self):
        """Close task-local recorder resources before shutting down simulation."""
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.close()
        super().close()

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._update_collision_events()
        dones = super()._get_dones()
        if self._paper_swarm_replay is not None:
            self._record_replay_state()
        return dones

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        self._clear_collision_state(env_ids)
        if self._paper_swarm_replay is None:
            return
        replay_batch = self._paper_swarm_replay.sample_batch(env_ids)
        self._paper_swarm_replay.reset_history(env_ids)
        if replay_batch is not None:
            self._restore_replay_batch(replay_batch)

    def _update_collision_events(self) -> None:
        if self._paper_swarm_robot_collision_now is None or self._paper_swarm_obstacle_collision_now is None:
            return
        active = self._active_mask()
        positions = self._agent_positions()
        pairwise = torch.cdist(positions, positions)
        eye = torch.eye(len(self.possible_agents), device=self.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = active.unsqueeze(1) & active.unsqueeze(2) & ~eye
        robot_collision_matrix = (pairwise < float(self.cfg.collision_distance)) & valid_pair
        robot_now = robot_collision_matrix.any(dim=-1)

        columns = getattr(self, "column_positions", None)
        if columns is None or columns.shape[1] == 0:
            obstacle_now = torch.zeros_like(robot_now)
        else:
            active_columns = torch.linalg.norm(columns[:, :, :2], dim=-1) < 100.0
            dist_xy = torch.linalg.norm(positions[:, :, None, :2] - columns[:, None, :, :2], dim=-1)
            obstacle_now = (
                dist_xy < float(getattr(self.cfg, "obstacle_collision_distance", 0.2))
            ) & active.unsqueeze(-1) & active_columns.unsqueeze(1)
            obstacle_now = obstacle_now.any(dim=-1)

        self._paper_swarm_robot_collision_events = robot_now & ~self._paper_swarm_robot_collision_now
        self._paper_swarm_obstacle_collision_events = obstacle_now & ~self._paper_swarm_obstacle_collision_now
        self._paper_swarm_robot_collision_now = robot_now
        self._paper_swarm_obstacle_collision_now = obstacle_now

    def _record_replay_state(self) -> None:
        assert self._paper_swarm_replay is not None
        snapshot = self._make_replay_snapshot()
        collision_envs = (
            self._paper_swarm_robot_collision_events.any(dim=1)
            | self._paper_swarm_obstacle_collision_events.any(dim=1)
        )
        grace_steps = max(0, round(float(self.cfg.collision_grace_period_s) / self.step_dt))
        collision_envs &= self.episode_length_buf >= grace_steps
        env_ids = collision_envs.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            count = self._paper_swarm_replay.record_collisions(env_ids)
            self.extras["replay/collision_records"] = torch.tensor(float(count), device=self.device).unsqueeze(0)
        self._paper_swarm_replay.push_state(snapshot)
        self.extras["replay/available"] = torch.tensor(
            float(self._paper_swarm_replay.available_count), device=self.device
        ).unsqueeze(0)

    def _make_replay_snapshot(self) -> SwarmReplaySnapshot:
        positions = self._agent_positions()
        orientations = torch.stack(
            [self.scene[agent_id].data.root_quat_w.torch for agent_id in self.possible_agents], dim=1
        )
        linear_velocities = torch.stack(
            [self.scene[agent_id].data.root_lin_vel_w.torch for agent_id in self.possible_agents], dim=1
        )
        angular_velocities = torch.stack(
            [self.scene[agent_id].data.root_ang_vel_w.torch for agent_id in self.possible_agents], dim=1
        )
        goals = self._agent_goals()
        columns = getattr(self, "column_positions", None)
        if columns is None:
            columns = torch.zeros(self.num_envs, 0, 3, device=self.device)
        if columns.shape[1] > 0:
            obstacle_mask = torch.linalg.norm(columns[:, :, :2], dim=-1) < 100.0
        else:
            obstacle_mask = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.bool)
        return SwarmReplaySnapshot(
            positions=positions,
            orientations=orientations,
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities,
            goals=goals,
            obstacle_positions=columns,
            obstacle_mask=obstacle_mask,
            episode_lengths=self.episode_length_buf.clone(),
        )

    def _restore_replay_batch(self, replay_batch: SwarmReplayBatch) -> None:
        env_ids = replay_batch.env_ids.to(device=self.device, dtype=torch.long)
        snapshot = replay_batch.snapshot
        if snapshot.obstacle_positions.numel() > 0:
            columns = snapshot.obstacle_positions.to(self.device).clone()
            inactive = ~snapshot.obstacle_mask.to(self.device).bool()
            columns[inactive] = torch.tensor((1000.0, 1000.0, 0.0), device=self.device)
            all_columns = getattr(self, "column_positions", None)
            if all_columns is None or all_columns.shape[1:] != columns.shape[1:]:
                all_columns = torch.zeros(self.num_envs, *columns.shape[1:], device=self.device)
            all_columns[env_ids] = columns
            self.column_positions = all_columns
        self.episode_length_buf[env_ids] = snapshot.episode_lengths.to(self.device)
        self._write_swarm_state(
            env_ids,
            positions=snapshot.positions.to(self.device),
            orientations=snapshot.orientations.to(self.device),
            linear_velocities=snapshot.linear_velocities.to(self.device),
            angular_velocities=snapshot.angular_velocities.to(self.device),
        )
        self._restore_goals(env_ids, snapshot.goals.to(self.device))

    def _write_swarm_state(
        self,
        env_ids: torch.Tensor,
        *,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        linear_velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
    ) -> None:
        for index, agent_id in enumerate(self.possible_agents):
            drone = self.scene[agent_id]
            root_pose = drone.data.default_root_pose.torch[env_ids].clone()
            root_velocity = drone.data.default_root_vel.torch[env_ids].clone()
            root_pose[:, :3] = positions[:, index] + self.scene.env_origins[env_ids]
            root_pose[:, 3:7] = orientations[:, index]
            root_velocity[:, :3] = linear_velocities[:, index]
            root_velocity[:, 3:6] = angular_velocities[:, index]
            drone.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            drone.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
            if hasattr(drone.data, "default_joint_pos"):
                drone.write_joint_position_to_sim_index(
                    position=drone.data.default_joint_pos.torch[env_ids], env_ids=env_ids
                )
                drone.write_joint_velocity_to_sim_index(
                    velocity=drone.data.default_joint_vel.torch[env_ids], env_ids=env_ids
                )

    def _restore_goals(self, env_ids: torch.Tensor, goals: torch.Tensor) -> None:
        first_command = None
        for agent_id in self.possible_agents:
            bundle = self._manager_bundles[self._agent_to_bundle[agent_id]]
            command = bundle.command_manager._terms.get("target_pose")
            if command is not None:
                first_command = command
                break
        if first_command is None:
            return
        _sync_swarm_pose_commands(self, env_ids, self.possible_agents, goals, first_command.time_left[env_ids])

    def _agent_positions(self) -> torch.Tensor:
        return torch.stack(
            [self.scene[agent_id].data.root_pos_w.torch - self.scene.env_origins for agent_id in self.possible_agents],
            dim=1,
        )

    def _agent_goals(self) -> torch.Tensor:
        goals = []
        for agent_id in self.possible_agents:
            bundle = self._manager_bundles[self._agent_to_bundle[agent_id]]
            goals.append(bundle.command_manager.get_command("target_pose"))
        return torch.stack(goals, dim=1)

    def _active_mask(self) -> torch.Tensor:
        mask_key = getattr(self.cfg, "active_agent_mask_key", None)
        if mask_key is None or not hasattr(self, mask_key):
            return torch.ones(self.num_envs, len(self.possible_agents), device=self.device, dtype=torch.bool)
        return getattr(self, mask_key)

    def _clear_collision_state(self, env_ids: torch.Tensor) -> None:
        for buffer in (
            self._paper_swarm_robot_collision_now,
            self._paper_swarm_obstacle_collision_now,
            self._paper_swarm_robot_collision_events,
            self._paper_swarm_obstacle_collision_events,
        ):
            if buffer is not None:
                buffer[env_ids] = False
