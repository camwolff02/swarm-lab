# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for debugging the paper_swarm task.

Records per-step state data (positions, velocities, goal distances)
into HDF5 datasets for offline analysis.
"""

from __future__ import annotations

import torch
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass


class DroneStateRecorder(RecorderTerm):
    """Records drone positions, velocities, and orientations every step."""

    def record_pre_step(self):
        """Record state before physics step."""
        env_view = self._env
        root_env = env_view.root if hasattr(env_view, "root") else env_view
        num_envs = root_env.num_envs
        possible_agents = root_env.possible_agents
        num_drones = len(possible_agents)
        device = root_env.device

        positions = torch.zeros(num_envs, num_drones, 3, device=device)
        quaternions = torch.zeros(num_envs, num_drones, 4, device=device)
        velocities = torch.zeros(num_envs, num_drones, 3, device=device)
        ang_velocities = torch.zeros(num_envs, num_drones, 3, device=device)

        for i, agent_id in enumerate(possible_agents):
            asset = root_env.scene[agent_id].data
            positions[:, i, :] = asset.root_pos_w.torch
            quaternions[:, i, :] = asset.root_quat_w.torch
            velocities[:, i, :] = asset.root_lin_vel_w.torch
            ang_velocities[:, i, :] = asset.root_ang_vel_w.torch

        return None, {
            "positions": positions,
            "quaternions": quaternions,
            "velocities": velocities,
            "ang_velocities": ang_velocities,
        }


class GoalDistanceRecorder(RecorderTerm):
    """Records goal positions and distances."""

    def record_pre_step(self):
        """Record goal info."""
        env_view = self._env
        root_env = env_view.root if hasattr(env_view, "root") else env_view
        num_envs = root_env.num_envs
        possible_agents = root_env.possible_agents
        num_drones = len(possible_agents)
        device = root_env.device

        goal_positions = torch.zeros(num_envs, num_drones, 3, device=device)
        goal_distances = torch.zeros(num_envs, num_drones, device=device)

        command_manager = getattr(env_view, "command_manager", None)
        if command_manager is not None:
            cmd = command_manager.get_command("target_pose")
            for i, agent_id in enumerate(possible_agents):
                goal_positions[:, i, :] = cmd[:, :3]
                pos = root_env.scene[agent_id].data.root_pos_w.torch
                goal_distances[:, i] = torch.norm(cmd[:, :3] - pos, dim=-1)

        return None, {
            "goal_positions": goal_positions,
            "goal_distances": goal_distances,
        }


@configclass
class DroneStateRecorderCfg(RecorderTermCfg):
    class_type: type = DroneStateRecorder


@configclass
class GoalDistanceRecorderCfg(RecorderTermCfg):
    class_type: type = GoalDistanceRecorder


@configclass
class PaperSwarmRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager config for paper_swarm debugging."""

    dataset_filename: str = "paper_swarm_dataset"

    record_drone_state = DroneStateRecorderCfg()
    record_goal_distance = GoalDistanceRecorderCfg()
