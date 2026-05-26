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

        return "drone_state", {
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

        for i, agent_id in enumerate(possible_agents):
            bundle_name = root_env._agent_to_bundle.get(agent_id)
            if bundle_name is None:
                continue
            command_manager = root_env._manager_bundles[bundle_name].command_manager
            if command_manager is None:
                continue
            cmd = command_manager.get_command("target_pose")
            goal_w = cmd[:, :3] + root_env.scene.env_origins
            goal_positions[:, i, :] = goal_w
            pos = root_env.scene[agent_id].data.root_pos_w.torch
            goal_distances[:, i] = torch.norm(goal_w - pos, dim=-1)

        return "goal", {
            "goal_positions": goal_positions,
            "goal_distances": goal_distances,
        }


class InitialStateCheckRecorder(RecorderTerm):
    """One-shot validation of initial drone states on first step after reset.

    Checks recorded into HDF5 as scalar flags per environment:
    - ``all_upright``: all active drone quaternions have |w| ≈ 1
    - ``all_in_bounds``: all active drones within configured XY/Z workspace
    - ``all_separated``: pairwise separation >= configured minimum
    - ``inactive_parked``: inactive drones are at z ≈ -10

    Configuration is read from ``self._env`` attributes set by the
    :class:`PaperSwarmEvalEnvCfg`.
    """

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        self._first_step = True

    def record_pre_step(self):
        """Run state check only on the very first call."""
        if not self._first_step:
            return None, None
        self._first_step = False

        env_view = self._env
        root_env = env_view.root if hasattr(env_view, "root") else env_view
        env_origins = root_env.scene.env_origins
        possible_agents = root_env.possible_agents
        num_envs = root_env.num_envs
        device = root_env.device

        # Read bounds from env config (set by PaperSwarmEvalEnvCfg)
        xy_bound = float(getattr(root_env, "eval_xy_bound", 1.5))
        z_min = float(getattr(root_env, "eval_z_min", 1.0))
        z_max = float(getattr(root_env, "eval_z_max", 1.5))
        min_sep = float(getattr(root_env, "eval_min_separation", 2.0))

        all_upright = torch.ones(num_envs, dtype=torch.bool, device=device)
        all_in_bounds = torch.ones(num_envs, dtype=torch.bool, device=device)
        all_separated = torch.ones(num_envs, dtype=torch.bool, device=device)
        inactive_parked = torch.ones(num_envs, dtype=torch.bool, device=device)

        for e in range(num_envs):
            pos_local = {}
            for agent_id in possible_agents:
                asset = root_env.scene[agent_id].data
                pos_w = asset.root_pos_w.torch[e]
                quat = asset.root_quat_w.torch[e]
                pos = pos_w - env_origins[e]

                is_active = pos[2] > 0.0
                if is_active:
                    # Check quaternion magnitude
                    if abs(quat.norm() - 1.0) > 0.01:
                        all_upright[e] = False
                    # Check within workspace
                    if abs(pos[0]) > xy_bound or abs(pos[1]) > xy_bound:
                        all_in_bounds[e] = False
                    if pos[2] < z_min or pos[2] > z_max:
                        all_in_bounds[e] = False
                    pos_local[agent_id] = pos[:2]
                else:
                    if abs(pos[2] - (-10.0)) > 1.0:
                        inactive_parked[e] = False

            # Pairwise separation
            agents = list(pos_local.keys())
            for j in range(len(agents)):
                for k in range(j + 1, len(agents)):
                    if torch.norm(pos_local[agents[j]] - pos_local[agents[k]]) < min_sep:
                        all_separated[e] = False
                        break

        return "initial_state", {
            "all_upright": all_upright.float(),
            "all_in_bounds": all_in_bounds.float(),
            "all_separated": all_separated.float(),
            "inactive_parked": inactive_parked.float(),
        }


@configclass
class DroneStateRecorderCfg(RecorderTermCfg):
    class_type: type = DroneStateRecorder


@configclass
class GoalDistanceRecorderCfg(RecorderTermCfg):
    class_type: type = GoalDistanceRecorder


@configclass
class InitialStateCheckRecorderCfg(RecorderTermCfg):
    class_type: type = InitialStateCheckRecorder


@configclass
class PaperSwarmRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager config for paper_swarm debugging."""

    dataset_filename: str = "paper_swarm_dataset"
    export_in_close: bool = True

    record_drone_state = DroneStateRecorderCfg()
    record_goal_distance = GoalDistanceRecorderCfg()
    check_initial_state = InitialStateCheckRecorderCfg()
