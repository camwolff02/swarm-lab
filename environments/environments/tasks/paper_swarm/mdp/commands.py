# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the paper_swarm task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_unique
from isaaclab_tasks.manager_based.drone_arl.mdp.commands.drone_pose_command import DroneUniformPoseCommand


class PaperSwarmPoseCommand(DroneUniformPoseCommand):
    """Pose command that samples coordinated swarm waypoints.

    The base drone pose command samples each agent independently. This task
    needs curriculum-aware target generation so early training starts with
    separated, obstacle-free targets and later admits unconstrained samples.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        root = self._env.root if hasattr(self._env, "root") else self._env
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        agent_ids = list(root.cfg.possible_agents)
        agent_id = self.cfg.asset_name
        agent_index = agent_ids.index(agent_id)

        target_plan = _get_or_sample_target_plan(root, env_ids_tensor, agent_ids)
        self.pose_command_b[env_ids_tensor] = target_plan[:, agent_index, :]
        _sync_swarm_pose_commands(root, env_ids_tensor, agent_ids, target_plan, self.time_left[env_ids_tensor])


def _get_or_sample_target_plan(root, env_ids: torch.Tensor, agent_ids: list[str]) -> torch.Tensor:
    plan_key = "_paper_swarm_target_plan"
    token_key = "_paper_swarm_target_plan_token"
    expected_token = (int(root.common_step_counter), tuple(int(x) for x in env_ids.detach().cpu().tolist()))
    plan = getattr(root, plan_key, None)
    token = getattr(root, token_key, None)
    if plan is None or token != expected_token:
        plan = _sample_pose_plan(root, env_ids, agent_ids, sample_kind="target")
        setattr(root, plan_key, plan)
        setattr(root, token_key, expected_token)
    return plan


def _sync_swarm_pose_commands(
    root,
    env_ids: torch.Tensor,
    agent_ids: list[str],
    target_plan: torch.Tensor,
    time_left: torch.Tensor,
) -> None:
    """Apply one coordinated target plan to every drone command term."""
    manager_bundles = getattr(root, "_manager_bundles", {})
    agent_to_bundle = getattr(root, "_agent_to_bundle", {})
    for agent_index, agent_id in enumerate(agent_ids):
        bundle_name = agent_to_bundle.get(agent_id)
        if bundle_name is None:
            continue
        command_manager = manager_bundles[bundle_name].command_manager
        if command_manager is None:
            continue
        command_term = command_manager._terms.get("target_pose")
        if command_term is None:
            continue
        command_term.pose_command_b[env_ids] = target_plan[:, agent_index, :]
        command_term.time_left[env_ids] = time_left


def _sample_pose_plan(root, env_ids: torch.Tensor, agent_ids: list[str], sample_kind: str) -> torch.Tensor:
    num_envs = len(env_ids)
    num_agents = len(agent_ids)
    plan = torch.zeros(num_envs, num_agents, 7, device=root.device)
    plan[..., 3] = 0.0
    plan[..., 4] = 0.0
    plan[..., 5] = 0.0
    plan[..., 6] = 1.0

    mask = _active_mask(root, agent_ids)
    workspace_xy = getattr(root, "_paper_swarm_workspace_xy", (-4.0, 4.0))
    workspace_z = getattr(root, "_paper_swarm_workspace_z", (1.0, 3.0))
    min_separation = getattr(root, f"_paper_swarm_{sample_kind}_min_separation", 2.0)
    safe_prob = getattr(root, f"_paper_swarm_{sample_kind}_safe_sampling_prob", 1.0)
    columns = getattr(root, "column_positions", None)
    column_radius = getattr(root, "_paper_swarm_column_radius", 0.15)
    column_safe_distance = getattr(root, "_paper_swarm_column_safe_distance", 0.6)

    for local_env_index, env_id in enumerate(env_ids):
        active_indices = torch.nonzero(mask[env_id], as_tuple=False).flatten()
        safe = bool(torch.rand((), device=root.device) < safe_prob)
        env_columns = None if columns is None else columns[env_id]
        positions = _sample_positions(
            count=len(active_indices),
            xy_bounds=workspace_xy,
            z_bounds=workspace_z,
            min_separation=min_separation if safe else 0.0,
            columns=env_columns if safe else None,
            column_radius=column_radius,
            column_safe_distance=column_safe_distance,
            device=root.device,
        )
        yaw = torch.empty(len(active_indices), device=root.device).uniform_(-torch.pi, torch.pi)
        quat = quat_unique(quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw))
        plan[local_env_index, active_indices, :3] = positions
        plan[local_env_index, active_indices, 3:7] = quat

    return plan


def _active_mask(root, agent_ids: list[str]) -> torch.Tensor:
    mask_key = getattr(root.cfg, "active_agent_mask_key", None)
    if mask_key is None or not hasattr(root, mask_key):
        return torch.ones(root.num_envs, len(agent_ids), dtype=torch.bool, device=root.device)
    return getattr(root, mask_key)


def _sample_positions(
    count: int,
    xy_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    min_separation: float,
    columns: torch.Tensor | None,
    column_radius: float,
    column_safe_distance: float,
    device: torch.device,
    max_attempts: int = 256,
) -> torch.Tensor:
    positions = torch.zeros(count, 3, device=device)
    for index in range(count):
        candidate = _random_position(xy_bounds, z_bounds, device)
        for _ in range(max_attempts):
            candidate = _random_position(xy_bounds, z_bounds, device)
            if _is_position_valid(
                candidate, positions[:index], min_separation, columns, column_radius, column_safe_distance
            ):
                break
        positions[index] = candidate
    return positions


def _random_position(
    xy_bounds: tuple[float, float], z_bounds: tuple[float, float], device: torch.device
) -> torch.Tensor:
    low, high = xy_bounds
    z_low, z_high = z_bounds
    position = torch.empty(3, device=device)
    position[:2].uniform_(low, high)
    position[2].uniform_(z_low, z_high)
    return position


def _is_position_valid(
    candidate: torch.Tensor,
    existing: torch.Tensor,
    min_separation: float,
    columns: torch.Tensor | None,
    column_radius: float,
    column_safe_distance: float,
) -> bool:
    if existing.numel() > 0 and min_separation > 0.0:
        if torch.linalg.norm(existing[:, :2] - candidate[:2], dim=-1).min() < min_separation:
            return False
    if columns is not None and columns.numel() > 0:
        valid_columns = columns[torch.linalg.norm(columns[:, :2], dim=-1) < 100.0]
        if valid_columns.numel() > 0:
            dist = torch.linalg.norm(valid_columns[:, :2] - candidate[:2], dim=-1).min()
            if dist < column_radius + column_safe_distance:
                return False
    return True
