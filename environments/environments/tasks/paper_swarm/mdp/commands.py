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
    expected_token = (root.common_step_counter, tuple(int(x) for x in env_ids.detach().cpu().tolist()))
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
    time_left: torch.Tensor | None = None,
) -> None:
    """Apply one coordinated target plan to every drone command term.

    Args:
        root: The root ManagerBasedMarlEnv instance.
        env_ids: Environment indices to update.
        agent_ids: Ordered list of agent ids whose commands are synced.
        target_plan: Target poses ``(len(env_ids), num_agents, 7)``.
        time_left: Optional per-agent time-left values. When ``None``, the
            time-left is copied from the first command term that has a valid
            ``target_pose`` entry.
    """
    manager_bundles = getattr(root, "_manager_bundles", {})
    agent_to_bundle = getattr(root, "_agent_to_bundle", {})
    if time_left is None:
        for agent_id in agent_ids:
            bundle_name = agent_to_bundle.get(agent_id)
            if bundle_name is None:
                continue
            command_manager = manager_bundles[bundle_name].command_manager
            if command_manager is None:
                continue
            term = command_manager._terms.get("target_pose")
            if term is not None:
                time_left = term.time_left[env_ids]
                break
    if time_left is None:
        return
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
    plan[..., 6] = 1.0

    mask = _active_mask(root, agent_ids)
    workspace_xy = getattr(root, "_paper_swarm_workspace_xy", (-4.0, 4.0))
    workspace_z = getattr(root, "_paper_swarm_workspace_z", (1.0, 3.0))
    min_separation = getattr(root, f"_paper_swarm_{sample_kind}_min_separation", 2.0)
    safe_prob = getattr(root, f"_paper_swarm_{sample_kind}_safe_sampling_prob", 1.0)
    columns = getattr(root, "column_positions", None)
    column_radius = getattr(root, "_paper_swarm_column_radius", 0.15)
    column_safe_distance = getattr(root, "_paper_swarm_column_safe_distance", 0.6)

    positions, yaws = _sample_positions_vectorized(
        active_mask=mask[env_ids],
        xy_bounds=workspace_xy,
        z_bounds=workspace_z,
        min_separation=min_separation,
        safe_prob=safe_prob,
        columns=columns[env_ids] if columns is not None else None,
        column_radius=column_radius,
        column_safe_distance=column_safe_distance,
        device=root.device,
    )

    flat_yaws = yaws.reshape(-1)
    quats = quat_unique(
        quat_from_euler_xyz(
            torch.zeros_like(flat_yaws),
            torch.zeros_like(flat_yaws),
            flat_yaws,
        )
    ).reshape(num_envs, num_agents, 4)

    active_2d = mask[env_ids]
    for a in range(num_agents):
        active_col = active_2d[:, a]
        if active_col.any():
            plan[active_col, a, :3] = positions[active_col, a]
            plan[active_col, a, 3:7] = quats[active_col, a]

    return plan


def _active_mask(root, agent_ids: list[str]) -> torch.Tensor:
    mask_key = getattr(root.cfg, "active_agent_mask_key", None)
    if mask_key is None or not hasattr(root, mask_key):
        return torch.ones(root.num_envs, len(agent_ids), dtype=torch.bool, device=root.device)
    return getattr(root, mask_key)


def _sample_positions_vectorized(
    active_mask: torch.Tensor,
    xy_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    min_separation: float,
    safe_prob: float,
    columns: torch.Tensor | None,
    column_radius: float,
    column_safe_distance: float,
    device: torch.device,
    max_attempts: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample separated positions for multiple environments in parallel.

    Generates positions for all environments and the maximum number of active
    agents simultaneously using batched rejection sampling.  Each environment
    can have a different number of active agents (controlled by *active_mask*).

    Args:
        active_mask: ``(num_envs, num_agents)`` bool tensor. Only rows/columns
            marked ``True`` receive sampled positions.
        xy_bounds: (min, max) [m] for horizontal axes.
        z_bounds: (min, max) [m] for vertical axis.
        min_separation: Minimum XY distance between any pair [m].  Enforced only
            when ``safe_prob`` is sampled as ``True`` for a given environment.
        safe_prob: Per-environment probability of enforcing *min_separation*
            and obstacle avoidance.
        columns: ``(num_envs, max_cols, 3)`` or ``None``.  Obstacle positions;
            rows with ``norm(col[:,:2]) >= 100`` are treated as inactive.
        column_radius: Obstacle column radius [m].
        column_safe_distance: Extra clearance from column surface [m].
        device: Target PyTorch device.
        max_attempts: Maximum random candidates generated per agent per env.

    Returns:
        ``(positions, yaws)`` where *positions* is ``(num_envs, max_agents, 3)``
        and *yaws* is ``(num_envs, max_agents)``.  Slots beyond an
        environment's active count are filled with zeros.
    """
    num_envs, num_agents = active_mask.shape
    low, high = xy_bounds
    z_low, z_high = z_bounds
    positions = torch.zeros(num_envs, num_agents, 3, device=device)
    yaws = torch.zeros(num_envs, num_agents, device=device)

    safe_mask = torch.rand(num_envs, device=device) < safe_prob
    effective_min_sep = torch.where(safe_mask, min_separation, 0.0)

    max_active = int(active_mask.sum(dim=-1).max().item())
    if max_active == 0:
        return positions, yaws

    for agent_idx in range(max_active):
        needs_agent = active_mask[:, agent_idx]

        candidates = torch.empty(num_envs, max_attempts, 3, device=device)
        candidates[:, :, 0].uniform_(low, high)
        candidates[:, :, 1].uniform_(low, high)
        candidates[:, :, 2].uniform_(z_low, z_high)

        valid = torch.ones(num_envs, max_attempts, dtype=torch.bool, device=device)

        if agent_idx > 0 and effective_min_sep.max() > 0:
            prev_xy = positions[:, :agent_idx, :2]
            cand_xy = candidates[:, :, :2]
            dists = torch.cdist(cand_xy, prev_xy)
            min_dists = dists.min(dim=-1).values
            sep_violation = (min_dists < effective_min_sep.unsqueeze(-1)) & needs_agent.unsqueeze(-1)
            valid = valid & ~sep_violation

        if columns is not None and columns.shape[1] > 0 and safe_mask.any():
            cand_xy = candidates[:, :, None, :2]
            col_xy = columns[:, None, :, :2]
            col_dists = torch.linalg.norm(cand_xy - col_xy, dim=-1)
            col_valid = col_dists >= (column_radius + column_safe_distance)
            col_active = torch.linalg.norm(columns[:, :, :2], dim=-1) < 100.0
            col_valid = col_valid | ~col_active[:, None, :]
            col_valid_all = col_valid.all(dim=-1) & safe_mask.unsqueeze(-1) | ~safe_mask.unsqueeze(-1)
            valid = valid & col_valid_all

        valid = valid & needs_agent.unsqueeze(-1)
        selected_idx = valid.float().argmax(dim=-1)
        positions[needs_agent, agent_idx] = candidates[needs_agent, selected_idx[needs_agent]]
        yaws[needs_agent, agent_idx] = (
            torch.empty(needs_agent.sum().item(), device=device).uniform_(-torch.pi, torch.pi)
        )

    return positions, yaws
