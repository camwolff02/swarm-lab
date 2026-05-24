# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the paper_swarm waypoint-navigation task.

Blend from both papers:
- From collision_swarm: distance-to-goal, collision avoidance, control effort
- From formation_swarm: multi-objective weighted structure, smoothness terms

All reward functions accept the agent env_view as first argument.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import action_rate_l2  # noqa: F401


def waypoint_tracking_reward(
    env, asset_cfg, agent_id: str, command_name: str, std: float, mask_key: str
) -> torch.Tensor:
    """Exponential position tracking reward: exp(-||pos_error||^2 / std^2).

    Args:
        std: Gaussian standard deviation [m] that controls reward spread.

    Returns:
        Reward [dimensionless], shape (num_envs,).
    """
    from .observations import _as_torch

    target_pos = env.command_manager.get_command(command_name)[:, :3]
    current_pos = _as_torch(env.scene[asset_cfg.name].data.root_pos_w)[: target_pos.shape[0]]
    pos_error = target_pos - current_pos
    reward = torch.exp(-torch.sum(pos_error**2, dim=-1) / (std**2))
    mask = _get_active_mask(env, agent_id, mask_key)
    return reward * mask


def heading_tracking_reward(
    env, asset_cfg, agent_id: str, command_name: str, std: float, mask_key: str
) -> torch.Tensor:
    """Exponential heading tracking reward: exp(-(yaw_error)^2 / std^2).

    Args:
        std: Gaussian standard deviation [rad] that controls reward spread.

    Returns:
        Reward [dimensionless], shape (num_envs,).
    """
    from .observations import _as_torch, _asset, _quat_to_euler

    from isaaclab.utils.math import wrap_to_pi

    command = env.command_manager.get_command(command_name)
    target_yaw = command[:, 5]
    quat = _as_torch(_asset(env, asset_cfg).data.root_quat_w)[: command.shape[0]]
    _, _, current_yaw = _quat_to_euler(quat)
    yaw_error = wrap_to_pi(target_yaw - current_yaw)
    reward = torch.exp(-(yaw_error**2) / (std**2))
    mask = _get_active_mask(env, agent_id, mask_key)
    return reward * mask


def reached_target_pose(
    env, asset_cfg, agent_id: str, command_name: str, distance_threshold: float, yaw_threshold: float, mask_key: str
) -> torch.Tensor:
    """Binary bonus when the drone is within goal tolerance.

    Args:
        distance_threshold: Position tolerance [m].
        yaw_threshold: Yaw tolerance [rad].

    Returns:
        Bonus [dimensionless], shape (num_envs,).
    """
    from .observations import _as_torch, _asset, _quat_to_euler

    from isaaclab.utils.math import wrap_to_pi

    target_pos = env.command_manager.get_command(command_name)[:, :3]
    n = target_pos.shape[0]
    current_pos = _as_torch(_asset(env, asset_cfg).data.root_pos_w)[:n]
    dist = torch.norm(target_pos - current_pos, dim=-1)

    _, target_yaw, _ = _decomposed_target(env, command_name)
    quat = _as_torch(_asset(env, asset_cfg).data.root_quat_w)[:n]
    _, _, current_yaw = _quat_to_euler(quat)
    yaw_err = wrap_to_pi(target_yaw - current_yaw).abs()

    reached = (dist < distance_threshold) & (yaw_err < yaw_threshold)
    mask = _get_active_mask(env, agent_id, mask_key)
    return reached.float() * mask


def collision_avoidance_reward(
    env, asset_cfg, agent_id: str, agent_ids: list[str], safe_distance: float, collision_distance: float, mask_key: str
) -> torch.Tensor:
    """Linear penalty for inter-agent proximity, zero beyond safe_distance [dimensionless].

    Args:
        safe_distance: Distance [m] above which penalty is zero.
        collision_distance: Distance [m] below which penalty is -1.0.

    Returns:
        Reward [dimensionless], shape (num_envs,). Clipped to [-1, 0].
    """
    from .observations import _all_root_pos, _as_torch

    ego_pos = _as_torch(_asset(env, asset_cfg).data.root_pos_w)
    mask = _get_active_mask(env, agent_id, mask_key)
    current_index = agent_ids.index(asset_cfg.name)
    all_pos = _all_root_pos(env, agent_ids)
    dist = torch.linalg.norm(all_pos - ego_pos.unsqueeze(1), dim=-1)
    valid = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    valid[current_index] = False
    penalty = torch.clamp((safe_distance - dist[:, valid]) / (safe_distance - collision_distance), 0.0, 1.0)
    total_penalty = penalty.sum(dim=-1)
    total_penalty = torch.clamp(total_penalty, 0.0, 1.0)
    return -total_penalty * mask


def obstacle_avoidance_reward(
    env, asset_cfg, column_positions_key: str, column_radius: float, safe_distance: float
) -> torch.Tensor:
    """Linear penalty for proximity to static obstacles [dimensionless].

    Args:
        column_radius: Obstacle radius [m].
        safe_distance: Distance [m] above which penalty is zero.

    Returns:
        Reward [dimensionless], shape (num_envs,). Clipped to [-1, 0].
    """
    from .observations import _as_torch

    asset = _asset(env, asset_cfg)
    drone_pos = _as_torch(asset.data.root_pos_w)

    columns = getattr(_root_env(env), column_positions_key, None)
    if columns is None or columns.shape[1] == 0:
        return torch.zeros(env.num_envs, device=env.device)

    dist_xy = torch.linalg.norm(drone_pos[:, None, :2] - columns[:, :, :2], dim=-1)
    min_dist = dist_xy.min(dim=-1).values

    penalty = torch.clamp((safe_distance - (min_dist - column_radius)) / (safe_distance - column_radius), 0.0, 1.0)
    return -penalty


def body_rate_l2(env, asset_cfg, agent_id: str, mask_key: str) -> torch.Tensor:
    """L2 penalty on body angular velocity [rad/s]^2, shape (num_envs,)."""
    from .observations import _as_torch

    ang_vel = _as_torch(_asset(env, asset_cfg).data.root_ang_vel_b)
    mask = _get_active_mask(env, agent_id, mask_key)
    return torch.sum(ang_vel**2, dim=-1) * mask


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_active_mask(env, agent_id: str, mask_key: str) -> torch.Tensor:
    """Get the active mask for a specific agent as a float vector, shape (num_envs,)."""
    from .observations import _active_mask

    root = env.root if hasattr(env, "root") else env
    agent_ids = root.cfg.possible_agents
    mask = _active_mask(env, agent_ids, mask_key)
    try:
        index = agent_ids.index(agent_id)
    except ValueError:
        return torch.ones(env.num_envs, device=env.device)
    return mask[:, index].float()


def _decomposed_target(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract position, yaw, and sin/cos from a pose command."""
    cmd = env.command_manager.get_command(command_name)
    pos = cmd[:, :3]
    _, _, yaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]
    return pos, yaw, torch.stack([torch.sin(cmd[:, 5]), torch.cos(cmd[:, 5])], dim=-1)


def _root_env(env) -> object:
    """Return the root env when the active view is a scoped proxy."""
    if hasattr(env, "root"):
        return env.root
    return env


def _asset(env, asset_cfg) -> object:
    """Look up an articulation asset from a SceneEntityCfg."""
    return _root_env(env).scene[asset_cfg.name]
