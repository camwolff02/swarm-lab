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
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, wrap_to_pi

from .observations import _asset, _root_env, get_agent_active_mask as _get_active_mask
from .observations import command_decompose as _decomposed_target


def goal_distance_reward(env, asset_cfg, agent_id: str, command_name: str, mask_key: str) -> torch.Tensor:
    """Negative distance-to-goal reward [m], shape (num_envs,)."""
    target_pos = env.command_manager.get_command(command_name)[:, :3] + _root_env(env).scene.env_origins
    current_pos = _asset(env, asset_cfg).data.root_pos_w.torch[: target_pos.shape[0]]
    mask = _get_active_mask(env, agent_id, mask_key)
    return -torch.linalg.norm(target_pos - current_pos, dim=-1) * mask


def waypoint_tracking_reward(
    env, asset_cfg, agent_id: str, command_name: str, std: float, mask_key: str
) -> torch.Tensor:
    """Exponential position tracking reward: exp(-||pos_error||^2 / std^2).

    Args:
        std: Gaussian standard deviation [m] that controls reward spread.

    Returns:
        Reward [dimensionless], shape (num_envs,).
    """
    target_pos = env.command_manager.get_command(command_name)[:, :3] + _root_env(env).scene.env_origins
    current_pos = env.scene[asset_cfg.name].data.root_pos_w.torch[: target_pos.shape[0]]
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
    command = env.command_manager.get_command(command_name)
    _, _, target_yaw = euler_xyz_from_quat(command[:, 3:7])
    quat = _asset(env, asset_cfg).data.root_quat_w.torch[: command.shape[0]]
    _, _, current_yaw = euler_xyz_from_quat(quat)
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
    target_pos = env.command_manager.get_command(command_name)[:, :3] + _root_env(env).scene.env_origins
    n = target_pos.shape[0]
    current_pos = _asset(env, asset_cfg).data.root_pos_w.torch[:n]
    dist = torch.norm(target_pos - current_pos, dim=-1)

    _, target_yaw, _ = _decomposed_target(env, command_name)
    quat = _asset(env, asset_cfg).data.root_quat_w.torch[:n]
    _, _, current_yaw = euler_xyz_from_quat(quat)
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
    from .observations import _all_root_pos

    root = _root_env(env)
    ego_pos = _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins
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
    env,
    asset_cfg,
    agent_id: str,
    column_positions_key: str,
    column_radius: float,
    safe_distance: float,
    mask_key: str,
) -> torch.Tensor:
    """Linear penalty for proximity to static obstacles [dimensionless].

    Args:
        column_radius: Obstacle radius [m].
        safe_distance: Distance [m] above which penalty is zero.

    Returns:
        Reward [dimensionless], shape (num_envs,). Clipped to [-1, 0].
    """
    asset = _asset(env, asset_cfg)
    root = _root_env(env)
    drone_pos = asset.data.root_pos_w.torch - root.scene.env_origins

    columns = getattr(root, column_positions_key, None)
    if columns is None or columns.shape[1] == 0:
        return torch.zeros(env.num_envs, device=env.device)

    dist_xy = torch.linalg.norm(drone_pos[:, None, :2] - columns[:, :, :2], dim=-1)
    min_dist = dist_xy.min(dim=-1).values

    penalty = torch.clamp((safe_distance - (min_dist - column_radius)) / (safe_distance - column_radius), 0.0, 1.0)
    mask = _get_active_mask(env, agent_id, mask_key)
    return -penalty * mask


def body_rate_l2(env, asset_cfg, agent_id: str, mask_key: str) -> torch.Tensor:
    """L2 penalty on body angular velocity [rad/s]^2, shape (num_envs,)."""
    ang_vel = _asset(env, asset_cfg).data.root_ang_vel_b.torch
    mask = _get_active_mask(env, agent_id, mask_key)
    return torch.sum(ang_vel**2, dim=-1) * mask


def crash_penalty(
    env, asset_cfg, agent_id: str, minimum_height: float, mask_key: str
) -> torch.Tensor:
    """Binary penalty when drone is below minimum_height [dimensionless]."""
    asset = _asset(env, asset_cfg)
    root_height = asset.data.root_pos_w.torch[:, 2]
    mask = _get_active_mask(env, agent_id, mask_key)
    return (root_height < minimum_height).float() * mask


def upright_reward(env, asset_cfg, agent_id: str, mask_key: str) -> torch.Tensor:
    """Reward upright body orientation, shape (num_envs,)."""
    quat = _asset(env, asset_cfg).data.root_quat_w.torch
    up_z = matrix_from_quat(quat)[:, 2, 2].clamp(min=-1.0, max=1.0)
    mask = _get_active_mask(env, agent_id, mask_key)
    return up_z * mask


def robot_collision_event_penalty(env, agent_id: str, mask_key: str) -> torch.Tensor:
    """One-step penalty for newly detected robot collisions."""
    root = env.root if hasattr(env, "root") else env
    events = getattr(root, "_paper_swarm_robot_collision_events", None)
    if events is None:
        return torch.zeros(env.num_envs, device=env.device)
    agent_ids = root.cfg.possible_agents
    try:
        index = agent_ids.index(agent_id)
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device)
    mask = _get_active_mask(env, agent_id, mask_key)
    return events[:, index].float() * mask


def obstacle_collision_event_penalty(env, agent_id: str, mask_key: str) -> torch.Tensor:
    """One-step penalty for newly detected obstacle collisions."""
    root = env.root if hasattr(env, "root") else env
    events = getattr(root, "_paper_swarm_obstacle_collision_events", None)
    if events is None:
        return torch.zeros(env.num_envs, device=env.device)
    agent_ids = root.cfg.possible_agents
    try:
        index = agent_ids.index(agent_id)
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device)
    mask = _get_active_mask(env, agent_id, mask_key)
    return events[:, index].float() * mask


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


