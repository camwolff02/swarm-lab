# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for cooperative CTBR drone waypoint navigation."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def _active_multiplier(env, agent_ids, agent_id: str, mask_key: str):
    if not hasattr(env, mask_key):
        return torch.ones(env.num_envs, device=env.device)
    return getattr(env, mask_key)[:, agent_ids.index(agent_id)].to(dtype=torch.float32)


def _position_error(env, asset_cfg, command_name: str) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    current_position = asset.data.root_pos_w.torch - env.scene.env_origins
    return command[:, :3] - current_position


def waypoint_tracking_reward(env, asset_cfg, agent_id: str, command_name: str, std: float, mask_key: str):
    """Exponential position error reward [-], shape [num_envs], float."""
    error = torch.linalg.norm(_position_error(env, asset_cfg, command_name), dim=-1)
    return torch.exp(-(error.square()) / std**2) * _active_multiplier(env, env.possible_agents, agent_id, mask_key)


def heading_tracking_reward(env, asset_cfg, agent_id: str, command_name: str, std: float, mask_key: str):
    """Exponential yaw error reward [-], shape [num_envs], float."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    if command.shape[-1] >= 7:
        _, _, target_yaw = math_utils.euler_xyz_from_quat(command[:, 3:7])
    else:
        target_yaw = torch.zeros_like(yaw)
    error = math_utils.wrap_to_pi(target_yaw - yaw)
    return torch.exp(-(error.square()) / std**2) * _active_multiplier(env, env.possible_agents, agent_id, mask_key)


def reached_target_pose(
    env,
    asset_cfg,
    agent_id: str,
    command_name: str,
    distance_threshold: float,
    yaw_threshold: float,
    mask_key: str,
):
    """Binary bonus when position [m] and yaw [rad] error are within thresholds, shape [num_envs], float."""
    position_reached = torch.linalg.norm(_position_error(env, asset_cfg, command_name), dim=-1) < distance_threshold
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    _, _, target_yaw = math_utils.euler_xyz_from_quat(command[:, 3:7])
    yaw_reached = torch.abs(math_utils.wrap_to_pi(target_yaw - yaw)) < yaw_threshold
    return (position_reached & yaw_reached).float() * _active_multiplier(env, env.possible_agents, agent_id, mask_key)


def collision_avoidance_reward(
    env,
    asset_cfg,
    agent_id: str,
    agent_ids,
    safe_distance: float,
    collision_distance: float,
    mask_key: str,
):
    """Linear penalty from collision_distance to safe_distance [m], clamped [0,1], shape [num_envs], float."""
    active = _active_multiplier(env, agent_ids, agent_id, mask_key)
    if active.sum() == 0:
        return active
    ego_pos = env.scene[asset_cfg.name].data.root_pos_w.torch - env.scene.env_origins
    reward = torch.zeros(env.num_envs, device=env.device)
    mask = getattr(env, mask_key, torch.ones((env.num_envs, len(agent_ids)), dtype=torch.bool, device=env.device))
    for other_index, other_id in enumerate(agent_ids):
        if other_id == agent_id:
            continue
        other_pos = env.scene[other_id].data.root_pos_w.torch - env.scene.env_origins
        dist = torch.linalg.norm(other_pos - ego_pos, dim=-1)
        other_active = mask[:, other_index].to(dtype=torch.float32)
        shaped = ((dist - collision_distance) / max(safe_distance - collision_distance, 1.0e-6)).clamp(0.0, 1.0)
        reward += (shaped - 1.0) * other_active
    return reward * active


def action_rate_l2(env, action_name: str):
    """L2 penalty on change in raw actions [-], shape [num_envs], float."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(term.raw_actions - env.action_manager.prev_action), dim=1)


def body_rate_l2(env, asset_cfg, agent_id: str, mask_key: str):
    """L2 penalty on body angular velocity [rad/s], shape [num_envs], float."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b.torch), dim=1) * _active_multiplier(
        env, env.possible_agents, agent_id, mask_key
    )
