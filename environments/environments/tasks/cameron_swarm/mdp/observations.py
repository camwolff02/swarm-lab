# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Cameron drone waypoint task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def _root_env(env):
    return getattr(env, "root", env)


def _asset(env, asset_cfg):
    return env.scene[asset_cfg.name]


def _root_pos_env(env, agent_id: str) -> torch.Tensor:
    root = _root_env(env)
    return root.scene[agent_id].data.root_pos_w.torch - root.scene.env_origins


def _root_quat(env, agent_id: str) -> torch.Tensor:
    root = _root_env(env)
    return root.scene[agent_id].data.root_quat_w.torch


def _active_mask(env, agent_ids, mask_key: str | None) -> torch.Tensor:
    if mask_key is None or not hasattr(env, mask_key):
        return torch.ones((env.num_envs, len(agent_ids)), dtype=torch.bool, device=env.device)
    return getattr(env, mask_key)


def root_lin_vel_b(env, asset_cfg):
    """Root linear velocity in body frame [m/s], shape [num_envs, 3], float."""
    return _asset(env, asset_cfg).data.root_lin_vel_b.torch


def root_ang_vel_b(env, asset_cfg):
    """Root angular velocity in body frame [rad/s], shape [num_envs, 3], float."""
    return _asset(env, asset_cfg).data.root_ang_vel_b.torch


def projected_gravity_b(env, asset_cfg):
    """Projected gravity vector in body frame [-], shape [num_envs, 3], float."""
    return _asset(env, asset_cfg).data.projected_gravity_b.torch


def relative_target_position_b(env, asset_cfg, command_name: str):
    """Target position in body frame [m], shape [num_envs, 3], float."""
    del asset_cfg
    return env.command_manager.get_command(command_name)[:, :3]


def target_yaw_error(env, asset_cfg, command_name: str):
    """Yaw error relative to target [rad], shape [num_envs, 1], float."""
    del asset_cfg
    command = env.command_manager.get_command(command_name)
    if command.shape[-1] >= 7:
        _, _, yaw = math_utils.euler_xyz_from_quat(command[:, 3:7])
        return math_utils.wrap_to_pi(yaw).unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


def relative_neighbor_positions_b(env, asset_cfg, agent_ids, max_neighbors: int, radius: float, mask_key: str):
    """Relative neighbor positions in body frame [m], shape [num_envs, max_neighbors * 3], float."""
    ego = asset_cfg.name
    ego_pos = _root_pos_env(env, ego)
    ego_quat = _root_quat(env, ego)
    mask = _active_mask(env, agent_ids, mask_key)
    features = []
    for index, agent_id in enumerate(agent_ids):
        if agent_id == ego:
            continue
        rel_w = _root_pos_env(env, agent_id) - ego_pos
        rel_b = math_utils.quat_apply_inverse(ego_quat, rel_w)
        active = mask[:, index].unsqueeze(-1)
        in_radius = torch.linalg.norm(rel_b, dim=-1, keepdim=True) <= radius
        features.append(torch.where(active & in_radius, rel_b / max(radius, 1.0e-6), torch.zeros_like(rel_b)))
    return _pad_features(features, env.num_envs, max_neighbors, 3, env.device)


def relative_neighbor_velocities_b(env, asset_cfg, agent_ids, max_neighbors: int, radius: float, mask_key: str):
    """Relative neighbor velocities in body frame [m/s], shape [num_envs, max_neighbors * 3], float."""
    ego = asset_cfg.name
    ego_pos = _root_pos_env(env, ego)
    ego_quat = _root_quat(env, ego)
    ego_vel = env.scene[ego].data.root_lin_vel_w.torch
    mask = _active_mask(env, agent_ids, mask_key)
    features = []
    for index, agent_id in enumerate(agent_ids):
        if agent_id == ego:
            continue
        rel_pos = _root_pos_env(env, agent_id) - ego_pos
        rel_vel_b = math_utils.quat_apply_inverse(ego_quat, env.scene[agent_id].data.root_lin_vel_w.torch - ego_vel)
        active = mask[:, index].unsqueeze(-1)
        in_radius = torch.linalg.norm(rel_pos, dim=-1, keepdim=True) <= radius
        features.append(torch.where(active & in_radius, rel_vel_b, torch.zeros_like(rel_vel_b)))
    return _pad_features(features, env.num_envs, max_neighbors, 3, env.device)


def _pad_features(features, num_envs: int, max_neighbors: int, width: int, device) -> torch.Tensor:
    if not features:
        return torch.zeros((num_envs, max_neighbors * width), device=device)
    stacked = torch.stack(features[:max_neighbors], dim=1)
    if stacked.shape[1] < max_neighbors:
        pad = torch.zeros((num_envs, max_neighbors - stacked.shape[1], width), device=device)
        stacked = torch.cat((stacked, pad), dim=1)
    return stacked.reshape(num_envs, max_neighbors * width)


def agent_active_flag(env, agent_ids, agent_id: str, mask_key: str):
    """Binary flag indicating whether this agent is active [-], shape [num_envs, 1], float."""
    mask = _active_mask(env, agent_ids, mask_key)
    index = agent_ids.index(agent_id)
    return mask[:, index : index + 1].to(dtype=torch.float32)


def last_action(env, action_name: str):
    """Raw action from the previous step, shape [num_envs, action_dim], float."""
    return env.action_manager.get_term(action_name).raw_actions


def _command_for_agent(env, agent_id: str, command_name: str) -> torch.Tensor:
    """Return a command tensor for an agent in V2 agent-grouped execution."""
    root = _root_env(env)
    if getattr(env, "agent_ids", None) == (agent_id,) and env.command_manager is not None:
        return env.command_manager.get_command(command_name)
    bundle_name = getattr(root, "_agent_to_bundle", {}).get(agent_id)
    if bundle_name is not None:
        bundle = root._manager_bundles[bundle_name]
        if bundle.command_manager is not None:
            return bundle.command_manager.get_command(command_name)
    return env.command_manager.get_command(command_name)


def swarm_global_state(
    env,
    agent_ids,
    command_name: str,
    include_root_state: bool,
    include_target_pose: bool,
    include_pairwise_distances: bool,
    mask_key: str,
):
    """Centralized swarm state vector for MAPPO critic.

    The V2 environment has no ``_agent_context``. Commands for each drone are
    fetched through the root env's manager-bundle table when the task runs with
    ``manager_grouping="agent"``.
    """
    root = _root_env(env)
    mask = _active_mask(root, agent_ids, mask_key).to(dtype=torch.float32)
    chunks = [mask]
    if include_root_state:
        states = []
        for agent_id in agent_ids:
            asset = root.scene[agent_id]
            states.append(
                torch.cat(
                    (
                        asset.data.root_pos_w.torch - root.scene.env_origins,
                        asset.data.root_quat_w.torch,
                        asset.data.root_lin_vel_w.torch,
                        asset.data.root_ang_vel_w.torch,
                    ),
                    dim=-1,
                )
            )
        chunks.append(torch.stack(states, dim=1).reshape(root.num_envs, -1))
    if include_target_pose:
        commands = [_command_for_agent(env, agent_id, command_name) for agent_id in agent_ids]
        chunks.append(torch.stack(commands, dim=1).reshape(root.num_envs, -1))
    if include_pairwise_distances:
        positions = torch.stack([_root_pos_env(root, agent_id) for agent_id in agent_ids], dim=1)
        distances = torch.cdist(positions, positions)
        active_pair = mask.unsqueeze(1) * mask.unsqueeze(2)
        chunks.append((distances * active_pair).reshape(root.num_envs, -1))
    return torch.cat(chunks, dim=-1)
