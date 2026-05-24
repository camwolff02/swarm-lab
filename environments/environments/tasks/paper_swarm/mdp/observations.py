# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the paper_swarm waypoint-navigation task.

Observations blend the structure from both papers:
- Self state (from formation_swarm): pos, vel, quat, omega, heading, up, rel_vel
- Goal info (from collision_swarm): relative target position, yaw error
- Neighbors (from both): relative pos, distance, relative vel
- Static SDF (from both): 3x3 signed distance grid
- Auxiliary: active flag, last action
"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import (
    base_ang_vel as root_ang_vel_b,  # noqa: F401
    base_lin_vel as root_lin_vel_b,  # noqa: F401
    last_action,  # noqa: F401
    projected_gravity as projected_gravity_b,  # noqa: F401
)
from isaaclab.utils.math import quat_apply, wrap_to_pi


def _root_env(env) -> object:
    """Return the root env when the active view is a scoped proxy."""
    if hasattr(env, "root"):
        return env.root
    return env


def _as_torch(value) -> torch.Tensor:
    """Return a torch view for Isaac Lab beta2 tensor wrappers."""
    if torch.is_tensor(value):
        return value
    if hasattr(value, "torch"):
        return value.torch
    return torch.as_tensor(value)


def _asset(env, asset_cfg) -> object:
    """Look up an articulation asset from a SceneEntityCfg."""
    return _root_env(env).scene[asset_cfg.name]


def _root_pos(env, agent_id: str) -> torch.Tensor:
    """World position of an articulation relative to the env origin [m], shape (num_envs, 3)."""
    return _as_torch(_root_env(env).scene[agent_id].data.root_pos_w)


def _root_quat(env, agent_id: str) -> torch.Tensor:
    """World quaternion of an articulation [w,x,y,z], shape (num_envs, 4)."""
    return _as_torch(_root_env(env).scene[agent_id].data.root_quat_w)


def _root_lin_vel(env, agent_id: str) -> torch.Tensor:
    """World linear velocity of an articulation [m/s], shape (num_envs, 3)."""
    return _as_torch(_root_env(env).scene[agent_id].data.root_lin_vel_w)


def _all_root_pos(env, agent_ids: list[str]) -> torch.Tensor:
    """Stack all agent root positions [m], shape (num_envs, num_agents, 3)."""
    return torch.stack([_root_pos(env, agent_id) for agent_id in agent_ids], dim=1)


def _all_root_lin_vel(env, agent_ids: list[str]) -> torch.Tensor:
    """Stack all agent root velocities [m/s], shape (num_envs, num_agents, 3)."""
    return torch.stack([_root_lin_vel(env, agent_id) for agent_id in agent_ids], dim=1)


def _active_mask(env, agent_ids: list[str], mask_key: str) -> torch.Tensor:
    """Bool tensor (num_envs, len(agent_ids)) indicating active agents."""
    if mask_key is None or not hasattr(env, mask_key):
        return torch.ones((env.num_envs, len(agent_ids)), dtype=torch.bool, device=env.device)
    return getattr(env, mask_key)


def _pad_features(
    features: torch.Tensor,
    max_neighbors: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad or truncate neighbor features to fixed size (num_envs, max_neighbors, feat_dim).

    Args:
        features: Per-agent features of shape (num_envs, num_agents, feat_dim).
        max_neighbors: Fixed number of neighbor slots.
        mask: Optional bool mask (num_envs, num_agents) for valid entries.

    Returns:
        Tensor of shape (num_envs, max_neighbors * feat_dim).
    """
    num_envs, num_agents, feat_dim = features.shape
    if mask is not None:
        features = features * mask.unsqueeze(-1).float()
    max_slots = min(num_agents, max_neighbors)
    out = torch.zeros(num_envs, max_neighbors * feat_dim, device=features.device, dtype=features.dtype)
    out[:, : max_slots * feat_dim] = features[:, :max_slots, :].reshape(num_envs, -1)
    return out


def _rotate_world_to_body(quat_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the body frame."""
    if vectors_w.ndim == 2:
        return quat_apply(_quat_conj(quat_w), vectors_w)
    quat = _quat_conj(quat_w).unsqueeze(1).expand(-1, vectors_w.shape[1], -1)
    rotated = quat_apply(quat.reshape(-1, 4), vectors_w.reshape(-1, 3))
    return rotated.reshape_as(vectors_w)


# -----------------------------------------------------------------------------
# Observation functions
# -----------------------------------------------------------------------------


def root_pos(env, asset_cfg) -> torch.Tensor:
    """World position [m], shape (num_envs, 3)."""
    return _as_torch(_asset(env, asset_cfg).data.root_pos_w)


def root_quat(env, asset_cfg) -> torch.Tensor:
    """World quaternion, shape (num_envs, 4)."""
    return _as_torch(_asset(env, asset_cfg).data.root_quat_w)


def relative_target_position_b(env, asset_cfg, command_name: str) -> torch.Tensor:
    """Target waypoint position relative to drone, in body frame [m], shape (num_envs, 3)."""
    return env.command_manager.get_command(command_name)[:, :3]


def target_yaw_error(env, asset_cfg, command_name: str) -> torch.Tensor:
    """Yaw error to target [rad], wrapped to [-pi, pi], shape (num_envs, 1)."""
    command = env.command_manager.get_command(command_name)
    target_yaw = command[:, 5]
    quat = _as_torch(_asset(env, asset_cfg).data.root_quat_w)[: command.shape[0]]
    _, _, current_yaw = _quat_to_euler(quat)
    error = wrap_to_pi(target_yaw - current_yaw)
    return error.unsqueeze(-1)


def distance_to_goal(env, asset_cfg, command_name: str) -> torch.Tensor:
    """Euclidean distance to goal waypoint [m], shape (num_envs, 1)."""
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    current_pos = _as_torch(_asset(env, asset_cfg).data.root_pos_w)[: target_pos.shape[0]]
    return torch.norm(target_pos - current_pos, dim=-1, keepdim=True)


def goal_reached_flag(
    env, asset_cfg, command_name: str, distance_threshold: float, yaw_threshold: float
) -> torch.Tensor:
    """Binary flag: is the goal reached? Shape (num_envs, 1)."""
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    n = target_pos.shape[0]
    current_pos = _as_torch(_asset(env, asset_cfg).data.root_pos_w)[:n]
    dist = torch.norm(target_pos - current_pos, dim=-1)

    _, target_yaw, _ = command_decompose(env, command_name)
    quat = _as_torch(_asset(env, asset_cfg).data.root_quat_w)[:n]
    _, _, current_yaw = _quat_to_euler(quat)
    yaw_err = wrap_to_pi(target_yaw - current_yaw).abs()
    reached = (dist < distance_threshold) & (yaw_err < yaw_threshold)
    return reached.float().unsqueeze(-1)


def neighbor_state_b(
    env, asset_cfg, agent_ids: list[str], max_neighbors: int, radius: float, mask_key: str
) -> torch.Tensor:
    """Per-agent neighbor features: relative pos (3) + relative vel (3), shape (num_envs, max_neighbors * 6).

    Positions and velocities are interleaved per-agent so the encoder can
    reshape to (B, max_neighbors, 6) tokens.
    """
    asset = _asset(env, asset_cfg)
    ego_pos = _as_torch(asset.data.root_pos_w)
    ego_vel = _as_torch(asset.data.root_lin_vel_w)
    ego_quat = _as_torch(asset.data.root_quat_w)
    mask = _active_mask(env, agent_ids, mask_key)
    current_index = agent_ids.index(asset_cfg.name)

    all_pos = _all_root_pos(env, agent_ids)
    all_vel = _all_root_lin_vel(env, agent_ids)
    rel_pos_w = all_pos - ego_pos.unsqueeze(1)
    rel_vel_w = all_vel - ego_vel.unsqueeze(1)
    rel_pos_b = _rotate_world_to_body(ego_quat, rel_pos_w)
    rel_vel_b = _rotate_world_to_body(ego_quat, rel_vel_w)

    valid = (torch.norm(rel_pos_w, dim=-1) < radius) & mask.bool()
    valid[:, current_index] = False
    keep = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    keep[current_index] = False

    features = torch.cat([rel_pos_b[:, keep, :], rel_vel_b[:, keep, :]], dim=-1)
    return _pad_features(features, max_neighbors, mask=valid[:, keep])


def relative_neighbor_positions_b(
    env, asset_cfg, agent_ids: list[str], max_neighbors: int, radius: float, mask_key: str
) -> torch.Tensor:
    """Relative positions of neighbors in body frame [m], shape (num_envs, max_neighbors * 3)."""
    asset = _asset(env, asset_cfg)
    ego_pos = _as_torch(asset.data.root_pos_w)
    ego_quat = _as_torch(asset.data.root_quat_w)
    mask = _active_mask(env, agent_ids, mask_key)
    current_index = agent_ids.index(asset_cfg.name)
    all_pos = _all_root_pos(env, agent_ids)
    rel_pos_w = all_pos - ego_pos.unsqueeze(1)
    rel_pos_b = _rotate_world_to_body(ego_quat, rel_pos_w)
    valid = (torch.norm(rel_pos_w, dim=-1) < radius) & mask.bool()
    valid[:, current_index] = False
    keep = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    keep[current_index] = False
    return _pad_features(rel_pos_b[:, keep, :], max_neighbors, mask=valid[:, keep])


def relative_neighbor_velocities_b(
    env, asset_cfg, agent_ids: list[str], max_neighbors: int, radius: float, mask_key: str
) -> torch.Tensor:
    """Relative velocities of neighbors in body frame [m/s], shape (num_envs, max_neighbors * 3)."""
    asset = _asset(env, asset_cfg)
    ego_pos = _as_torch(asset.data.root_pos_w)
    ego_vel = _as_torch(asset.data.root_lin_vel_w)
    ego_quat = _as_torch(asset.data.root_quat_w)
    mask = _active_mask(env, agent_ids, mask_key)
    current_index = agent_ids.index(asset_cfg.name)
    all_pos = _all_root_pos(env, agent_ids)
    all_vel = _all_root_lin_vel(env, agent_ids)
    rel_pos_w = all_pos - ego_pos.unsqueeze(1)
    rel_vel_w = all_vel - ego_vel.unsqueeze(1)
    rel_vel_b = _rotate_world_to_body(ego_quat, rel_vel_w)
    valid = (torch.norm(rel_pos_w, dim=-1) < radius) & mask.bool()
    valid[:, current_index] = False
    keep = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    keep[current_index] = False
    return _pad_features(rel_vel_b[:, keep, :], max_neighbors, mask=valid[:, keep])


# TODO(cpsquare) - this is a copy of what is in the cameron_swarm observations.py
# we should refactor out a shared version and put in mdp/common/
def static_sdf(
    env, asset_cfg, column_positions_key: str = "column_positions", grid_size: int = 3, grid_resolution: float = 0.1
) -> torch.Tensor:
    """3x3 signed distance grid from drone position to nearest static obstacles [m], shape (num_envs, 9).

    The SDF is sampled at grid points around the drone's XY position. No obstacles
    means the SDF is clamped to a large positive distance.
    """
    asset = _asset(env, asset_cfg)
    drone_pos = _as_torch(asset.data.root_pos_w)

    columns = getattr(_root_env(env), column_positions_key, None)
    if columns is None:
        return 20.0 * torch.ones(drone_pos.shape[0], grid_size * grid_size, device=env.device)

    num_envs = drone_pos.shape[0]
    max_dist = 20.0
    if columns.shape[1] == 0:
        return torch.full((num_envs, grid_size * grid_size), max_dist, device=env.device)

    half = (grid_size - 1) / 2.0
    offsets = torch.tensor(
        [
            [(i - half) * grid_resolution, (j - half) * grid_resolution]
            for i in range(grid_size)
            for j in range(grid_size)
        ],
        device=env.device,
        dtype=drone_pos.dtype,
    )
    sample_xy = drone_pos[:, None, :2] + offsets[None, :, :]
    dist = torch.linalg.norm(sample_xy[:, :, None, :] - columns[:, None, :, :2], dim=-1)
    return torch.clamp(dist.min(dim=-1).values, max=max_dist)


def agent_active_flag(env, agent_ids: list[str], agent_id: str, mask_key: str) -> torch.Tensor:
    """Binary: is this agent active? Shape (num_envs, 1)."""
    mask = _active_mask(env, agent_ids, mask_key)
    try:
        agent_index = agent_ids.index(agent_id)
    except ValueError:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return mask[:, agent_index].float().unsqueeze(-1)


def swarm_global_state(
    env,
    agent_ids: list[str],
    command_name: str,
    include_root_state: bool,
    include_target_pose: bool,
    include_pairwise_distances: bool,
    mask_key: str,
) -> torch.Tensor:
    """Centralized critic state: active mask + root states + target poses + pairwise distances.

    Shape: (num_envs, state_dim).
    """
    mask = _active_mask(env, agent_ids, mask_key)
    num_envs = env.num_envs
    parts = [mask.float()]

    if include_root_state:
        root_states = []
        for agent_id in agent_ids:
            asset = _root_env(env).scene[agent_id].data
            root_states.append(
                torch.cat(
                    [
                        _as_torch(asset.root_pos_w),
                        _as_torch(asset.root_quat_w),
                        _as_torch(asset.root_lin_vel_w),
                        _as_torch(asset.root_ang_vel_w),
                    ],
                    dim=-1,
                )
            )
        parts.append(torch.stack(root_states, dim=1).reshape(num_envs, -1))

    if include_target_pose:
        command = env.command_manager.get_command(command_name)
        parts.append(command.reshape(num_envs, -1))

    if include_pairwise_distances:
        positions = torch.stack([_root_pos(env, aid) for aid in agent_ids], dim=1)
        dists = torch.cdist(positions, positions)
        active_mask = mask.float().unsqueeze(-1) * mask.float().unsqueeze(-2)
        dists = dists * active_mask
        parts.append(dists.reshape(num_envs, -1))

    return torch.cat(parts, dim=-1)


def paper_swarm_global_state(
    env,
    agent_ids: list[str],
    command_name: str,
    mask_key: str,
) -> torch.Tensor:
    """Centralized MAPPO state computed once for the whole swarm.

    Returns:
        Tensor containing active mask, root state, target pose commands, and
        pairwise distances, shape ``(num_envs, 232)`` for the default 8-drone
        task.
    """
    root = _root_env(env)
    mask = _active_mask(root, agent_ids, mask_key).float()
    positions = _all_root_pos(root, agent_ids)

    root_states = []
    commands = []
    for agent_id in agent_ids:
        asset = root.scene[agent_id].data
        root_states.append(
            torch.cat(
                [
                    _as_torch(asset.root_pos_w),
                    _as_torch(asset.root_quat_w),
                    _as_torch(asset.root_lin_vel_w),
                    _as_torch(asset.root_ang_vel_w),
                ],
                dim=-1,
            )
        )
        bundle = root._manager_bundles[root._agent_to_bundle[agent_id]]
        commands.append(bundle.command_manager.get_command(command_name))

    active_pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    pairwise_distances = torch.cdist(positions, positions) * active_pair_mask
    return torch.cat(
        [
            mask,
            torch.stack(root_states, dim=1).reshape(root.num_envs, -1),
            torch.stack(commands, dim=1).reshape(root.num_envs, -1),
            pairwise_distances.reshape(root.num_envs, -1),
        ],
        dim=-1,
    )


def paper_swarm_default_global_state(env) -> torch.Tensor:
    """Default centralized MAPPO state for the paper swarm task."""
    return paper_swarm_global_state(
        env,
        agent_ids=[f"drone_{index}" for index in range(8)],
        command_name="target_pose",
        mask_key="active_drones",
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _quat_to_euler(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion [w,x,y,z] to roll, pitch, yaw [rad]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1.0, torch.sign(sinp) * (torch.pi / 2.0), torch.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def command_decompose(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract position, yaw, and sin/cos from a pose command."""
    cmd = env.command_manager.get_command(command_name)
    pos = cmd[:, :3]
    _, _, yaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]
    return pos, yaw, torch.stack([torch.sin(cmd[:, 5]), torch.cos(cmd[:, 5])], dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate [w, -x, -y, -z], shape (..., 4)."""
    qc = q.clone()
    qc[..., 1:] *= -1
    return qc
