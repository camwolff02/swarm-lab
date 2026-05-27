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
)
from isaaclab.envs.mdp import (
    base_lin_vel as root_lin_vel_b,  # noqa: F401
)
from isaaclab.envs.mdp import (
    last_action,  # noqa: F401
)
from isaaclab.envs.mdp import (
    projected_gravity as projected_gravity_b,  # noqa: F401
)
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_apply_inverse, wrap_to_pi


def _root_env(env) -> object:
    """Return the root env when the active view is a scoped proxy."""
    if hasattr(env, "root"):
        return env.root
    return env


_CACHE_ATTR = "_paper_swarm_obs_step_cache"
_CACHE_STEP_ATTR = "_paper_swarm_obs_step_cached_at"


def _get_cached(env, key: str, factory, *args) -> torch.Tensor:
    """Return a per-step cached tensor, computing lazily once per step.

    The cache lives on the root environment instance so it is scoped to one
    environment and cleaned up when the env is garbage-collected.  It is
    invalidated automatically whenever ``common_step_counter`` advances.
    """
    root = _root_env(env)
    step = int(root.common_step_counter)
    cache = getattr(root, _CACHE_ATTR, None)
    cached_at = getattr(root, _CACHE_STEP_ATTR, None)
    if cache is None or cached_at != step:
        cache = {}
        setattr(root, _CACHE_ATTR, cache)
        setattr(root, _CACHE_STEP_ATTR, step)
    if key not in cache:
        cache[key] = factory(*args)
    return cache[key]


def _asset(env, asset_cfg) -> object:
    """Look up an articulation asset from a SceneEntityCfg."""
    return _root_env(env).scene[asset_cfg.name]


def _root_pos(env, agent_id: str) -> torch.Tensor:
    """World position of an articulation relative to the env origin [m], shape (num_envs, 3)."""
    root = _root_env(env)
    return root.scene[agent_id].data.root_pos_w.torch - root.scene.env_origins


def _root_quat(env, agent_id: str) -> torch.Tensor:
    """World quaternion of an articulation [x, y, z, w], shape (num_envs, 4)."""
    return _root_env(env).scene[agent_id].data.root_quat_w.torch


def _root_lin_vel(env, agent_id: str) -> torch.Tensor:
    """World linear velocity of an articulation [m/s], shape (num_envs, 3)."""
    return _root_env(env).scene[agent_id].data.root_lin_vel_w.torch


def _all_root_pos(env, agent_ids: list[str]) -> torch.Tensor:
    """Stack all agent root positions [m], shape (num_envs, num_agents, 3)."""
    return _get_cached(env, "all_pos", _all_root_pos_uncached, env, agent_ids)


def _all_root_pos_uncached(env, agent_ids: list[str]) -> torch.Tensor:
    return torch.stack([_root_pos(env, agent_id) for agent_id in agent_ids], dim=1)


def _all_root_lin_vel(env, agent_ids: list[str]) -> torch.Tensor:
    """Stack all agent root velocities [m/s], shape (num_envs, num_agents, 3)."""
    return _get_cached(env, "all_vel", _all_root_lin_vel_uncached, env, agent_ids)


def _all_root_lin_vel_uncached(env, agent_ids: list[str]) -> torch.Tensor:
    return torch.stack([_root_lin_vel(env, agent_id) for agent_id in agent_ids], dim=1)


def _active_mask(env, agent_ids: list[str], mask_key: str) -> torch.Tensor:
    """Bool tensor (num_envs, len(agent_ids)) indicating active agents."""
    root = _root_env(env)
    if mask_key is None or not hasattr(root, mask_key):
        return torch.ones((root.num_envs, len(agent_ids)), dtype=torch.bool, device=root.device)
    return getattr(root, mask_key)


def get_agent_active_mask(env, agent_id: str, mask_key: str) -> torch.Tensor:
    """Float mask for a single agent, shape ``(num_envs,)``.

    This is the canonical helper shared by reward, termination, and observation
    terms.  It returns a float tensor so callers can multiply reward/penalty
    values without an extra cast.
    """
    root = _root_env(env)
    agent_ids = root.cfg.possible_agents
    mask = _active_mask(env, agent_ids, mask_key)
    try:
        index = agent_ids.index(agent_id)
    except ValueError:
        return torch.ones(root.num_envs, device=root.device)
    return mask[:, index].float()


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


def _nearest_neighbor_features(
    features: torch.Tensor,
    distances: torch.Tensor,
    valid: torch.Tensor,
    max_neighbors: int,
) -> torch.Tensor:
    """Select nearest valid neighbor features and pad to a fixed flat shape.

    Args:
        features: Candidate neighbor features, shape ``(num_envs, num_candidates, feat_dim)``.
        distances: Candidate XY/XYZ distances [m], shape ``(num_envs, num_candidates)``.
        valid: Candidate validity mask, shape ``(num_envs, num_candidates)``.
        max_neighbors: Number of output neighbor slots.

    Returns:
        Flattened nearest-neighbor features, shape ``(num_envs, max_neighbors * feat_dim)``.
    """
    num_envs, num_candidates, feat_dim = features.shape
    if num_candidates == 0 or max_neighbors == 0:
        return torch.zeros(num_envs, max_neighbors * feat_dim, device=features.device, dtype=features.dtype)

    masked_distances = distances.masked_fill(~valid, torch.inf)
    order = torch.argsort(masked_distances, dim=1)
    sorted_features = torch.gather(features, 1, order.unsqueeze(-1).expand(-1, -1, feat_dim))
    sorted_valid = torch.gather(valid, 1, order)

    slots = min(num_candidates, max_neighbors)
    out = torch.zeros(num_envs, max_neighbors, feat_dim, device=features.device, dtype=features.dtype)
    out[:, :slots, :] = sorted_features[:, :slots, :] * sorted_valid[:, :slots].unsqueeze(-1).to(features.dtype)
    return out.reshape(num_envs, max_neighbors * feat_dim)


def _rotate_world_to_body(quat_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the body frame."""
    if vectors_w.ndim == 2:
        return quat_apply_inverse(quat_w, vectors_w)
    quat = quat_w.unsqueeze(1).expand(-1, vectors_w.shape[1], -1)
    rotated = quat_apply_inverse(quat.reshape(-1, 4), vectors_w.reshape(-1, 3))
    return rotated.reshape_as(vectors_w)


# -----------------------------------------------------------------------------
# Observation functions
# -----------------------------------------------------------------------------


def root_pos(env, asset_cfg) -> torch.Tensor:
    """Position relative to the environment origin [m], shape (num_envs, 3)."""
    root = _root_env(env)
    return _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins


def root_rotation_matrix(env, asset_cfg) -> torch.Tensor:
    """Flattened world-frame rotation matrix (3x3), shape (num_envs, 9).

    Converted from the root quaternion [x, y, z, w] so the policy sees a
    linear orientation representation instead of a quaternion.
    """
    q = _asset(env, asset_cfg).data.root_quat_w.torch
    return matrix_from_quat(q).reshape(q.shape[0], 9)


def relative_target_position_b(env, asset_cfg, command_name: str) -> torch.Tensor:
    """Target waypoint position relative to drone, in body frame [m], shape (num_envs, 3)."""
    target_pos_w = env.command_manager.get_command(command_name)[:, :3] + _root_env(env).scene.env_origins
    current_pos_w = _asset(env, asset_cfg).data.root_pos_w.torch[: target_pos_w.shape[0]]
    quat_w = _asset(env, asset_cfg).data.root_quat_w.torch[: target_pos_w.shape[0]]
    return _rotate_world_to_body(quat_w, target_pos_w - current_pos_w)


def target_yaw_error(env, asset_cfg, command_name: str) -> torch.Tensor:
    """Yaw error to target [rad], wrapped to [-pi, pi], shape (num_envs, 1)."""
    command = env.command_manager.get_command(command_name)
    _, _, target_yaw = euler_xyz_from_quat(command[:, 3:7])
    quat = _asset(env, asset_cfg).data.root_quat_w.torch[: command.shape[0]]
    _, _, current_yaw = euler_xyz_from_quat(quat)
    error = wrap_to_pi(target_yaw - current_yaw)
    return error.unsqueeze(-1)


def distance_to_goal(env, asset_cfg, command_name: str) -> torch.Tensor:
    """Euclidean distance to goal waypoint [m], shape (num_envs, 1)."""
    target_pos = env.command_manager.get_command(command_name)[:, :3] + _root_env(env).scene.env_origins
    current_pos = _asset(env, asset_cfg).data.root_pos_w.torch[: target_pos.shape[0]]
    return torch.norm(target_pos - current_pos, dim=-1, keepdim=True)


def goal_reached_flag(
    env, asset_cfg, command_name: str, distance_threshold: float, yaw_threshold: float
) -> torch.Tensor:
    """Binary flag: is the goal reached? Shape (num_envs, 1)."""
    target_pos = env.command_manager.get_command(command_name)[:, :3] + _root_env(env).scene.env_origins
    n = target_pos.shape[0]
    current_pos = _asset(env, asset_cfg).data.root_pos_w.torch[:n]
    dist = torch.norm(target_pos - current_pos, dim=-1)

    _, target_yaw, _ = command_decompose(env, command_name)
    quat = _asset(env, asset_cfg).data.root_quat_w.torch[:n]
    _, _, current_yaw = euler_xyz_from_quat(quat)
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
    ego_pos = _root_pos(env, asset_cfg.name)
    ego_vel = asset.data.root_lin_vel_w.torch
    ego_quat = asset.data.root_quat_w.torch
    mask = _active_mask(env, agent_ids, mask_key)
    current_index = agent_ids.index(asset_cfg.name)

    all_pos = _all_root_pos(env, agent_ids)
    all_vel = _all_root_lin_vel(env, agent_ids)
    rel_pos_w = all_pos - ego_pos.unsqueeze(1)
    rel_vel_w = all_vel - ego_vel.unsqueeze(1)
    rel_pos_b = _rotate_world_to_body(ego_quat, rel_pos_w)
    rel_vel_b = _rotate_world_to_body(ego_quat, rel_vel_w)

    distances = torch.norm(rel_pos_w, dim=-1)
    valid = (distances < radius) & mask.bool()
    valid[:, current_index] = False
    keep = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    keep[current_index] = False

    features = torch.cat([rel_pos_b[:, keep, :], rel_vel_b[:, keep, :]], dim=-1)
    return _nearest_neighbor_features(features, distances[:, keep], valid[:, keep], max_neighbors)


def relative_neighbor_positions_b(
    env, asset_cfg, agent_ids: list[str], max_neighbors: int, radius: float, mask_key: str
) -> torch.Tensor:
    """Relative positions of neighbors in body frame [m], shape (num_envs, max_neighbors * 3)."""
    asset = _asset(env, asset_cfg)
    ego_pos = _root_pos(env, asset_cfg.name)
    ego_quat = asset.data.root_quat_w.torch
    mask = _active_mask(env, agent_ids, mask_key)
    current_index = agent_ids.index(asset_cfg.name)
    all_pos = _all_root_pos(env, agent_ids)
    rel_pos_w = all_pos - ego_pos.unsqueeze(1)
    rel_pos_b = _rotate_world_to_body(ego_quat, rel_pos_w)
    distances = torch.norm(rel_pos_w, dim=-1)
    valid = (distances < radius) & mask.bool()
    valid[:, current_index] = False
    keep = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    keep[current_index] = False
    return _nearest_neighbor_features(rel_pos_b[:, keep, :], distances[:, keep], valid[:, keep], max_neighbors)


def relative_neighbor_velocities_b(
    env, asset_cfg, agent_ids: list[str], max_neighbors: int, radius: float, mask_key: str
) -> torch.Tensor:
    """Relative velocities of neighbors in body frame [m/s], shape (num_envs, max_neighbors * 3)."""
    asset = _asset(env, asset_cfg)
    ego_pos = _root_pos(env, asset_cfg.name)
    ego_vel = asset.data.root_lin_vel_w.torch
    ego_quat = asset.data.root_quat_w.torch
    mask = _active_mask(env, agent_ids, mask_key)
    current_index = agent_ids.index(asset_cfg.name)
    all_pos = _all_root_pos(env, agent_ids)
    all_vel = _all_root_lin_vel(env, agent_ids)
    rel_pos_w = all_pos - ego_pos.unsqueeze(1)
    rel_vel_w = all_vel - ego_vel.unsqueeze(1)
    rel_vel_b = _rotate_world_to_body(ego_quat, rel_vel_w)
    distances = torch.norm(rel_pos_w, dim=-1)
    valid = (distances < radius) & mask.bool()
    valid[:, current_index] = False
    keep = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    keep[current_index] = False
    return _nearest_neighbor_features(rel_vel_b[:, keep, :], distances[:, keep], valid[:, keep], max_neighbors)


# TODO(cpsquare) - this is a copy of what is in the cameron_swarm observations.py
# we should refactor out a shared version and put in mdp/common/
def static_sdf(
    env,
    asset_cfg,
    column_positions_key: str = "column_positions",
    grid_size: int = 3,
    grid_resolution: float = 0.1,
    column_radius: float = 0.0,
) -> torch.Tensor:
    """3x3 signed distance grid from drone position to nearest static obstacles [m], shape (num_envs, 9).

    SDF = distance_to_obstacle_surface = distance_to_center - column_radius.
    No obstacles means the SDF is clamped to a large positive distance.
    """
    asset = _asset(env, asset_cfg)
    root = _root_env(env)
    drone_pos = asset.data.root_pos_w.torch - root.scene.env_origins

    columns = getattr(root, column_positions_key, None)
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
    dist = torch.linalg.norm(sample_xy[:, :, None, :] - columns[:, None, :, :2], dim=-1) - column_radius
    return torch.clamp(dist.min(dim=-1).values, max=max_dist)


def drone_identity(env, agent_ids: list[str], agent_id: str) -> torch.Tensor:
    """One-hot drone identity, shape (num_envs, len(agent_ids))."""
    identity = torch.eye(len(agent_ids), device=env.device)[agent_ids.index(agent_id)]
    return identity.unsqueeze(0).expand(env.num_envs, -1)


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
                        asset.root_pos_w.torch - _root_env(env).scene.env_origins,
                        asset.root_quat_w.torch,
                        asset.root_lin_vel_w.torch,
                        asset.root_ang_vel_w.torch,
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
    column_positions_key: str,
    max_static_columns: int,
) -> torch.Tensor:
    """Centralized MAPPO state computed once for the whole swarm.

    Works with any number of managed agents.  Unmanaged drones contribute
    physics state (they are still simulated) and zero commands so the critic
    always receives the same fixed-dimension observation regardless of how
    many agents are actively controlled by the RL pipeline.

    Returns:
        Tensor containing active mask, root state, target pose commands,
        pairwise distances, obstacle positions, and obstacle mask.
    """
    root = _root_env(env)

    mask = _active_mask(root, agent_ids, mask_key).float()
    positions = _all_root_pos(root, agent_ids)

    agent_to_bundle: dict[str, str] = getattr(root, "_agent_to_bundle", {})
    root_states = []
    commands = []
    for agent_id in agent_ids:
        asset = root.scene[agent_id].data
        root_states.append(
            torch.cat(
                [
                    asset.root_pos_w.torch - root.scene.env_origins,
                    asset.root_quat_w.torch,
                    asset.root_lin_vel_w.torch,
                    asset.root_ang_vel_w.torch,
                ],
                dim=-1,
            )
        )
        if agent_id in agent_to_bundle:
            bundle = root._manager_bundles[agent_to_bundle[agent_id]]
            commands.append(bundle.command_manager.get_command(command_name))
        else:
            commands.append(torch.zeros(root.num_envs, 7, device=root.device))

    active_pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    pairwise_distances = torch.cdist(positions, positions) * active_pair_mask
    columns = getattr(root, column_positions_key, None)
    if columns is None or columns.shape[1] == 0:
        column_positions = torch.zeros(root.num_envs, max_static_columns, 3, device=root.device)
        column_mask = torch.zeros(root.num_envs, max_static_columns, device=root.device)
    else:
        column_positions = torch.zeros(root.num_envs, max_static_columns, 3, device=root.device)
        column_mask = torch.zeros(root.num_envs, max_static_columns, device=root.device)
        count = min(max_static_columns, columns.shape[1])
        active_columns = torch.linalg.norm(columns[:, :count, :2], dim=-1) < 100.0
        column_positions[:, :count] = columns[:, :count]
        column_positions[:, :count] = torch.where(
            active_columns.unsqueeze(-1), column_positions[:, :count], torch.zeros_like(column_positions[:, :count])
        )
        column_mask[:, :count] = active_columns.float()
    return torch.cat(
        [
            mask,
            torch.stack(root_states, dim=1).reshape(root.num_envs, -1),
            torch.stack(commands, dim=1).reshape(root.num_envs, -1),
            pairwise_distances.reshape(root.num_envs, -1),
            column_positions.reshape(root.num_envs, -1),
            column_mask,
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
        column_positions_key="column_positions",
        max_static_columns=10,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def command_decompose(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract position, yaw, and sin/cos from a pose command."""
    cmd = env.command_manager.get_command(command_name)
    pos = cmd[:, :3]
    _, _, yaw = euler_xyz_from_quat(cmd[:, 3:7])
    return pos, yaw, torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
