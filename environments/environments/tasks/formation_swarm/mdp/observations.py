# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Observation functions for the Xie et al. formation swarm task.

Observations match the paper specification:

:math:`D_\\text{obs} = D_\\text{self} + (N-1)D_\\text{other} + B D_\\text{dynamic} + D_\\text{static}`.

Each term function receives the ``_AgentGroupEnvView`` proxy and returns a
per-agent tensor of shape ``(num_envs, dim)``.
"""

from __future__ import annotations

import torch

# Re-export built-in observation terms used by the task config.
from isaaclab.envs.mdp import base_ang_vel as root_ang_vel_b  # noqa: F401
from isaaclab.envs.mdp import base_lin_vel as root_lin_vel_b  # noqa: F401
from isaaclab.utils.math import quat_apply

_EGO_AGENT_ID_ATTR = "_formation_ego_agent_id"


def _root_env(env):
    return env.root if hasattr(env, "root") else env


def _ego_agent_id(env) -> str:
    root = _root_env(env)
    if hasattr(root, _EGO_AGENT_ID_ATTR):
        return getattr(root, _EGO_AGENT_ID_ATTR)
    return env.agent_ids[0]


def _all_agent_ids(env) -> list[str]:
    return getattr(_root_env(env).cfg, "possible_agents", [])


def _all_asset_names(env) -> tuple[str, ...]:
    root = _root_env(env)
    asset_names = getattr(root, "_formation_asset_names", None)
    if asset_names is not None:
        return tuple(asset_names)
    ma_spec = getattr(root, "ma_spec", None)
    if ma_spec is not None:
        return tuple(ma_spec.agents[agent_id].asset_name for agent_id in root.possible_agents)
    return tuple(_all_agent_ids(env))


def _num_drones(env) -> int:
    return getattr(_root_env(env).cfg, "num_drones", len(_all_agent_ids(env)))


def _num_balls(env) -> int:
    return getattr(_root_env(env).cfg, "num_balls", 0)


def _asset(env, asset_cfg):
    return _root_env(env).scene[asset_cfg.name]


# -----------------------------------------------------------------------------
# Caching helpers for per-step observation computation
# -----------------------------------------------------------------------------

_CACHE_ATTR = "_formation_obs_step_cache"
_CACHE_STEP_ATTR = "_formation_obs_step_cached_at"


def _get_cached(env, key: str, factory, *args) -> torch.Tensor:
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


def _all_positions(env) -> torch.Tensor:
    """Stack all agent positions relative to env origin [m], shape (num_envs, num_drones, 3)."""
    root = _root_env(env)
    asset_names = _all_asset_names(env)
    return _get_cached(
        env,
        "all_pos",
        lambda: torch.stack(
            [root.scene[asset_name].data.root_pos_w.torch - root.scene.env_origins for asset_name in asset_names],
            dim=1,
        ),
    )


def _all_velocities(env) -> torch.Tensor:
    """Stack all agent linear velocities [m/s], shape (num_envs, num_drones, 3)."""
    root = _root_env(env)
    asset_names = _all_asset_names(env)
    return _get_cached(
        env,
        "all_vel",
        lambda: torch.stack([root.scene[asset_name].data.root_lin_vel_w.torch for asset_name in asset_names], dim=1),
    )


def _all_quats(env) -> torch.Tensor:
    """Stack all agent quaternions [x, y, z, w], shape (num_envs, num_drones, 4)."""
    root = _root_env(env)
    asset_names = _all_asset_names(env)
    return _get_cached(
        env,
        "all_quat",
        lambda: torch.stack([root.scene[asset_name].data.root_quat_w.torch for asset_name in asset_names], dim=1),
    )


def _all_omegas(env) -> torch.Tensor:
    """Stack all agent angular velocities [rad/s], shape (num_envs, num_drones, 3)."""
    root = _root_env(env)
    asset_names = _all_asset_names(env)
    return _get_cached(
        env,
        "all_omega",
        lambda: torch.stack([root.scene[asset_name].data.root_ang_vel_w.torch for asset_name in asset_names], dim=1),
    )


# -----------------------------------------------------------------------------
# Observation term functions
# -----------------------------------------------------------------------------


def formation_ego_pos(env, asset_cfg) -> torch.Tensor:
    """Ego position relative to env origin [m], shape (num_envs, 3)."""
    return _asset(env, asset_cfg).data.root_pos_w.torch - _root_env(env).scene.env_origins


def formation_ego_quat(env, asset_cfg) -> torch.Tensor:
    """Ego quaternion [x, y, z, w], shape (num_envs, 4)."""
    return _asset(env, asset_cfg).data.root_quat_w.torch


def formation_ego_heading(env, asset_cfg) -> torch.Tensor:
    """Ego heading direction (body x-axis in world frame), shape (num_envs, 3)."""
    quat = _asset(env, asset_cfg).data.root_quat_w.torch
    fwd = torch.tensor([1.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3)
    return quat_apply(quat, fwd)


def formation_ego_up(env, asset_cfg) -> torch.Tensor:
    """Ego up direction (body z-axis in world frame), shape (num_envs, 3)."""
    quat = _asset(env, asset_cfg).data.root_quat_w.torch
    up = torch.tensor([0.0, 0.0, 1.0], device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3)
    return quat_apply(quat, up)


def formation_last_action(env, action_name: str = "ctbr") -> torch.Tensor:
    """Last action reordered from CTBR [wx,wy,wz,c] to paper [c,wx,wy,wz], shape (num_envs, 4)."""
    action = env.action_manager.get_term(action_name).raw_actions
    return torch.cat((action[:, 3:4], action[:, :3]), dim=-1)


def formation_target_vel_rel(env, asset_cfg) -> torch.Tensor:
    """Relative velocity: target_vel minus ego velocity [m/s], shape (num_envs, 3)."""
    root = _root_env(env)
    target_vel = getattr(root, "_formation_target_vel", None)
    if target_vel is None:
        return torch.zeros(_asset(env, asset_cfg).data.root_lin_vel_w.torch.shape, device=root.device)
    return target_vel.view(1, 3) - _asset(env, asset_cfg).data.root_lin_vel_w.torch


def drone_identity(env, agent_id: str) -> torch.Tensor:
    """One-hot identity tensor, shape (num_envs, num_drones)."""
    root = _root_env(env)
    agent_ids = root.cfg.possible_agents
    identity = torch.eye(len(agent_ids), device=root.device, dtype=torch.float32)
    index = agent_ids.index(agent_id)
    return identity[index].unsqueeze(0).expand(root.num_envs, -1)


def formation_other_drone_obs(env, asset_cfg) -> torch.Tensor:
    r"""Relative position, distance, and velocity to all other drones.

    For ego drone :math:`i` and neighbor :math:`j \neq i`:

    .. math::

        \left[\frac{p_i-p_j}{2},\; \frac{\lVert p_i-p_j\rVert_2}{2},\; v_i-v_j\right].

    Returns shape ``(num_envs, (num_drones - 1) * 7)``.
    """
    root = _root_env(env)
    agent_ids = root.cfg.possible_agents
    positions = _all_positions(env)
    velocities = _all_velocities(env)
    ego_index = agent_ids.index(_ego_agent_id(env))

    relative_pos = positions[:, ego_index : ego_index + 1, :] - positions
    relative_vel = velocities[:, ego_index : ego_index + 1, :] - velocities

    mask = torch.ones(len(agent_ids), device=root.device, dtype=torch.bool)
    mask[ego_index] = False
    other_pos = relative_pos[:, mask, :]
    other_vel = relative_vel[:, mask, :]
    other_dist = torch.linalg.norm(other_pos, dim=-1, keepdim=True)
    return torch.cat((other_pos / 2.0, other_dist / 2.0, other_vel), dim=-1).reshape(root.num_envs, -1)


def formation_ball_obs(env, asset_cfg) -> torch.Tensor:
    """Per-agent dynamic obstacle observation for all balls.

    Returns shape ``(num_envs, num_balls * 10)``.
    Inactive balls are masked to zero.
    """
    root = _root_env(env)
    num_balls = _num_balls(env)

    if num_balls == 0:
        return torch.zeros(root.num_envs, 0, device=root.device)

    ego_pos = _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins
    balls_pos = getattr(root, "_formation_ball_positions", None)
    balls_vel = getattr(root, "_formation_ball_velocities", None)
    ball_active = getattr(root, "_formation_ball_active", None)

    if balls_pos is None:
        balls_pos = torch.zeros(root.num_envs, num_balls, 3, device=root.device)
    if balls_vel is None:
        balls_vel = torch.zeros_like(balls_pos)
    if ball_active is None:
        ball_active = torch.zeros(root.num_envs, num_balls, device=root.device, dtype=torch.bool)

    ego_vel = _asset(env, asset_cfg).data.root_lin_vel_w.torch
    rel_pos = balls_pos - ego_pos.unsqueeze(1)
    rel_vel = balls_vel - ego_vel.unsqueeze(1)
    ball_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)

    obs = torch.cat(
        (
            ball_dist,
            rel_pos,
            rel_vel,
            balls_vel,
        ),
        dim=-1,
    )
    inactive = ~ball_active.unsqueeze(-1)
    obs = obs.masked_fill(inactive, 0.0)

    return obs.reshape(root.num_envs, -1)


def formation_static_sdf(env, asset_cfg) -> torch.Tensor:
    """3x3 grid-based signed distance field to static columns.

    For each of 9 grid offsets around the drone, returns the minimum XY distance
    to any active column. Returns shape ``(num_envs, static_sdf_dim)``.
    """
    root = _root_env(env)

    ego_pos = _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins
    columns = getattr(root, "_formation_column_positions", None)
    active_static = getattr(root, "_formation_active_static_obstacles", 0)

    if active_static == 0 or columns is None or columns.shape[1] == 0:
        return torch.zeros(root.num_envs, 9, device=root.device)

    grid_offsets = getattr(root, "_formation_grid_offsets", None)
    if grid_offsets is None:
        grid_offsets = torch.tensor(
            [
                [-0.1, -0.1],
                [-0.1, 0.0],
                [-0.1, 0.1],
                [0.0, -0.1],
                [0.0, 0.0],
                [0.0, 0.1],
                [0.1, -0.1],
                [0.1, 0.0],
                [0.1, 0.1],
            ],
            device=root.device,
            dtype=torch.float32,
        )

    grid = ego_pos[:, :2].unsqueeze(1) + grid_offsets.view(1, 9, 2)
    rel = columns[:, :active_static, None, :2] - grid[:, None, :, :]
    return torch.linalg.norm(rel, dim=-1).amin(dim=1)
