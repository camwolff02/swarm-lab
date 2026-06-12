# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for the Xie et al. formation swarm task."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import time_out  # noqa: F401


def _root_env(env):
    return env.root if hasattr(env, "root") else env


def _asset(env, asset_cfg):
    return _root_env(env).scene[asset_cfg.name]


def _all_agent_ids(env) -> list[str]:
    return _root_env(env).cfg.possible_agents


def _all_asset_names(env) -> tuple[str, ...]:
    root = _root_env(env)
    asset_names = getattr(root, "_formation_asset_names", None)
    if asset_names is not None:
        return tuple(asset_names)
    ma_spec = getattr(root, "ma_spec", None)
    if ma_spec is not None:
        return tuple(ma_spec.agents[agent_id].asset_name for agent_id in root.possible_agents)
    return tuple(_all_agent_ids(env))


def drone_crash(env, asset_cfg, agent_id: str) -> torch.Tensor:
    """True when the drone is below crash_min_height or above crash_max_height.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    pos = _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins
    crash_min = getattr(root.cfg, "crash_min_height", 0.2)
    crash_max = getattr(root.cfg, "crash_max_height", 2.8)
    return (pos[:, 2] < crash_min) | (pos[:, 2] > crash_max)


def drone_too_close(env, asset_cfg, agent_id: str) -> torch.Tensor:
    """True when any pair of drones is closer than hard_safe_distance.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    agent_ids = _all_agent_ids(env)
    asset_names = _all_asset_names(env)
    positions = torch.stack(
        [root.scene[asset_name].data.root_pos_w.torch - root.scene.env_origins for asset_name in asset_names], dim=1
    )
    pairwise = torch.cdist(positions, positions)
    eye = torch.eye(len(agent_ids), device=root.device, dtype=torch.bool)
    pairwise = pairwise.masked_fill(eye, torch.inf)
    min_dist = pairwise.amin(dim=-1)
    ego_idx = agent_ids.index(agent_id)
    hard_safe = getattr(root.cfg, "hard_safe_distance", 0.15)
    return min_dist[:, ego_idx] < hard_safe


def drone_hit_ball(env, asset_cfg, agent_id: str) -> torch.Tensor:
    """True when the drone is within ball_radius of any active ball.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    num_balls = getattr(root.cfg, "num_balls", 0)
    if num_balls == 0:
        return torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)

    balls_pos = getattr(root, "_formation_ball_positions", None)
    ball_active = getattr(root, "_formation_ball_active", None)
    if balls_pos is None or ball_active is None:
        return torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)

    pos = _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins
    ball_dist = torch.linalg.norm(balls_pos - pos.unsqueeze(1), dim=-1)
    ball_radius = getattr(root.cfg, "ball_radius", 0.15)
    return ((ball_dist < ball_radius) & ball_active).any(dim=-1)


def drone_hit_column(env, asset_cfg, agent_id: str) -> torch.Tensor:
    """True when the drone is within column_radius of any active static column.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    active_static = getattr(root, "_formation_active_static_obstacles", 0)
    if active_static == 0:
        return torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)

    columns = getattr(root, "_formation_column_positions", None)
    if columns is None:
        return torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)

    pos = _asset(env, asset_cfg).data.root_pos_w.torch - root.scene.env_origins
    col_dist = torch.linalg.norm(
        (columns[:, :active_static] - pos.unsqueeze(1))[:, :, :2], dim=-1
    )
    column_radius = getattr(root.cfg, "column_radius", 0.15)
    return (col_dist < column_radius).any(dim=-1)
