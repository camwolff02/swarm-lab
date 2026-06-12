# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event functions for the Xie et al. formation swarm task."""

from __future__ import annotations

import torch


def _root_env(env):
    return env.root if hasattr(env, "root") else env


def sample_static_columns(env, env_ids: torch.Tensor) -> None:
    """Sample static column positions on a staggered grid aligned with target velocity.

    Stores the result in ``_formation_column_positions`` on the root env.
    """
    root = _root_env(env)
    env_ids = torch.as_tensor(env_ids, device=root.device, dtype=torch.long)
    cfg = root.cfg

    active_static = getattr(root, "_formation_active_static_obstacles", 0)
    total_static = getattr(cfg, "static_obstacles", 0)
    columns = getattr(root, "_formation_column_positions", None)
    if columns is None or columns.shape[1] != total_static:
        columns = torch.zeros(root.num_envs, total_static, 3, device=root.device)

    columns[env_ids] = torch.tensor((0.0, 0.0, -10.0), device=root.device)

    if active_static == 0:
        setattr(root, "_formation_column_positions", columns)
        return

    target_vel = getattr(root, "_formation_target_vel", torch.zeros(3, device=root.device))
    target_speed = torch.linalg.norm(target_vel[:2]).clamp_min(1.0)

    grid_size = getattr(cfg, "grid_size", 0.5)
    grid_border = getattr(cfg, "grid_border", 2.0)
    static_margin = getattr(cfg, "static_margin", 2.0)
    episode_length_s = getattr(cfg, "episode_length_s", 9.0)

    length = float(target_speed * episode_length_s) - 2.0 * static_margin
    cols = int((2.0 * grid_border) // grid_size)
    rows = max(int(length // grid_size), 1)
    total = rows * cols

    n_env = len(env_ids)
    random_values = torch.randint(total, (n_env, total), device=root.device)
    indices = torch.argsort(random_values, dim=-1)[:, :active_static]
    grid_a = indices // cols
    grid_b = indices % cols

    x0 = grid_a.float() * grid_size + static_margin
    y0 = grid_b.float() * grid_size - grid_border
    y0 = torch.where((grid_a % 2) == 0, y0 + grid_size / 2.0, y0)

    sin_theta = target_vel[1] / target_speed
    cos_theta = target_vel[0] / target_speed
    x = x0 * cos_theta - y0 * sin_theta
    y = x0 * sin_theta + y0 * cos_theta
    z = torch.zeros_like(x)

    columns[env_ids, :active_static] = torch.stack((x, y, z), dim=-1)
    setattr(root, "_formation_column_positions", columns)


def reset_swarm_root_state(env, env_ids: torch.Tensor) -> None:
    """Reset drone positions to formation-relative starting positions with hover actions.

    Writes root state to sim for all agents, resets ball state, and seeds hover
    actions via ``_set_hover_actions`` on the root env.
    """
    root = _root_env(env)
    env_ids = torch.as_tensor(env_ids, device=root.device, dtype=torch.long)
    cfg = root.cfg

    target_pos = getattr(root, "_formation_target_pos", torch.zeros(3, device=root.device))
    formation_offsets = getattr(root, "_formation_offsets", None)
    if formation_offsets is None:
        formation_offsets = getattr(cfg, "formation", None)
        formation_size = getattr(cfg, "formation_size", 1.0)
        if formation_offsets is not None:
            formation_offsets = torch.tensor(formation_offsets, device=root.device, dtype=torch.float32) * formation_size
        else:
            num_drones = getattr(cfg, "num_drones", 3)
            formation_offsets = torch.zeros(num_drones, 3, device=root.device)

    positions = target_pos.view(1, 1, 3) + formation_offsets.view(1, -1, 3)
    positions = positions.expand(len(env_ids), -1, -1)

    if hasattr(root, "_write_swarm_state"):
        root._write_swarm_state(
            env_ids, positions, None,
            torch.zeros_like(positions), torch.zeros_like(positions),
        )

    if hasattr(root, "_set_hover_actions"):
        root._set_hover_actions(env_ids)

    if hasattr(root, "_formation_current_action_features"):
        root._formation_current_action_features[env_ids] = 0.0
        root._formation_previous_action_features[env_ids] = 0.0
        root._formation_last_actions[env_ids] = 0.0
        if hasattr(root, "_formation_previous_actions"):
            root._formation_previous_actions[env_ids] = 0.0


def reset_balls(env, env_ids: torch.Tensor) -> None:
    """Reset ball state for the given environment indices.

    Inactive balls are moved below ground. Active ball launch times are sampled
    according to the throw threshold and time range.
    """
    root = _root_env(env)
    env_ids = torch.as_tensor(env_ids, device=root.device, dtype=torch.long)
    cfg = root.cfg
    num_balls = getattr(cfg, "num_balls", 0)
    max_ep_length = root.max_episode_length if hasattr(root, "max_episode_length") else 450

    ball_active = getattr(root, "_formation_ball_active", None)
    ball_launched = getattr(root, "_formation_ball_launched", None)
    balls_pos = getattr(root, "_formation_ball_positions", None)
    balls_vel = getattr(root, "_formation_ball_velocities", None)
    ball_launch_step = getattr(root, "_formation_ball_launch_step", None)

    if ball_active is None:
        ball_active = torch.zeros(root.num_envs, num_balls, device=root.device, dtype=torch.bool)
    if ball_launched is None:
        ball_launched = torch.zeros_like(ball_active)
    if balls_pos is None:
        balls_pos = torch.zeros(root.num_envs, num_balls, 3, device=root.device)
    if balls_vel is None:
        balls_vel = torch.zeros_like(balls_pos)
    if ball_launch_step is None:
        ball_launch_step = torch.zeros(root.num_envs, num_balls, device=root.device, dtype=torch.long)

    ball_active[env_ids] = False
    ball_launched[env_ids] = False
    balls_pos[env_ids] = torch.tensor((0.0, 0.0, -10.0), device=root.device)
    balls_vel[env_ids] = 0.0
    ball_launch_step[env_ids] = max_ep_length + 1

    active_balls = getattr(root, "_formation_active_balls", num_balls)
    if active_balls > 0:
        throw_threshold = getattr(root, "_formation_throw_threshold_steps", 150)
        throw_range = getattr(root, "_formation_throw_time_range_steps", 450)
        launch_offsets = torch.rand(len(env_ids), active_balls, device=root.device) * throw_range
        ball_launch_step[env_ids, :active_balls] = (throw_threshold + launch_offsets).long()

    setattr(root, "_formation_ball_active", ball_active)
    setattr(root, "_formation_ball_launched", ball_launched)
    setattr(root, "_formation_ball_positions", balls_pos)
    setattr(root, "_formation_ball_velocities", balls_vel)
    setattr(root, "_formation_ball_launch_step", ball_launch_step)
