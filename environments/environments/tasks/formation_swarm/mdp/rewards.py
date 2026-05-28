# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Reward functions for the Xie et al. formation swarm task.

The task uses a multi-objective reward decomposition:

.. math::

    r = w_s r_\text{smooth} + w_o r_\text{obstacle}
      + w_f r_\text{forward} + w_\ell r_\text{formation}.

Each MORL group is a weighted sum of sub-terms, computed once per step and cached
to avoid redundant computation across reward terms.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply


def _root_env(env):
    return env.root if hasattr(env, "root") else env


def _asset(env, asset_cfg):
    return _root_env(env).scene[asset_cfg.name]


def _all_agent_ids(env) -> list[str]:
    return _root_env(env).cfg.possible_agents


# -----------------------------------------------------------------------------
# Caching helper
# -----------------------------------------------------------------------------

_REWARD_CACHE_ATTR = "_formation_reward_cache"
_REWARD_CACHE_STEP_ATTR = "_formation_reward_step"


def _reward_components(env):
    """Compute and cache all reward sub-components once per step."""
    root = _root_env(env)
    step = int(root.common_step_counter)
    cache = getattr(root, _REWARD_CACHE_ATTR, None)
    cached_step = getattr(root, _REWARD_CACHE_STEP_ATTR, None)
    if cache is not None and cached_step == step:
        return cache

    cfg = root.cfg
    num_drones = cfg.num_drones
    num_balls = cfg.num_balls
    agent_ids = _all_agent_ids(env)

    positions = torch.stack(
        [root.scene[agent_id].data.root_pos_w.torch - root.scene.env_origins for agent_id in agent_ids], dim=1
    )
    velocities = torch.stack(
        [root.scene[agent_id].data.root_lin_vel_w.torch for agent_id in agent_ids], dim=1
    )
    quats = torch.stack(
        [root.scene[agent_id].data.root_quat_w.torch for agent_id in agent_ids], dim=1
    )
    omegas = torch.stack(
        [root.scene[agent_id].data.root_ang_vel_w.torch for agent_id in agent_ids], dim=1
    )

    heading = quat_apply(
        quats.reshape(-1, 4),
        torch.tensor([1.0, 0.0, 0.0], device=root.device).expand(positions.numel() // 3, 3),
    ).reshape(root.num_envs, num_drones, 3)

    target_pos = getattr(root, "_formation_target_pos", torch.zeros(3, device=root.device))
    target_vel = getattr(root, "_formation_target_vel", torch.zeros(3, device=root.device))
    target_heading = getattr(root, "_formation_target_heading", torch.zeros(3, device=root.device))
    formation_l = getattr(root, "_formation_l", None)
    formation_l_unnorm = getattr(root, "_formation_l_unnormalized", None)
    standard_formation_size = getattr(root, "_standard_formation_size", torch.tensor(1.0, device=root.device))
    safe_distance = getattr(cfg, "safe_distance", 0.4)
    hard_safe_distance = getattr(cfg, "hard_safe_distance", 0.15)
    obs_safe_distance = getattr(cfg, "obs_safe_distance", 0.4)
    soft_obs_safe_distance = getattr(cfg, "soft_obs_safe_distance", 0.6)
    column_radius = getattr(cfg, "column_radius", 0.15)

    # -- formation Laplacian costs --
    if formation_l is not None:
        distances = torch.cdist(positions, positions)
        degree = distances.sum(dim=-1)
        scale = degree.clamp_min(1.0e-6).pow(-0.5)
        adj = scale.unsqueeze(-1) * distances * scale.unsqueeze(-2)
        eye = torch.eye(num_drones, device=root.device, dtype=positions.dtype)
        current_l = eye.expand_as(adj) - adj
        cost_l = torch.linalg.matrix_norm(formation_l.unsqueeze(0) - current_l, dim=(-2, -1)).unsqueeze(-1)
    else:
        cost_l = torch.zeros(root.num_envs, 1, device=root.device)

    if formation_l_unnorm is not None:
        distances = torch.cdist(positions, positions)
        degree = distances.sum(dim=-1)
        cost_l_unnorm = torch.linalg.matrix_norm(
            (degree.unsqueeze(-1) - distances) - formation_l_unnorm.unsqueeze(0), dim=(-2, -1)
        ).unsqueeze(-1)
    else:
        cost_l_unnorm = torch.zeros(root.num_envs, 1, device=root.device)

    # -- size and separation --
    pairwise = torch.cdist(positions, positions)
    size = pairwise.amax(dim=(1, 2)).unsqueeze(-1)
    eye_mask = torch.eye(num_drones, device=root.device, dtype=torch.bool)
    pairwise_no_self = pairwise.masked_fill(eye_mask, torch.inf)
    separation = pairwise_no_self.amin(dim=-1)
    too_close = separation < hard_safe_distance

    # -- formation rewards --
    reward_formation = (
        1.0 / (1.0 + torch.square((cost_l - 0.04) / num_drones * 10.0)) - (0.04 * num_drones)
    ) / 2.0
    reward_size = 1.0 / (1.0 + torch.square(size - standard_formation_size))
    reward_size = reward_size + 1.0 / (1.0 + cost_l_unnorm)
    reward_size = ((reward_size - 2.0) / num_drones - (0.04 * num_drones)) * 3.0 + 2.36
    separation_reward = -(separation < safe_distance).float()
    too_close_reward = too_close.float()

    # -- forward rewards --
    heading_error = torch.linalg.norm(target_heading.view(1, 1, 3) - heading, dim=-1)
    reward_heading = torch.clamp(1.0 - heading_error, min=0.0)

    final_pos = getattr(root, "_formation_final_pos", target_pos)
    formation_offsets = getattr(root, "_formation_offsets", None)
    if formation_offsets is None:
        formation_offsets = torch.zeros(num_drones, 3, device=root.device)
    pos_error = torch.linalg.norm(
        positions - (final_pos.view(1, 1, 3) + formation_offsets.view(1, num_drones, 3)), dim=-1
    )
    pos_reward = 1.0 / (1.0 + pos_error)
    vel_error = velocities - target_vel.view(1, 1, 3)
    vel_reward = torch.clamp(
        torch.clamp(torch.linalg.norm(target_vel), min=1.0) - torch.linalg.norm(vel_error, dim=-1), min=0.0
    )
    height_reward = torch.clamp(1.0 - (positions[..., 2] - target_pos[2]).abs(), min=0.0)

    # -- obstacle rewards --
    active_static = getattr(root, "_formation_active_static_obstacles", 0)
    columns = getattr(root, "_formation_column_positions", None)
    ball_hit_distance = getattr(cfg, "ball_radius", 0.15)

    if num_balls > 0:
        balls_pos = getattr(root, "_formation_ball_positions", None)
        balls_vel = getattr(root, "_formation_ball_velocities", None)
        ball_active = getattr(root, "_formation_ball_active", None)
        ball_launched = getattr(root, "_formation_ball_launched", None)
        active_balls = getattr(root, "_formation_active_balls", num_balls)

        if balls_pos is None:
            balls_pos = torch.zeros(root.num_envs, num_balls, 3, device=root.device)
        if balls_vel is None:
            balls_vel = torch.zeros_like(balls_pos)
        if ball_active is None:
            ball_active = torch.zeros(root.num_envs, num_balls, device=root.device, dtype=torch.bool)
        if ball_launched is None:
            ball_launched = torch.zeros_like(ball_active)

        rel_ball_pos = balls_pos.unsqueeze(1) - positions.unsqueeze(2)
        ball_dist = torch.linalg.norm(rel_ball_pos, dim=-1)
        active_ball = ball_active.unsqueeze(1)
        ball_hard = torch.zeros_like(ball_dist)

        ball_hard_reward_coeff = getattr(cfg, "ball_hard_reward_coeff", 100.0)
        ball_reward_coeff = getattr(cfg, "ball_reward_coeff", 10.0)
        ratio = ball_hard_reward_coeff / ball_reward_coeff
        ball_hard_coeff = ratio if ball_reward_coeff > 0 else 10.0

        ball_hard[ball_dist < obs_safe_distance] = -ball_hard_coeff
        k = 0.5 * ball_hard_coeff / (soft_obs_safe_distance - obs_safe_distance + 1e-6)
        ball_soft = (
            ball_dist.clamp(obs_safe_distance, soft_obs_safe_distance) - soft_obs_safe_distance
        ) * k
        ball_soft = ball_soft + (ball_dist - soft_obs_safe_distance).clamp_min(0.0)
        ball_reward = ((ball_hard + ball_soft) * active_ball).amin(dim=-1)

        ball_any_mask = active_ball.any(dim=-1)
        launched_active = (
            ball_launched[:, :active_balls].all(dim=-1, keepdim=True)
            if active_balls > 0
            else torch.zeros(root.num_envs, 1, device=root.device, dtype=torch.bool)
        )
        after_throw_mask = (~ball_any_mask) & launched_active
        hit_ball = ((ball_dist < ball_hit_distance) & active_ball).any(dim=-1)
    else:
        ball_any_mask = torch.zeros(root.num_envs, 1, device=root.device, dtype=torch.bool)
        after_throw_mask = torch.zeros_like(ball_any_mask)
        ball_reward = torch.zeros(root.num_envs, num_drones, device=root.device)
        hit_ball = torch.zeros(root.num_envs, num_drones, device=root.device, dtype=torch.bool)
        active_balls = 0

    if active_static > 0 and columns is not None:
        rel_col = columns[:, :active_static].unsqueeze(1) - positions.unsqueeze(2)
        col_dist = torch.linalg.norm(rel_col[..., :2], dim=-1)
        cube_reward = (
            col_dist.clamp(column_radius, obs_safe_distance) - obs_safe_distance
        ).mean(dim=-1)
        hit_column = (col_dist < column_radius).any(dim=-1)
        use_cube_mask = getattr(cfg, "use_cube_reward_mask", False)
        if use_cube_mask:
            column_near = (col_dist < (soft_obs_safe_distance + 1.0)).any(dim=(1, 2)).unsqueeze(-1)
        else:
            column_near = torch.zeros(root.num_envs, 1, device=root.device, dtype=torch.bool)
    else:
        cube_reward = torch.zeros(root.num_envs, num_drones, device=root.device)
        hit_column = torch.zeros_like(hit_ball)
        column_near = torch.zeros(root.num_envs, 1, device=root.device, dtype=torch.bool)

    hit_reward = (hit_ball | hit_column).float()
    crash = (positions[..., 2] < getattr(cfg, "crash_min_height", 0.2)) | (
        positions[..., 2] > getattr(cfg, "crash_max_height", 2.8)
    )
    bad_terminate = crash | too_close | hit_ball | hit_column
    bad_env = bad_terminate.any(dim=-1, keepdim=True)

    max_ep_length = root.max_episode_length if hasattr(root, "max_episode_length") else 450
    time_out = root.episode_length_buf >= max_ep_length - 1
    truncated = time_out.view(-1, 1)

    # -- smoothness rewards --
    last_actions = getattr(root, "_formation_last_actions", None)
    previous_action_features = getattr(root, "_formation_previous_action_features", None)
    current_action_features = getattr(root, "_formation_current_action_features", None)
    previous_actions_global = getattr(root, "_formation_previous_actions", last_actions)

    if last_actions is None:
        last_actions = torch.zeros(root.num_envs, num_drones, 4, device=root.device)
    if previous_action_features is None:
        previous_action_features = last_actions.clone()
    if current_action_features is None:
        current_action_features = last_actions.clone()
    if previous_actions_global is None:
        previous_actions_global = last_actions.clone()

    collective_ratio = ((current_action_features[..., 0] + 1.0) * 0.5).clamp(0.0, 1.0)
    action_dim = getattr(cfg, "action_dim", 4)
    effort = torch.clamp(2.5 - action_dim * collective_ratio, min=0.0)
    throttle_diff = torch.abs(current_action_features[..., 0] - previous_action_features[..., 0])
    throttle_smooth = torch.clamp(0.5 - throttle_diff, min=0.0)
    action_diff = torch.linalg.norm(last_actions - previous_actions_global, dim=-1)
    action_smooth = torch.clamp(2.5 - action_diff, min=0.0)
    spin = torch.clamp(1.5 - omegas[..., 2].abs(), min=0.0)

    # -- MORL coefficients --
    has_obstacle_coeff = getattr(cfg, "has_obstacle_coeff", 0.2)
    no_obstacle_coeff = getattr(cfg, "no_obstacle_coeff", 1.0)
    obstacle_present = ball_any_mask | column_near
    coeff = torch.where(
        obstacle_present,
        torch.full_like(obstacle_present, has_obstacle_coeff, dtype=torch.float32),
        torch.full_like(obstacle_present, no_obstacle_coeff, dtype=torch.float32),
    )
    after_throw_coeff = getattr(cfg, "after_throw_coeff", 0.2)
    truncated_reward = getattr(cfg, "truncated_reward", 10.0)

    components = {
        "cost_l": cost_l,
        "cost_l_unnorm": cost_l_unnorm,
        "reward_formation": reward_formation,
        "reward_size": reward_size,
        "separation_reward": separation_reward,
        "too_close_reward": too_close_reward,
        "reward_heading": reward_heading,
        "pos_reward": pos_reward,
        "vel_reward": vel_reward,
        "height_reward": height_reward,
        "ball_reward": ball_reward,
        "cube_reward": cube_reward,
        "hit_reward": hit_reward,
        "effort": effort,
        "action_smooth": action_smooth,
        "spin": spin,
        "throttle_smooth": throttle_smooth,
        "bad_env": bad_env,
        "truncated": truncated,
        "coeff": coeff,
        "after_throw_mask": after_throw_mask,
        "obstacle_present": obstacle_present,
        "ball_any_mask": ball_any_mask,
        "column_near": column_near,
        "after_throw_coeff": after_throw_coeff,
        "truncated_reward": truncated_reward,
    }
    setattr(root, _REWARD_CACHE_ATTR, components)
    setattr(root, _REWARD_CACHE_STEP_ATTR, step)
    return components


# -----------------------------------------------------------------------------
# Reward term functions
# -----------------------------------------------------------------------------


def formation_smooth_reward(env, asset_cfg) -> torch.Tensor:
    """MORL smoothness reward: effort + action smoothness + spin + throttle smoothness.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    cfg = root.cfg
    comp = _reward_components(env)

    truncated_reward = comp["truncated_reward"]
    morl_smooth = (
        comp["effort"] * getattr(cfg, "effort_weight", 0.5)
        + comp["action_smooth"] * getattr(cfg, "action_smoothness_weight", 1.0)
        + comp["spin"] * getattr(cfg, "spin_reward_coeff", 1.0)
        + comp["throttle_smooth"] * getattr(cfg, "throttle_smoothness_weight", 2.0)
        + comp["truncated"] * truncated_reward
        - comp["bad_env"] * truncated_reward
    )
    return morl_smooth.squeeze(-1) if morl_smooth.ndim == 2 and morl_smooth.shape[-1] == 1 else morl_smooth.mean(dim=-1)


def formation_obstacle_reward(env, asset_cfg) -> torch.Tensor:
    """MORL obstacle reward: ball + column + hit penalty + truncated reward.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    cfg = root.cfg
    comp = _reward_components(env)

    truncated_reward = comp["truncated_reward"]
    morl_obstacle = (
        comp["ball_reward"] * getattr(cfg, "ball_reward_coeff", 10.0)
        + comp["cube_reward"] * getattr(cfg, "static_hard_coeff", 1.0)
        + comp["hit_reward"] * getattr(cfg, "hit_penalty", -20.0)
        + comp["truncated"] * truncated_reward
        - comp["bad_env"] * truncated_reward
    )
    return morl_obstacle.squeeze(-1) if morl_obstacle.ndim == 2 and morl_obstacle.shape[-1] == 1 else morl_obstacle.mean(dim=-1)


def formation_formation_reward(env, asset_cfg) -> torch.Tensor:
    """MORL formation reward: size alignment + Laplacian formation + separation penalties.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    cfg = root.cfg
    comp = _reward_components(env)

    formation_size_coeff = getattr(cfg, "formation_size_coeff", 5.0)
    formation_coeff = getattr(cfg, "formation_coeff", 5.0)
    separation_coeff = getattr(cfg, "separation_coeff", 1.0)
    too_close_penalty = getattr(cfg, "too_close_penalty", -10.0)
    truncated_reward = comp["truncated_reward"]

    morl_formation = (
        (comp["reward_size"] * comp["coeff"] + comp["reward_size"] * comp["after_throw_mask"] * comp["after_throw_coeff"])
        * formation_size_coeff
        + comp["reward_formation"] * formation_coeff * comp["coeff"]
        + comp["separation_reward"] * separation_coeff
        + comp["too_close_reward"] * too_close_penalty
        + comp["truncated"] * truncated_reward
        - comp["bad_env"] * truncated_reward
    )
    return morl_formation.squeeze(-1) if morl_formation.ndim == 2 and morl_formation.shape[-1] == 1 else morl_formation.mean(dim=-1)


def formation_forward_reward(env, asset_cfg) -> torch.Tensor:
    """MORL forward reward: height + position + velocity + heading.

    Returns shape ``(num_envs,)``.
    """
    root = _root_env(env)
    cfg = root.cfg
    comp = _reward_components(env)

    height_coeff = getattr(cfg, "height_coeff", 5.0)
    position_reward_coeff = getattr(cfg, "position_reward_coeff", 50.0)
    velocity_coeff = getattr(cfg, "velocity_coeff", 10.0)
    heading_coeff = getattr(cfg, "heading_coeff", 1.0)
    truncated_reward = comp["truncated_reward"]

    morl_forward = (
        comp["height_reward"] * height_coeff * comp["coeff"]
        + comp["pos_reward"] * position_reward_coeff * comp["truncated"]
        + comp["vel_reward"] * velocity_coeff * comp["coeff"]
        + comp["reward_heading"] * heading_coeff
    ) * comp["coeff"]
    morl_forward = morl_forward + comp["truncated"] * truncated_reward - comp["bad_env"] * truncated_reward
    return morl_forward.squeeze(-1) if morl_forward.ndim == 2 and morl_forward.shape[-1] == 1 else morl_forward.mean(dim=-1)
