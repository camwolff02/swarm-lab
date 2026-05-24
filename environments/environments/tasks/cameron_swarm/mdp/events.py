# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for resetting the multi-drone waypoint task."""

from __future__ import annotations

import math

import torch


def reset_drone_root_state_uniform(
    env,
    env_ids,
    agent_ids,
    xy_bounds,
    z_bounds,
    min_separation,
    lin_vel_range,
    ang_vel_range,
    mask_key: str,
):
    """Reset agent root states with uniform positions [m] respecting min_separation [m] and zero orientation."""
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    positions = _sample_separated_positions(env, env_ids, len(agent_ids), xy_bounds, z_bounds, min_separation)
    mask = getattr(env, mask_key, torch.ones((env.num_envs, len(agent_ids)), dtype=torch.bool, device=env.device))

    for index, agent_id in enumerate(agent_ids):
        asset = env.scene[agent_id]
        root_pose = asset.data.default_root_pose.torch[env_ids].clone()
        pos = positions[:, index] + env.scene.env_origins[env_ids]
        inactive = ~mask[env_ids, index]
        if inactive.any():
            pos[inactive] = env.scene.env_origins[env_ids[inactive]] + torch.tensor((0.0, 0.0, -10.0), device=env.device)
        root_pose[:, :3] = pos
        root_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
        lin_vel = torch.empty((len(env_ids), 3), device=env.device).uniform_(*lin_vel_range)
        ang_vel = torch.empty((len(env_ids), 3), device=env.device).uniform_(*ang_vel_range)
        velocity = torch.cat((lin_vel, ang_vel), dim=-1)
        velocity[inactive] = 0.0
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def _sample_separated_positions(env, env_ids, num_agents: int, xy_bounds, z_bounds, min_separation: float) -> torch.Tensor:
    positions = torch.zeros((len(env_ids), num_agents, 3), device=env.device)
    low = torch.tensor((xy_bounds[0], xy_bounds[0], z_bounds[0]), device=env.device)
    high = torch.tensor((xy_bounds[1], xy_bounds[1], z_bounds[1]), device=env.device)
    for env_local in range(len(env_ids)):
        chosen = []
        for _ in range(num_agents):
            for _attempt in range(128):
                sample = low + torch.rand(3, device=env.device) * (high - low)
                if not chosen:
                    break
                distances = torch.linalg.norm(torch.stack(chosen)[:, :2] - sample[:2], dim=-1)
                if bool((distances >= min_separation).all()):
                    break
            chosen.append(sample)
        positions[env_local] = torch.stack(chosen)
    if not torch.isfinite(positions).all():
        raise RuntimeError("Non-finite drone reset position sampled.")
    return positions
