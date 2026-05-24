# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the drone waypoint MARL task."""

from __future__ import annotations

import torch


def _active(env, agent_ids, agent_id: str, mask_key: str) -> torch.Tensor:
    if not hasattr(env, mask_key):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return getattr(env, mask_key)[:, agent_ids.index(agent_id)]


def time_out(env):
    """True when episode length [steps] exceeds max_episode_length, shape [num_envs], bool."""
    return env.episode_length_buf >= env.max_episode_length


def drone_out_of_bounds(env, asset_cfg, agent_id: str, xy_bounds, z_bounds, mask_key: str):
    """True when drone position [m] is outside XY or Z bounds, shape [num_envs], bool."""
    pos = env.scene[asset_cfg.name].data.root_pos_w.torch - env.scene.env_origins
    xy_bad = (pos[:, 0] < xy_bounds[0]) | (pos[:, 0] > xy_bounds[1]) | (pos[:, 1] < xy_bounds[0]) | (pos[:, 1] > xy_bounds[1])
    z_bad = (pos[:, 2] < z_bounds[0]) | (pos[:, 2] > z_bounds[1])
    return (xy_bad | z_bad) & _active(env, env.possible_agents, agent_id, mask_key)


def drone_pairwise_collision(env, asset_cfg, agent_id: str, agent_ids, collision_distance: float, mask_key: str):
    """True when distance [m] to any active neighbor is below collision_distance, shape [num_envs], bool."""
    ego_pos = env.scene[asset_cfg.name].data.root_pos_w.torch - env.scene.env_origins
    mask = getattr(env, mask_key, torch.ones((env.num_envs, len(agent_ids)), dtype=torch.bool, device=env.device))
    collided = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for other_index, other_id in enumerate(agent_ids):
        if other_id == agent_id:
            continue
        other_pos = env.scene[other_id].data.root_pos_w.torch - env.scene.env_origins
        other_active = mask[:, other_index]
        collided |= (torch.linalg.norm(other_pos - ego_pos, dim=-1) < collision_distance) & other_active
    return collided & _active(env, agent_ids, agent_id, mask_key)
