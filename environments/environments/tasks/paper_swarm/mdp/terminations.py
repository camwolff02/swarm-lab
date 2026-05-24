# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for the paper_swarm waypoint-navigation task."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import time_out  # noqa: F401


def drone_out_of_bounds(
    env, asset_cfg, agent_id: str, xy_bounds: tuple[float, float], z_bounds: tuple[float, float], mask_key: str
) -> torch.Tensor:
    """Drone position exceeds workspace bounds.

    Args:
        xy_bounds: (min, max) XY limits [m].
        z_bounds: (min, max) Z limits [m].

    Returns:
        Bool tensor, shape (num_envs,). True if out of bounds.
    """
    from .observations import _as_torch

    asset = env.scene[asset_cfg.name]
    pos = _as_torch(asset.data.root_pos_w)
    out_of_xy = (torch.abs(pos[:, 0]) > xy_bounds[1]) | (torch.abs(pos[:, 1]) > xy_bounds[1])
    out_of_z = (pos[:, 2] < z_bounds[0]) | (pos[:, 2] > z_bounds[1])
    mask = _get_active_mask(env, agent_id, mask_key)
    return (out_of_xy | out_of_z) & (mask > 0)


def drone_pairwise_collision(
    env, asset_cfg, agent_id: str, agent_ids: list[str], collision_distance: float, mask_key: str
) -> torch.Tensor:
    """True if drone is within collision_distance of any other active drone.

    Args:
        collision_distance: Threshold distance [m] for collision.

    Returns:
        Bool tensor, shape (num_envs,).
    """
    from .observations import _all_root_pos, _as_torch

    ego_pos = _as_torch(env.scene[asset_cfg.name].data.root_pos_w)
    mask = _get_active_mask(env, agent_id, mask_key)
    if mask.sum() == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    current_index = agent_ids.index(asset_cfg.name)
    all_pos = _all_root_pos(env, agent_ids)
    valid = torch.ones(len(agent_ids), dtype=torch.bool, device=ego_pos.device)
    valid[current_index] = False
    dist = torch.linalg.norm(all_pos[:, valid, :] - ego_pos.unsqueeze(1), dim=-1)
    collision = (dist < collision_distance).any(dim=-1)

    return collision & (mask > 0)


def _get_active_mask(env, agent_id: str, mask_key: str) -> torch.Tensor:
    """Get the active mask for a specific agent as a float vector, shape (num_envs,)."""
    from .observations import _active_mask

    root = env.root if hasattr(env, "root") else env
    agent_ids = root.cfg.possible_agents
    mask = _active_mask(env, agent_ids, mask_key)
    try:
        index = agent_ids.index(agent_id)
    except ValueError:
        return torch.ones(env.num_envs, device=env.device)
    return mask[:, index].float()
