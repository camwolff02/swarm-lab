# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and event functions for the paper_swarm waypoint-navigation task.

Spawns drones at random separated positions, places static column obstacles,
and hides inactive agents.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import sample_uniform


def reset_drone_root_state_uniform(
    env,
    env_ids: torch.Tensor,
    agent_ids: list[str],
    xy_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    min_separation: float,
    lin_vel_range: tuple[float, float],
    ang_vel_range: tuple[float, float],
    mask_key: str,
) -> dict[str, torch.Tensor]:
    """Sample separated positions and set zero-rotation hover start state.

    Inactive agents are placed at the environment origin, resting on the ground plane.

    Args:
        xy_bounds: (min, max) [m] for each axis.
        z_bounds: (min, max) [m].
        min_separation: Minimum XY distance between any pair [m].
        lin_vel_range: (min, max) [m/s].
        ang_vel_range: (min, max) [rad/s].
        mask_key: Key for active agent mask attribute.

    Returns:
        dict mapping agent_id to root_state tensor (num_envs, 13).
    """
    from .commands import _sample_positions_vectorized
    from .observations import _active_mask

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    mask = _active_mask(env, agent_ids, mask_key)
    resets = {}
    root = env.root if hasattr(env, "root") else env
    spawn_safe_prob = getattr(root, "_paper_swarm_spawn_safe_sampling_prob", 1.0)
    spawn_min_separation = getattr(root, "_paper_swarm_spawn_min_separation", min_separation)
    column_radius = getattr(root, "_paper_swarm_column_radius", 0.15)
    column_safe_distance = getattr(root, "_paper_swarm_column_safe_distance", 0.6)
    columns = getattr(root, "column_positions", None)

    sampled_positions, _ = _sample_positions_vectorized(
        active_mask=mask[env_ids],
        xy_bounds=xy_bounds,
        z_bounds=z_bounds,
        min_separation=spawn_min_separation,
        safe_prob=spawn_safe_prob,
        columns=columns[env_ids] if columns is not None else None,
        column_radius=column_radius,
        column_safe_distance=column_safe_distance,
        device=env.device,
    )

    for i, agent_id in enumerate(agent_ids):
        is_active = mask[env_ids, i]
        asset = env.scene[agent_id]
        root_pose = asset.data.default_root_pose.torch[env_ids].clone()
        root_velocity = asset.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] = env.scene.env_origins[env_ids] + torch.tensor((0.0, 0.0, 0.05), device=env.device)
        root_pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        root_velocity[:, :] = 0.0

        if is_active.any():
            active_count = int(is_active.sum().item())
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(active_count, 1)
            lin_vel = sample_uniform(
                lin_vel_range[0], lin_vel_range[1], (active_count, 3), device=env.device
            )
            ang_vel = sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (active_count, 3), device=env.device
            )

            root_pose[is_active, :3] = sampled_positions[is_active, i] + env.scene.env_origins[env_ids[is_active]]
            root_pose[is_active, 3:7] = quat
            root_velocity[is_active, :3] = lin_vel
            root_velocity[is_active, 3:6] = ang_vel

        # Spread inactive drones in a grid so they don't clip into each other
        GRID_SPACING = 1.0
        GRID_COLS = 3
        if i > 0:
            gx = ((i - 1) % GRID_COLS) * GRID_SPACING
            gy = ((i - 1) // GRID_COLS) * GRID_SPACING
            root_pose[~is_active, :3] = (
                env.scene.env_origins[env_ids[~is_active]] + torch.tensor((gx, gy, 0.05), device=env.device)
            )

        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        resets[agent_id] = torch.cat([root_pose, root_velocity], dim=-1)

    return resets


def sample_static_columns(
    env,
    env_ids: torch.Tensor,
    num_columns: int,
    grid_size: float,
    grid_border: float,
    margin: float,
    height: float,
    column_radius: float,
    column_positions_key: str = "column_positions",
) -> None:
    """Sample static column positions using a zigzag grid.

    Columns are placed in a forward-facing grid aligned with the environment's
    forward direction. The grid is centered at the origin.

    Column positions are stored on the environment attribute named by
    *column_positions_key*.  Inactive columns (outside the workspace) receive
    ``(1000, 1000, 0)``.

    Args:
        num_columns: Number of columns to place.
        grid_size: Cell size [m].
        grid_border: Width of the grid in both directions [m].
        margin: Margin from the start/end of the grid [m].
        height: Column height [m].
        column_radius: Column radius [m].
        column_positions_key: Attribute name to store column positions on env.
    """
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    num_envs = len(env_ids)
    device = env.device

    root = env.root if hasattr(env, "root") else env
    current_columns = int(getattr(root, "_paper_swarm_num_static_columns", num_columns))
    current_columns = max(0, min(num_columns, current_columns))

    rows = int(2 * grid_border / grid_size)
    cols = int(2 * grid_border / grid_size) if num_columns > 0 else 0
    total_cells = rows * cols

    if total_cells == 0:
        setattr(env, column_positions_key, torch.zeros(env.num_envs, 0, 3, device=device))
        return

    cell_centers = _get_or_build_cell_centers(env, rows, cols, grid_size, grid_border, device)

    all_positions = getattr(env, column_positions_key, None)
    if all_positions is None or all_positions.shape != (env.num_envs, num_columns, 3):
        all_positions = torch.zeros(env.num_envs, num_columns, 3, device=device)

    positions = torch.zeros(num_envs, num_columns, 3, device=device)
    positions[:, :, 0] = 1000.0
    positions[:, :, 1] = 1000.0

    if current_columns > 0:
        perms = torch.stack([torch.randperm(total_cells, device=device)[:current_columns] for _ in range(num_envs)])
        positions[:, :current_columns, :2] = cell_centers[perms]
        positions[:, :current_columns, 2] = 0.0

    all_positions[env_ids] = positions
    setattr(env, column_positions_key, all_positions)
    root.extras["static_column_count"] = current_columns


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


_CELL_CENTER_CACHE_KEY = "_paper_swarm_cell_centers_cache"


def _get_or_build_cell_centers(
    env, rows: int, cols: int, grid_size: float, grid_border: float, device: torch.device
) -> torch.Tensor:
    """Return precomputed cell centre XY positions, cached on the root env."""
    root = env.root if hasattr(env, "root") else env
    cache_key = (_CELL_CENTER_CACHE_KEY, rows, cols, grid_size, grid_border)
    cache = getattr(root, _CELL_CENTER_CACHE_KEY, None)
    if cache is not None and isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key]

    total_cells = rows * cols
    cell_centers = torch.zeros(total_cells, 2, device=device)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            cell_centers[idx, 0] = -grid_border + col * grid_size
            cell_centers[idx, 1] = -grid_border + row * grid_size
            idx += 1

    if cache is None:
        cache = {}
    cache[cache_key] = cell_centers
    setattr(root, _CELL_CENTER_CACHE_KEY, cache)
    return cell_centers


def reset_drone_hover_thrust(
    env,
    env_ids: torch.Tensor,
    agent_ids: list[str],
    collective_hover_thrust: float,
    possible_agent_ids: list[str] | None = None,
):
    """Set hover thrust on managed drones, zero thrust on the rest.

    Only drones listed in ``possible_agent_ids`` receive hover thrust.
    All others get zero thrust — they sit inert on the ground without
    unstable ground-contact dynamics.
    """
    managed = set(possible_agent_ids or agent_ids)
    for agent_id in agent_ids:
        asset = env.scene[agent_id]
        if agent_id in managed:
            hover_per_motor = float(collective_hover_thrust) / asset.num_thrusters
            thrust = torch.full(
                (len(env_ids), asset.num_thrusters),
                hover_per_motor,
                device=env.device,
                dtype=torch.float32,
            )
        else:
            thrust = torch.zeros(len(env_ids), asset.num_thrusters, device=env.device, dtype=torch.float32)
        asset.set_thrust_target(thrust, env_ids=env_ids)
