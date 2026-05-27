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
    from .observations import _active_mask
    from .commands import _sample_positions

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    mask = _active_mask(env, agent_ids, mask_key)
    resets = {}
    root = env.root if hasattr(env, "root") else env
    spawn_safe_prob = getattr(root, "_paper_swarm_spawn_safe_sampling_prob", 1.0)
    spawn_min_separation = getattr(root, "_paper_swarm_spawn_min_separation", min_separation)
    column_radius = getattr(root, "_paper_swarm_column_radius", 0.15)
    column_safe_distance = getattr(root, "_paper_swarm_column_safe_distance", 0.6)
    columns = getattr(root, "column_positions", None)

    sampled_positions = torch.zeros(len(env_ids), len(agent_ids), 3, device=env.device)
    for local_env_index, env_id in enumerate(env_ids):
        active_indices = torch.nonzero(mask[env_id], as_tuple=False).flatten()
        safe = bool(torch.rand((), device=env.device) < spawn_safe_prob)
        env_columns = None if columns is None else columns[env_id]
        sampled_positions[local_env_index, active_indices] = _sample_positions(
            count=len(active_indices),
            xy_bounds=xy_bounds,
            z_bounds=z_bounds,
            min_separation=spawn_min_separation if safe else 0.0,
            columns=env_columns if safe else None,
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

    cell_centers = torch.zeros(total_cells, 2, device=device)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            x = -grid_border + col * grid_size
            y = -grid_border + row * grid_size
            cell_centers[idx, 0] = x
            cell_centers[idx, 1] = y
            idx += 1

    all_positions = getattr(env, column_positions_key, None)
    if all_positions is None or all_positions.shape != (env.num_envs, num_columns, 3):
        all_positions = torch.zeros(env.num_envs, num_columns, 3, device=device)
    positions = torch.zeros(num_envs, num_columns, 3, device=device)
    positions[:, :, 0] = 1000.0
    positions[:, :, 1] = 1000.0
    for e in range(num_envs):
        if current_columns == 0:
            continue
        perm = torch.randperm(total_cells, device=device)[:current_columns]
        positions[e, :current_columns, :2] = cell_centers[perm]
        positions[e, :current_columns, 2] = 0.0

    all_positions[env_ids] = positions
    setattr(env, column_positions_key, all_positions)
    root.extras["static_column_count"] = current_columns


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _sample_separated_positions(
    n: int,
    xy_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    min_separation: float,
    device: torch.device,
    max_attempts: int = 128,
) -> torch.Tensor:
    """Rejection-sample N positions with minimum XY separation.

    Args:
        n: Number of positions to sample.
        xy_bounds: (min, max) per axis [m].
        z_bounds: (min, max) [m].
        min_separation: Minimum XY distance between any pair [m].
        max_attempts: Maximum attempts per agent before giving up.

    Returns:
        Positions tensor (n, 3).
    """
    low = xy_bounds[0]
    high = xy_bounds[1]
    z_low, z_high = z_bounds

    positions = torch.zeros(n, 3, device=device)
    for i in range(n):
        placed = False
        for _ in range(max_attempts):
            candidate = torch.zeros(3, device=device)
            candidate[0] = low + (high - low) * torch.rand(1, device=device)
            candidate[1] = low + (high - low) * torch.rand(1, device=device)
            candidate[2] = z_low + (z_high - z_low) * torch.rand(1, device=device)

            if i == 0:
                placed = True
                break

            dists = torch.norm(positions[:i, :2] - candidate[:2].unsqueeze(0), dim=-1)
            if (dists >= min_separation).all():
                placed = True
                break

        if not placed:
            candidate[:2] = low + (high - low) * torch.rand(2, device=device)
        positions[i] = candidate


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
