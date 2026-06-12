# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and event functions for the paper_swarm waypoint-navigation task.

Spawns drones at random separated positions, places static column obstacles,
manages passive hovering drones, and hides inactive agents.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform


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

    Inactive agents that are passive hovering drones are placed at sampled
    hover altitudes. Remaining inactive agents are placed at the environment
    origin, resting on the ground plane.

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
    root = env.root if hasattr(env, "root") else env
    mask = _active_mask(env, agent_ids, mask_key)
    resets = {}
    spawn_safe_prob = getattr(root, "_paper_swarm_spawn_safe_sampling_prob", 1.0)
    spawn_min_separation = getattr(root, "_paper_swarm_spawn_min_separation", min_separation)
    column_radius = getattr(root, "_paper_swarm_column_radius", 0.15)
    column_safe_distance = getattr(root, "_paper_swarm_column_safe_distance", 0.6)
    columns = getattr(root, "column_positions", None)

    passive_drone_ids = getattr(root, "_passive_drone_ids", [])

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

    _sample_passive_hover_positions(
        root=root,
        env_ids=env_ids,
        passive_drone_ids=passive_drone_ids,
        agent_ids=agent_ids,
        active_positions=sampled_positions,
        active_mask=mask[env_ids],
        columns=columns,
        column_radius=column_radius,
        column_safe_distance=column_safe_distance,
    )

    hover_positions = getattr(root, "_passive_drone_hover_positions", None)
    if hover_positions is not None:
        passive_to_idx = {aid: i for i, aid in enumerate(passive_drone_ids)}
    passive_mask = _active_mask(env, agent_ids, getattr(root, "_passive_mask_key", "passive_drones"))

    for i, agent_id in enumerate(agent_ids):
        is_active = mask[env_ids, i]
        is_passive_eligible = agent_id in (passive_to_idx if hover_positions is not None else {})
        is_passive_active = passive_mask[env_ids, i] if is_passive_eligible else torch.zeros_like(is_active)

        asset = env.scene[agent_id]
        root_pose = asset.data.default_root_pose.torch[env_ids].clone()
        root_velocity = asset.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] = env.scene.env_origins[env_ids] + torch.tensor((0.0, 0.0, 0.05), device=env.device)
        root_pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        root_velocity[:, :] = 0.0

        if is_active.any():
            active_count = int(is_active.sum().item())
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(active_count, 1)
            lin_vel = sample_uniform(lin_vel_range[0], lin_vel_range[1], (active_count, 3), device=env.device)
            ang_vel = sample_uniform(ang_vel_range[0], ang_vel_range[1], (active_count, 3), device=env.device)
            root_pose[is_active, :3] = sampled_positions[is_active, i] + env.scene.env_origins[env_ids[is_active]]
            root_pose[is_active, 3:7] = quat
            root_velocity[is_active, :3] = lin_vel
            root_velocity[is_active, 3:6] = ang_vel

        if hover_positions is not None and is_passive_active.any():
            passive_idx = passive_to_idx[agent_id]
            pos_yaw = hover_positions[env_ids, passive_idx]
            root_pose[is_passive_active, :3] = pos_yaw[is_passive_active, :3] + env.scene.env_origins[
                env_ids[is_passive_active]
            ]
            root_pose[is_passive_active, 3:7] = quat_from_euler_xyz(
                torch.zeros(int(is_passive_active.sum().item()), device=env.device),
                torch.zeros(int(is_passive_active.sum().item()), device=env.device),
                pos_yaw[is_passive_active, 3],
            )

        # Spread genuinely inactive drones (not active, not passive) in a grid
        GRID_SPACING = 1.0
        GRID_COLS = 3
        is_inactive = ~is_active & ~is_passive_active
        if i > 0 and is_inactive.any():
            gx = ((i - 1) % GRID_COLS) * GRID_SPACING
            gy = ((i - 1) // GRID_COLS) * GRID_SPACING
            root_pose[is_inactive, :3] = (
                env.scene.env_origins[env_ids[is_inactive]] + torch.tensor((gx, gy, 0.05), device=env.device)
            )

        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        resets[agent_id] = torch.cat([root_pose, root_velocity], dim=-1)

    return resets


def _sample_passive_hover_positions(
    root,
    env_ids: torch.Tensor,
    passive_drone_ids: list[str],
    agent_ids: list[str],
    active_positions: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    columns: torch.Tensor | None = None,
    column_radius: float = 0.15,
    column_safe_distance: float = 0.6,
) -> None:
    """Sample hover positions for passive drones and store on the root env.

    Positions are sampled with separation between passive drones and obstacle
    avoidance so they never spawn inside each other or inside columns.

    Args:
        root: The root environment instance.
        env_ids: Environment indices to sample for.
        passive_drone_ids: List of passive drone agent IDs.
        agent_ids: All drone agent IDs (needed for position indexing).
        active_positions: Active-agent sampled positions [m], shape (num_envs, num_agents, 3).
        active_mask: Active-agent mask, shape (num_envs, num_agents).
        columns: Optional column positions tensor ``(num_envs, num_cols, 3)``.
        column_radius: Column radius for safe sampling [m].
        column_safe_distance: Safe distance from columns [m].
    """
    if not passive_drone_ids:
        return

    num_envs = len(env_ids)
    num_passive = len(passive_drone_ids)
    device = root.device

    hover_positions = getattr(root, "_passive_drone_hover_positions", None)
    if hover_positions is None or hover_positions.shape != (root.num_envs, num_passive, 4):
        hover_positions = torch.zeros(root.num_envs, num_passive, 4, device=device)

    x_min, x_max = (-3.5, 3.5)
    y_min, y_max = (-3.5, 3.5)
    hover_z = 2.0
    min_separation = 1.0

    positions = torch.zeros(num_envs, num_passive, 3, device=device)
    yaws = torch.zeros(num_envs, num_passive, device=device)
    max_attempts = 64

    for p in range(num_passive):
        candidates = torch.empty(num_envs, max_attempts, 3, device=device)
        candidates[:, :, 0].uniform_(x_min, x_max)
        candidates[:, :, 1].uniform_(y_min, y_max)
        candidates[:, :, 2] = hover_z
        valid = torch.ones(num_envs, max_attempts, dtype=torch.bool, device=device)

        if p > 0:
            prev_xy = positions[:, :p, :2]
            cand_xy = candidates[:, :, :2]
            dists = torch.cdist(cand_xy, prev_xy)
            valid = valid & (dists.min(dim=-1).values >= min_separation)

        if active_positions is not None and active_mask is not None and active_mask.any():
            cand_xy = candidates[:, :, :2]
            active_xy = active_positions[:, :, :2]
            dists = torch.cdist(cand_xy, active_xy)
            clear = (dists >= min_separation) | ~active_mask[:, None, :]
            valid = valid & clear.all(dim=-1)

        env_columns = columns[env_ids] if columns is not None else None
        if env_columns is not None and env_columns.shape[1] > 0:
            cand_xy = candidates[:, :, None, :2]
            col_xy = env_columns[:, None, :, :2]
            col_dists = torch.linalg.norm(cand_xy - col_xy, dim=-1)
            col_active = torch.linalg.norm(env_columns[:, :, :2], dim=-1) < 100.0
            col_clear = (col_dists >= (column_radius + column_safe_distance)) | ~col_active[:, None, :]
            valid = valid & col_clear.all(dim=-1)

        selected_idx = valid.float().argmax(dim=-1)
        positions[:, p] = candidates[range(num_envs), selected_idx]

    yaws.uniform_(-torch.pi, torch.pi)
    hover_positions[env_ids, :, 0] = positions[:, :, 0]
    hover_positions[env_ids, :, 1] = positions[:, :, 1]
    hover_positions[env_ids, :, 2] = positions[:, :, 2]
    hover_positions[env_ids, :, 3] = yaws

    setattr(root, "_passive_drone_hover_positions", hover_positions)


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
    """Set hover thrust on managed and passive drones, zero thrust on the rest.

    Only drones listed in ``possible_agent_ids`` or identified as passive
    hovering drones receive hover thrust. All others get zero thrust.

    Also resamples per-motor thruster parameters (``thrust_const``, time
    constants) via ``Thruster.reset_idx()`` for every drone, so the thrust-
    constant domain randomisation configured in ``CRAZYFLIE_CFG`` takes
    effect on each episode reset.
    """
    from isaaclab_contrib.actuators import Thruster

    managed = set(possible_agent_ids or agent_ids)
    root = env.root if hasattr(env, "root") else env
    passive_drone_ids = getattr(root, "_passive_drone_ids", [])
    managed |= set(passive_drone_ids)
    for agent_id in agent_ids:
        asset = env.scene[agent_id]

        for actuator in asset.actuators.values():
            if isinstance(actuator, Thruster):
                actuator.reset_idx(env_ids)

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
