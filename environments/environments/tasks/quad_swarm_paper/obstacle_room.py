"""Paper-specific room, obstacle, start, and goal sampling."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from cpsquare_lab.tasks.swarm.grid_sdf import (
    get_cell_centers,
)

from . import paper_spec as spec


def sample_obstacle_occupancy(
    num_envs: int,
    *,
    density: float = spec.OBSTACLE_DENSITY,
    grid_shape: tuple[int, int] = spec.OBSTACLE_GRID_SHAPE,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample paper-style obstacle occupancy over the centered 8x8 grid."""

    num_cells = grid_shape[0] * grid_shape[1]
    obstacle_count = int(float(density) * num_cells)
    occupancy = torch.zeros((num_envs, num_cells), device=device, dtype=torch.bool)
    for env_id in range(num_envs):
        if obstacle_count > 0:
            indices = torch.randperm(num_cells, device=device)[:obstacle_count]
            occupancy[env_id, indices] = True
    return occupancy.reshape(num_envs, *grid_shape)


def sample_obstacle_field(
    num_envs: int,
    *,
    density: float = spec.OBSTACLE_DENSITY,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed obstacle slots and masks for a sampled paper obstacle map."""

    occupancy = sample_obstacle_occupancy(num_envs, density=density, device=device)
    cell_centers = get_cell_centers(spec.OBSTACLE_GRID_SHAPE, spec.OBSTACLE_CELL_SIZE, device=device)
    obstacle_positions = torch.zeros((num_envs, cell_centers.shape[0], 3), device=device, dtype=torch.float32)
    obstacle_positions[..., :2] = cell_centers.view(1, -1, 2)
    obstacle_positions[..., 2] = spec.ROOM_HEIGHT * 0.5
    return obstacle_positions, occupancy.reshape(num_envs, -1)


def sample_start_goal_pairs(
    num_envs: int,
    num_drones: int,
    *,
    room_size: float = spec.ROOM_SIZE,
    altitude: float = 1.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample start/goal pairs on opposite sides of the room."""

    base_angles = torch.linspace(0.0, 2.0 * math.pi, num_drones + 1, device=device)[:-1]
    phase = torch.rand((num_envs, 1), device=device) * 2.0 * math.pi
    angles = base_angles.unsqueeze(0) + phase
    radius = 0.35 * float(room_size)

    starts = torch.zeros((num_envs, num_drones, 3), device=device, dtype=torch.float32)
    starts[..., 0] = radius * torch.cos(angles)
    starts[..., 1] = radius * torch.sin(angles)
    starts[..., 2] = altitude
    goals = starts.clone()
    goals[..., :2] *= -1.0

    yaw = angles + math.pi
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(roll.reshape(-1), pitch.reshape(-1), yaw.reshape(-1))
    orientations = orientations.reshape(num_envs, num_drones, 4)
    return starts, goals, orientations


def sample_obstacle_aware_start_goal_pairs(
    obstacle_mask: torch.Tensor,
    obstacle_positions: torch.Tensor,
    num_drones: int,
    *,
    altitude: float = 1.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample start/goal cells from the free obstacle grid.

    The reference implementation samples the obstacle layout before scenario
    positions, then places agents and goals in free cells.  This keeps fresh
    resets from starting a quadrotor inside an occupied obstacle column.
    """

    num_envs, num_slots = obstacle_mask.shape
    starts = torch.zeros((num_envs, num_drones, 3), device=device, dtype=torch.float32)
    goals = torch.zeros_like(starts)

    fallback_positions = obstacle_positions.to(device=device, dtype=torch.float32)
    for env_id in range(num_envs):
        free_indices = torch.nonzero(~obstacle_mask[env_id], as_tuple=False).squeeze(-1)
        if free_indices.numel() < 2 * num_drones:
            free_indices = torch.arange(num_slots, device=device)

        shuffled = free_indices[torch.randperm(free_indices.numel(), device=device)]
        start_indices = shuffled[:num_drones]
        goal_indices = shuffled[num_drones : 2 * num_drones]
        if goal_indices.numel() < num_drones:
            goal_indices = shuffled[:num_drones].flip(0)

        starts[env_id, :, :2] = fallback_positions[env_id, start_indices, :2]
        goals[env_id, :, :2] = fallback_positions[env_id, goal_indices, :2]

    starts[..., 2] = altitude
    goals[..., 2] = altitude

    yaw = torch.atan2(goals[..., 1] - starts[..., 1], goals[..., 0] - starts[..., 0])
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(roll.reshape(-1), pitch.reshape(-1), yaw.reshape(-1))
    orientations = orientations.reshape(num_envs, num_drones, 4)
    return starts, goals, orientations
