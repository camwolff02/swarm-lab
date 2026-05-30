# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the paper_swarm waypoint-navigation task."""

from __future__ import annotations

import torch


def _root_env(env):
    return env.root if hasattr(env, "root") else env


def passive_drone_count_curriculum(
    env,
    env_ids: torch.Tensor,
    min_passive: int,
    max_passive: int,
    ramp_steps: int,
) -> None:
    """Ramp the number of passive hovering drones from min_passive to max_passive.

    Sets ``_paper_swarm_num_passive_active`` on the root environment.  The
    ``_passive_drone_ids`` attribute determines which drones are eligible
    to hover; this curriculum controls how many of them are actually active
    (hovering) at a given training step.
    """
    root = _root_env(env)
    step = root.common_step_counter
    progress = min(float(step) / max(1, ramp_steps), 1.0)
    target_count = round(min_passive + progress * (max_passive - min_passive))
    target_count = max(min_passive, min(max_passive, target_count))
    root._paper_swarm_num_passive_active = target_count
    _set_passive_drone_mask(root, env_ids, target_count)
    root.extras["passive_drone_count"] = target_count


def _set_passive_drone_mask(root, env_ids: torch.Tensor, target_count: int) -> None:
    """Set the per-environment passive-drone visibility/control mask."""
    passive_drone_ids = list(getattr(root, "_passive_drone_ids", []))
    if not passive_drone_ids:
        return
    if env_ids is None:
        env_ids = torch.arange(root.num_envs, device=root.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=root.device, dtype=torch.long)

    all_drone_ids = [f"drone_{index}" for index in range(8)]
    mask_key = getattr(root, "_passive_mask_key", "passive_drones")
    old_mask = getattr(root, mask_key, None)
    if old_mask is None or old_mask.shape != (root.num_envs, len(all_drone_ids)):
        passive_mask = torch.zeros(root.num_envs, len(all_drone_ids), device=root.device, dtype=torch.bool)
    else:
        passive_mask = old_mask.clone()

    passive_indices = torch.tensor([all_drone_ids.index(agent_id) for agent_id in passive_drone_ids], device=root.device)
    passive_mask[env_ids.unsqueeze(-1), passive_indices.unsqueeze(0)] = False

    active_count = max(0, min(len(passive_drone_ids), int(target_count)))
    if active_count > 0:
        active_indices = torch.tensor(
            [all_drone_ids.index(agent_id) for agent_id in passive_drone_ids[:active_count]],
            device=root.device,
        )
        passive_mask[env_ids.unsqueeze(-1), active_indices.unsqueeze(0)] = True

    setattr(root, mask_key, passive_mask)


def active_agent_count_curriculum(
    env,
    env_ids: torch.Tensor,
    agent_ids: list[str],
    min_agents: int,
    max_agents: int,
    ramp_steps: int,
    mask_key: str,
    selection: str = "prefix",
) -> None:
    """Ramp the number of active agents from min_agents to max_agents.

    Args:
        min_agents: Starting number of active agents.
        max_agents: Final number of active agents.
        ramp_steps: Steps to complete the ramp.
        selection: "prefix" or "random" for choosing which agents are active.
    """
    root = _root_env(env)
    if env_ids is None:
        env_ids = torch.arange(root.num_envs, device=root.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=root.device, dtype=torch.long)
    num_envs = len(env_ids)
    num_agents = len(agent_ids)

    step = root.common_step_counter
    progress = min(step / max(1, ramp_steps), 1.0)
    target_count = round(min_agents + progress * (max_agents - min_agents))
    target_count = max(min_agents, min(max_agents, target_count))

    old_mask = getattr(root, mask_key, None)
    if old_mask is None or old_mask.shape != (root.num_envs, num_agents):
        mask = torch.zeros(root.num_envs, num_agents, device=root.device, dtype=torch.bool)
    else:
        mask = old_mask.clone()
    mask[env_ids] = False
    if selection == "prefix":
        mask[env_ids, :target_count] = True
    else:
        for e in range(num_envs):
            indices = torch.randperm(num_agents, device=root.device)[:target_count]
            mask[env_ids[e], indices] = True

    setattr(root, mask_key, mask)
    root.extras["active_agent_count"] = target_count


def paper_swarm_task_curriculum(
    env,
    env_ids: torch.Tensor,
    workspace_xy: tuple[float, float],
    workspace_z: tuple[float, float],
    max_static_columns: int,
    obstacle_start_step: int,
    obstacle_ramp_steps: int,
    randomization_start_step: int,
    randomization_ramp_steps: int,
    start_safe_sampling_prob: float,
    end_safe_sampling_prob: float,
    start_min_separation: float,
    end_min_separation: float,
    column_radius: float,
    column_safe_distance: float,
) -> None:
    """Stage the task from easy safe flight to cluttered random flight.

    The schedule follows the structure used by the formation paper: first learn
    obstacle-free flight, then add static obstacles, then train on a harder
    randomized setting. The collision paper's obstacle curriculum is reflected
    by gradually increasing obstacle density while keeping SDF observations
    fixed-size.

    Early stages force spawn and target positions to be separated from each
    other and from obstacles. The final stage anneals toward raw random samples,
    allowing intersecting starts and targets to appear during training.
    """
    root = _root_env(env)
    step = root.common_step_counter

    obstacle_progress = 0.0
    if step >= obstacle_start_step:
        obstacle_progress = min((step - obstacle_start_step) / max(1, obstacle_ramp_steps), 1.0)
    num_static_columns = round(obstacle_progress * max_static_columns)

    randomization_progress = 0.0
    if step >= randomization_start_step:
        randomization_progress = min((step - randomization_start_step) / max(1, randomization_ramp_steps), 1.0)

    safe_prob = start_safe_sampling_prob + randomization_progress * (
        end_safe_sampling_prob - start_safe_sampling_prob
    )
    min_sep = start_min_separation + randomization_progress * (end_min_separation - start_min_separation)

    root._paper_swarm_workspace_xy = workspace_xy
    root._paper_swarm_workspace_z = workspace_z
    root._paper_swarm_num_static_columns = num_static_columns
    root._paper_swarm_spawn_safe_sampling_prob = safe_prob
    root._paper_swarm_target_safe_sampling_prob = safe_prob
    root._paper_swarm_spawn_min_separation = min_sep
    root._paper_swarm_target_min_separation = min_sep
    root._paper_swarm_column_radius = column_radius
    root._paper_swarm_column_safe_distance = column_safe_distance

    root.extras["static_column_count"] = num_static_columns
    root.extras["safe_sampling_prob"] = safe_prob
    root.extras["spawn_target_min_separation"] = min_sep


def curriculum_fraction(env, start_step: int, end_step: int) -> float:
    """Clipped linear fraction between start_step and end_step."""
    if end_step <= start_step:
        return 1.0
    progress = (_root_env(env).common_step_counter - start_step) / (end_step - start_step)
    return float(max(0.0, min(1.0, progress)))


def expand_target_range_curriculum(
    env,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    start_xy: float,
    end_xy: float,
    start_z_delta: float,
    end_z_delta: float,
) -> dict[str, float]:
    """Linearly expand target waypoint range from drone position outward.

    Starts with targets at the drone's hover location (no movement needed)
    and progressively increases the XY and Z range so the policy must fly
    further to reach waypoints.  Pattern matches the lab_5 hover curriculum.
    """
    del env_ids
    frac = curriculum_fraction(env, start_step, end_step)
    xy = start_xy + frac * (end_xy - start_xy)
    z_delta = start_z_delta + frac * (end_z_delta - start_z_delta)
    ranges = env.command_manager.cfg.target_pose.ranges
    ranges.pos_x = (-xy, xy)
    ranges.pos_y = (-xy, xy)
    ranges.pos_z = (1.0 - z_delta, 1.0 + z_delta)
    return {"frac": frac, "xy": xy, "z_delta": z_delta}


_OBS_NOISE_TERMS = (
    "root_lin_vel_b",
    "root_ang_vel_b",
    "projected_gravity_b",
    "root_pos",
    "root_rotation_matrix",
    "neighbor_state",
    "static_sdf",
    "target_pos_b",
    "target_yaw_error",
    "distance_to_goal",
)


def update_observation_noise_curriculum(
    env,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    final_noise: float,
) -> dict[str, float]:
    """Linearly ramp observation noise from zero to *final_noise*.

    Modifies the ``noise`` attribute of each observation term in the
    ``policy`` observation group at runtime.  This follows the lab_5 / lab_6
    DR pattern.
    """
    del env_ids
    frac = curriculum_fraction(env, start_step, end_step)
    noise = frac * final_noise
    for term_name in _OBS_NOISE_TERMS:
        term = getattr(env.observation_manager.cfg.policy, term_name, None)
        if term is None or not hasattr(term, "noise"):
            continue
        term.noise.n_min = -noise
        term.noise.n_max = noise
    return {"frac": frac, "obs_noise": noise}
