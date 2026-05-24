# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the paper_swarm waypoint-navigation task.

1. Active agent count ramp: 1 -> N over configurable steps.
2. Waypoint randomization: anneals collision-safe sampling.
"""

from __future__ import annotations

import torch


def _root_env(env):
    return env.root if hasattr(env, "root") else env


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


def waypoint_randomization_curriculum(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    agent_ids: list[str],
    workspace_xy: tuple[float, float],
    workspace_z: tuple[float, float],
    start_step: int,
    ramp_steps: int,
    start_safe_sampling_prob: float,
    end_safe_sampling_prob: float,
    start_min_separation: float,
    end_min_separation: float,
    mask_key: str,
) -> None:
    """Anneals waypoint sampling from safe (separated) to stochastic (random).

    Before start_step: full safe sampling (start_safe_sampling_prob, start_min_separation).
    During ramp: linear interpolation between start and end values.
    After ramp: fully stochastic (end_safe_sampling_prob, end_min_separation).

    Stores current values on env for use by the command sampler.
    """
    root = _root_env(env)
    step = root.common_step_counter
    if step < start_step:
        progress = 0.0
    else:
        progress = min((step - start_step) / max(1, ramp_steps), 1.0)

    safe_prob = start_safe_sampling_prob + progress * (end_safe_sampling_prob - start_safe_sampling_prob)
    min_sep = start_min_separation + progress * (end_min_separation - start_min_separation)

    setattr(root, "_waypoint_safe_sampling_prob", safe_prob)
    setattr(root, "_waypoint_min_separation", min_sep)
    root.extras["waypoint_safe_sampling_prob"] = safe_prob
    root.extras["waypoint_min_separation"] = min_sep
