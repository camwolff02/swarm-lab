# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the drone waypoint MARL task."""

from __future__ import annotations

import torch


def _root_env(env):
    return getattr(env, "root", env)


def _global_step(env) -> int:
    root = _root_env(env)
    return int(getattr(root, "common_step_counter", getattr(root, "_sim_step_counter", 0)))


def active_agent_count_curriculum(
    env,
    env_ids,
    agent_ids,
    min_agents: int,
    max_agents: int,
    ramp_steps: int,
    mask_key: str,
    selection: str = "prefix",
):
    """Linearly ramp active agent count from min_agents to max_agents over ramp_steps [steps]."""
    del env_ids, selection
    root = _root_env(env)
    if not hasattr(root, "extras") or root.extras is None:
        root.extras = {}
    step = _global_step(root)
    progress = 1.0 if ramp_steps <= 0 else min(max(step / float(ramp_steps), 0.0), 1.0)
    active_count = int(round(min_agents + progress * (max_agents - min_agents)))
    active_count = max(min_agents, min(max_agents, active_count))
    mask = torch.zeros((root.num_envs, len(agent_ids)), dtype=torch.bool, device=root.device)
    mask[:, :active_count] = True
    setattr(root, mask_key, mask)
    setattr(env, mask_key, mask)
    root.extras[mask_key] = mask
    root.extras["active_agent_count"] = active_count
    return {"active_count": active_count}


def waypoint_randomization_curriculum(
    env,
    env_ids,
    command_name: str,
    agent_ids,
    workspace_xy,
    workspace_z,
    start_step: int,
    ramp_steps: int,
    start_safe_sampling_prob: float,
    end_safe_sampling_prob: float,
    start_min_separation: float,
    end_min_separation: float,
    mask_key: str,
):
    """Linearly anneal safe-waypoint sampling probability and min separation [m] over ramp_steps [steps]."""
    del command_name, agent_ids, workspace_xy, workspace_z, env_ids, mask_key
    root = _root_env(env)
    if not hasattr(root, "extras") or root.extras is None:
        root.extras = {}
    step = _global_step(root)
    progress = 0.0 if step < start_step else (1.0 if ramp_steps <= 0 else min((step - start_step) / float(ramp_steps), 1.0))
    safe_prob = start_safe_sampling_prob + progress * (end_safe_sampling_prob - start_safe_sampling_prob)
    min_sep = start_min_separation + progress * (end_min_separation - start_min_separation)
    root._waypoint_safe_sampling_prob = float(safe_prob)
    root._waypoint_min_separation = float(min_sep)
    setattr(env, "_waypoint_safe_sampling_prob", float(safe_prob))
    setattr(env, "_waypoint_min_separation", float(min_sep))
    root.extras["waypoint_safe_sampling_prob"] = float(safe_prob)
    root.extras["waypoint_min_separation"] = float(min_sep)
    return {"safe_sampling_prob": float(safe_prob), "min_separation": float(min_sep)}
