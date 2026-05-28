# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the Xie et al. formation swarm task."""

from __future__ import annotations

import torch


def _root_env(env):
    return env.root if hasattr(env, "root") else env


def formation_curriculum_obstacles(env, env_ids: torch.Tensor) -> None:
    """No-op curriculum term.

    The formation_swarm task resolves active obstacles at env init time
    based on 'curriculum_stage'. This term ensures the curriculum manager
    has at least one term registered.
    """
    root = _root_env(env)
    env_ids = torch.as_tensor(env_ids, device=root.device, dtype=torch.long)

    if getattr(root, "_formation_curriculum_applied", True):
        return

    cfg = root.cfg
    curriculum_stage = getattr(cfg, "curriculum_stage", 0)
    num_balls = getattr(cfg, "num_balls", 2)
    static_obstacles = getattr(cfg, "static_obstacles", 10)

    if curriculum_stage == 1:
        active_static = 0
        active_balls = 0
        throw_threshold = getattr(cfg, "curriculum_delayed_throw_threshold_steps", 1000)
        throw_range = getattr(cfg, "curriculum_delayed_throw_time_range_steps", 800)
    elif curriculum_stage == 2:
        active_static = static_obstacles
        active_balls = 0
        throw_threshold = getattr(cfg, "curriculum_delayed_throw_threshold_steps", 1000)
        throw_range = getattr(cfg, "curriculum_delayed_throw_time_range_steps", 800)
    elif curriculum_stage == 3:
        active_static = static_obstacles
        active_balls = num_balls
        throw_threshold = getattr(cfg, "throw_threshold_steps", 150)
        throw_range = getattr(cfg, "throw_time_range_steps", 450)
    else:
        active_static = getattr(cfg, "active_static_obstacles", None) or static_obstacles
        active_balls = getattr(cfg, "active_balls", None) or num_balls
        throw_threshold = getattr(cfg, "throw_threshold_steps", 150)
        throw_range = getattr(cfg, "throw_time_range_steps", 450)

    active_static = max(0, min(int(active_static), static_obstacles))
    active_balls = max(0, min(int(active_balls), num_balls))

    root._formation_active_static_obstacles = active_static
    root._formation_active_balls = active_balls
    root._formation_throw_threshold_steps = int(throw_threshold)
    root._formation_throw_time_range_steps = int(throw_range)
    root._formation_curriculum_applied = True
