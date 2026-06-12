# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for manager-based multi-agent RL environments.

This module adds the reinforcement-learning layer on top of
:mod:`manager_based_ma_env_cfg`. The split is intentional: the classical
multi-agent config owns observations, actions, and commands, while this module
owns rewards, terminations, curriculum, and reset aggregation. Critic inputs are
ordinary observation manager groups named ``"critic"``.
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any, Literal

from isaaclab.utils.configclass import configclass

from .manager_based_ma_env_cfg import AgentCfg, AgentGroupCfg, AgentID, ManagerBasedMaEnvCfg

ResetMode = Literal["any", "all"]


@configclass
class AgentRlCfg(AgentCfg):
    """Manager configuration for one concrete RL agent.

    This extends :class:`AgentCfg` with the manager terms required by RL
    algorithms.

    Attributes:
        rewards: Reward manager configuration.
        terminations: Termination manager configuration.
        curriculum: Optional curriculum manager configuration.
    """

    rewards: Any = MISSING
    terminations: Any = MISSING
    curriculum: Any | None = None


@configclass
class ManagerBasedMarlEnvCfg(ManagerBasedMaEnvCfg):
    """Configuration for a manager-based multi-agent RL environment.

    Attributes:
        agents: Explicit per-agent RL configs.
        agent_groups: Group declarations that expand one :class:`AgentRlCfg` to
            multiple concrete agents.
        reset_on: Environment-level reset aggregation. ``"any"`` resets all
            agents when any agent terminates/truncates; ``"all"`` waits for all
            agents in the aggregation.
        episode_length_s: Episode duration [s].
        is_finite_horizon: Whether time limits are part of the MDP.
    """

    agents: dict[AgentID, AgentRlCfg] = field(default_factory=dict)
    agent_groups: list[AgentGroupCfg] = field(default_factory=list)

    reset_on: ResetMode = "any"

    episode_length_s: float = MISSING
    is_finite_horizon: bool = False
