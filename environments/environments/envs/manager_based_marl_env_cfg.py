# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for manager-based multi-agent RL environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils.configclass import configclass

from .manager_based_ma_env_cfg import AgentProfileCfg, ManagerBasedMaEnvCfg

StateMode = Literal["none", "concat_policy", "observation_group", "custom"]


@configclass
class AgentRlProfileCfg(AgentProfileCfg):
    """Reusable MARL manager bundle for one embodiment or role."""

    rewards: Any = MISSING
    terminations: Any = MISSING
    curriculum: Any | None = None


@configclass
class MultiAgentStateCfg:
    """Optional centralized state / critic observation configuration."""

    mode: StateMode = "none"

    # DirectMARLEnvCfg-style semantics:
    #   0  -> no centralized state
    #  -1  -> infer by concatenation / manager output
    #  >0  -> explicit flat state dimension
    # Gym spaces or nested spaces may also be supplied by advanced users.
    state_space: Any = 0

    # Used when ``mode == "observation_group"``.
    group_name: str = "critic"

    # Used when ``mode == "custom"``. The function should accept the env and
    # return either a tensor-like global state or dict[agent_id, state].
    function: Callable[[Any], Any] | None = None

    # SKRL's multi-agent abstractions commonly expose state spaces per agent.
    # If True, runtime adapters may duplicate a global state into a dict keyed
    # by agent. If False, runtime adapters may expose a single global state.
    expose_as_dict: bool = False


@configclass
class ManagerBasedMarlEnvCfg(ManagerBasedMaEnvCfg):
    """Configuration for a manager-based multi-agent RL environment."""

    profiles: dict[str, AgentRlProfileCfg] = MISSING

    # Optional centralized/global state config. It is harmless for IPPO and
    # becomes important for CTDE algorithms such as MAPPO.
    state: MultiAgentStateCfg = MultiAgentStateCfg()

    # Same semantics as ManagerBasedRLEnvCfg / DirectMARLEnvCfg.
    episode_length_s: float = MISSING
    is_finite_horizon: bool = False
