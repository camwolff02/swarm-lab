# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic manager-based multi-agent swarm runtime."""

from typing import TYPE_CHECKING

from .manager_based_ma_env_cfg import (
    AgentProfileCfg,
    AgentSetCfg,
    ManagerBasedMaEnvCfg,
    MultiAgentOptionsCfg,
    compile_multi_agent_spec,
)
from .manager_based_marl_env_cfg import (
    ManagerBasedMarlEnvCfg,
    MultiAgentStateCfg,
)

if TYPE_CHECKING:
    from .manager_based_ma_env import ManagerBasedMaEnv
    from .manager_based_marl_env import ManagerBasedMarlEnv

__all__ = [
    "AgentProfileCfg",
    "AgentSetCfg",
    "ManagerBasedMaEnv",
    "ManagerBasedMaEnvCfg",
    "ManagerBasedMarlEnv",
    "ManagerBasedMarlEnvCfg",
    "MultiAgentOptionsCfg",
    "MultiAgentStateCfg",
    "compile_multi_agent_spec",
]


def __getattr__(name: str):
    if name == "ManagerBasedMaEnv":
        from .manager_based_ma_env import ManagerBasedMaEnv

        return ManagerBasedMaEnv
    if name == "ManagerBasedMarlEnv":
        from .manager_based_marl_env import ManagerBasedMarlEnv

        return ManagerBasedMarlEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
