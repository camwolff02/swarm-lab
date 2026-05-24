# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based multi-agent environment runtime package.

The runtime classes (ManagerBasedMaEnv, ManagerBasedMarlEnv) are lazy-loaded
via __getattr__ so that importing this package for its config types does not
trigger early PXR/USD imports before SimulationApp is running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .manager_based_ma_env_cfg import (
    AgentCfg,
    AgentGroupCfg,
    AgentGroupRuntimeSpec,
    AgentRuntimeSpec,
    ManagerBasedMaEnvCfg,
    MultiAgentRuntimeSpec,
    compile_multi_agent_spec,
)
from .manager_based_marl_env_cfg import AgentRlCfg, ManagerBasedMarlEnvCfg

if TYPE_CHECKING:
    from .manager_based_ma_env import ManagerBasedMaEnv
    from .manager_based_marl_env import ManagerBasedMarlEnv

__all__ = [
    "AgentCfg",
    "AgentGroupCfg",
    "AgentGroupRuntimeSpec",
    "AgentRlCfg",
    "AgentRuntimeSpec",
    "ManagerBasedMaEnv",
    "ManagerBasedMaEnvCfg",
    "ManagerBasedMarlEnv",
    "ManagerBasedMarlEnvCfg",
    "MultiAgentRuntimeSpec",
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
