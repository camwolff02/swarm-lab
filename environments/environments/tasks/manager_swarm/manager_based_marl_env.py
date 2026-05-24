# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based MARL environment runtime."""

from __future__ import annotations

from typing import Any

import torch

from .manager_based_ma_env import ManagerBasedMaEnv
from .manager_based_marl_env_cfg import ManagerBasedMarlEnvCfg


class ManagerBasedMarlEnv(ManagerBasedMaEnv):
    """Manager-based MARL environment with centralized state support.

    Extends :class:`ManagerBasedMaEnv` with a :meth:`state` method that
    computes a global state tensor for CTDE algorithms such as MAPPO. State
    mode is configured via :attr:`cfg.state`.
    """

    cfg: ManagerBasedMarlEnvCfg

    def state(self) -> Any:
        if self.cfg.state.mode == "none":
            return None
        if self.cfg.state.mode == "custom":
            if self.cfg.state.function is None:
                raise RuntimeError("cfg.state.mode='custom' requires cfg.state.function")
            return self.cfg.state.function(self)
        if self.cfg.state.mode == "concat_policy":
            self.state_buf = torch.cat(
                [self.obs_dict[agent].reshape(self.num_envs, -1) for agent in self.possible_agents], dim=-1
            )
            return self._format_state(self.state_buf)
        if self.cfg.state.mode == "observation_group":
            group_name = self.cfg.state.group_name
            if not self.cfg.state.expose_as_dict:
                with self._agent_context(self.possible_agents[0]):
                    self.state_buf = self.observation_manager.compute_group(group_name, update_history=False)
                return self.state_buf
            state_by_agent = {}
            for agent_id in self.possible_agents:
                with self._agent_context(agent_id):
                    state_by_agent[agent_id] = self.observation_manager.compute_group(group_name, update_history=False)
            self.state_buf = state_by_agent
            return self.state_buf
        raise ValueError(f"Unsupported multi-agent state mode: {self.cfg.state.mode!r}")

    def _format_state(self, state: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.cfg.state.expose_as_dict:
            return {agent_id: state for agent_id in self.possible_agents}
        return state
