# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based MARL runtime built on the classical MA workflow."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from isaaclab.managers import CurriculumManager, RewardManager, TerminationManager

from .manager_based_ma_env import ManagerBasedMaEnv, _ManagerBundle
from .manager_based_marl_env_cfg import ManagerBasedMarlEnvCfg


class ManagerBasedMarlEnv(ManagerBasedMaEnv):
    """Manager-based multi-agent RL environment.

    This class adds reward, termination, curriculum, and centralized-state
    managers on top of the classical pooled manager workflow. It still relies on
    DirectMARLEnv's inherited step loop through ``ManagerBasedMaEnv``.
    """

    cfg: ManagerBasedMarlEnvCfg

    def __init__(self, cfg: ManagerBasedMarlEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.state_space = cfg.state.state_space
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, actions):
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_pre_step()
        observations, rewards, terminated, truncated, extras = super().step(actions)
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_post_step()
        return observations, rewards, terminated, truncated, extras

    def _create_classical_manager_bundle(self, runtime) -> _ManagerBundle:
        bundle = super()._create_classical_manager_bundle(runtime)
        profile = runtime.profile

        bundle.reward_manager = RewardManager(profile.rewards, bundle.env_view)
        bundle.termination_manager = TerminationManager(profile.terminations, bundle.env_view)
        bundle.curriculum_manager = (
            CurriculumManager(profile.curriculum, bundle.env_view) if profile.curriculum is not None else None
        )

        bundle.env_view.reward_manager = bundle.reward_manager
        bundle.env_view.termination_manager = bundle.termination_manager
        bundle.env_view.curriculum_manager = bundle.curriculum_manager
        return bundle

    def _make_state_space(self) -> gym.Space | None:
        state_cfg = self.cfg.state
        if state_cfg.mode == "none" or not state_cfg.state_space:
            return None
        if isinstance(state_cfg.state_space, gym.Space):
            return state_cfg.state_space
        if isinstance(state_cfg.state_space, int) and state_cfg.state_space > 0:
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_cfg.state_space,), dtype=np.float32)
        if state_cfg.mode == "concat_policy":
            dim = sum(int(np.prod(space.shape)) for space in self.observation_spaces.values())
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        if state_cfg.mode == "observation_group":
            dim = 0
            for bundle in self._manager_bundles.values():
                obs_dim = bundle.observation_manager.group_obs_dim[state_cfg.group_name]
                if not isinstance(obs_dim, tuple):
                    raise ValueError(f"State observation group {state_cfg.group_name!r} must be concatenated.")
                dim += int(np.prod(obs_dim))
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        if state_cfg.mode == "custom":
            if isinstance(state_cfg.state_space, gym.Space):
                return state_cfg.state_space
            if isinstance(state_cfg.state_space, int) and state_cfg.state_space > 0:
                return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_cfg.state_space,), dtype=np.float32)
            return None
        raise ValueError(f"Unsupported multi-agent state mode: {state_cfg.mode!r}")

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        self._compute_curriculum(env_ids)
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_pre_reset(env_ids)
        super()._reset_idx(env_ids)
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_post_reset(env_ids)

    def _compute_curriculum(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        for bundle in self._manager_bundles.values():
            if bundle.curriculum_manager is not None:
                bundle.curriculum_manager.compute(env_ids=env_ids)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards: dict[str, torch.Tensor] = {}
        for bundle in self._manager_bundles.values():
            reward = bundle.reward_manager.compute(dt=self.step_dt)
            rewards.update(self._fanout_reward_or_done(reward, bundle.runtime, dtype=None))
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        reset_mode = self.cfg.ma_options.reset_on
        if reset_mode not in ("global", "any_agent", "all_agents"):
            raise ValueError(
                f"Unsupported multi-agent reset mode: {reset_mode!r}. "
                "V2 only supports env-level reset aggregation, not per-agent reset."
            )

        raw_terminated_by_set: list[torch.Tensor] = []
        raw_truncated_by_set: list[torch.Tensor] = []
        per_agent_terminated: dict[str, torch.Tensor] = {}
        per_agent_truncated: dict[str, torch.Tensor] = {}

        for bundle in self._manager_bundles.values():
            bundle.termination_manager.compute()
            raw_terminated = bundle.termination_manager.terminated
            raw_truncated = bundle.termination_manager.time_outs
            if reset_mode in ("global", "any_agent"):
                raw_terminated_by_set.append(self._reduce_set_done_tensor(raw_terminated, bundle.runtime, op="any"))
                raw_truncated_by_set.append(self._reduce_set_done_tensor(raw_truncated, bundle.runtime, op="any"))
            elif reset_mode == "all_agents":
                raw_terminated_by_set.append(self._reduce_set_done_tensor(raw_terminated, bundle.runtime, op="all"))
                raw_truncated_by_set.append(self._reduce_set_done_tensor(raw_truncated, bundle.runtime, op="all"))
            per_agent_terminated.update(self._fanout_reward_or_done(raw_terminated, bundle.runtime, dtype=torch.bool))
            per_agent_truncated.update(self._fanout_reward_or_done(raw_truncated, bundle.runtime, dtype=torch.bool))

        if reset_mode in ("global", "any_agent"):
            global_terminated = self._reduce_tensor_list(raw_terminated_by_set, op="any")
            global_truncated = self._reduce_tensor_list(raw_truncated_by_set, op="any")
            terminated = {agent_id: global_terminated for agent_id in self.possible_agents}
            truncated = {agent_id: global_truncated for agent_id in self.possible_agents}
        elif reset_mode == "all_agents":
            global_terminated = self._reduce_tensor_list(raw_terminated_by_set, op="all")
            global_truncated = self._reduce_tensor_list(raw_truncated_by_set, op="all")
            terminated = {agent_id: global_terminated for agent_id in self.possible_agents}
            truncated = {agent_id: global_truncated for agent_id in self.possible_agents}
        else:
            terminated = per_agent_terminated
            truncated = per_agent_truncated

        self.reset_terminated = self._reduce_tensor_list(list(terminated.values()), op="any")
        self.reset_time_outs = self._reduce_tensor_list(list(truncated.values()), op="any")
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        return terminated, truncated

    def _get_states(self) -> Any:
        return self.state()

    def state(self) -> Any:
        """Compute the optional centralized training state.

        Returns:
            ``None`` for decentralized algorithms, a tensor for shared global
            state, or a dict keyed by agent id when ``expose_as_dict`` is true.
        """

        state_cfg = self.cfg.state
        if state_cfg.mode == "none":
            return None
        if state_cfg.mode == "custom":
            if state_cfg.function is None:
                raise RuntimeError("cfg.state.mode='custom' requires cfg.state.function")
            self.state_buf = state_cfg.function(self)
            return self.state_buf
        if state_cfg.mode == "concat_policy":
            if not self.obs_dict:
                self.obs_dict = self._get_observations()
            state = torch.cat(
                [self.obs_dict[agent_id].reshape(self.num_envs, -1) for agent_id in self.possible_agents], dim=-1
            )
            self.state_buf = state
            return self._format_state(state)
        if state_cfg.mode == "observation_group":
            state_parts = []
            for bundle in self._manager_bundles.values():
                if hasattr(bundle.observation_manager, "compute_group"):
                    obs = bundle.observation_manager.compute_group(state_cfg.group_name, update_history=False)
                else:
                    obs = bundle.observation_manager.compute(update_history=False)[state_cfg.group_name]
                state_parts.append(obs.reshape(self.num_envs, -1))
            state = torch.cat(state_parts, dim=-1) if len(state_parts) > 1 else state_parts[0]
            self.state_buf = state
            return self._format_state(state)
        raise ValueError(f"Unsupported multi-agent state mode: {state_cfg.mode!r}")

    def _format_state(self, state: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.cfg.state.expose_as_dict:
            return {agent_id: state for agent_id in self.possible_agents}
        return state

    def _fanout_reward_or_done(self, value: torch.Tensor, runtime, dtype) -> dict[str, torch.Tensor]:
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype=dtype)
        if value.ndim >= 2 and value.shape[1] == runtime.num_agents:
            return {agent_id: value[:, i] for i, agent_id in enumerate(runtime.agent_ids)}
        if value.ndim >= 2 and value.shape[-1] == runtime.num_agents:
            return {agent_id: value[:, i] for i, agent_id in enumerate(runtime.agent_ids)}
        if value.ndim == 1:
            return {agent_id: value for agent_id in runtime.agent_ids}
        if value.ndim >= 2 and runtime.num_agents > 1 and value.shape[-1] % runtime.num_agents == 0:
            reshaped = value.reshape(value.shape[0], runtime.num_agents, -1)
            if reshaped.shape[-1] == 1:
                reshaped = reshaped.squeeze(-1)
            return {agent_id: reshaped[:, i] for i, agent_id in enumerate(runtime.agent_ids)}
        return {agent_id: value.reshape(self.num_envs, -1).squeeze(-1) for agent_id in runtime.agent_ids}

    def _reduce_set_done_tensor(self, value: torch.Tensor, runtime, *, op: str) -> torch.Tensor:
        if value.dtype != torch.bool:
            value = value.to(dtype=torch.bool)
        if value.ndim == 1:
            return value
        if value.ndim >= 2 and value.shape[1] == runtime.num_agents:
            dims = tuple(range(1, value.ndim))
            return value.any(dim=dims) if op == "any" else value.all(dim=dims)
        if value.ndim >= 2 and value.shape[-1] == runtime.num_agents:
            return value.any(dim=-1) if op == "any" else value.all(dim=-1)
        reshaped = value.reshape(value.shape[0], runtime.num_agents, -1)
        dims = tuple(range(1, reshaped.ndim))
        return reshaped.any(dim=dims) if op == "any" else reshaped.all(dim=dims)

    def _reduce_tensor_list(self, tensors: list[torch.Tensor], *, op: str) -> torch.Tensor:
        if not tensors:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if len(tensors) == 1:
            return tensors[0]
        stacked = torch.stack(tensors, dim=0)
        if op == "any":
            return stacked.any(dim=0)
        if op == "all":
            return stacked.all(dim=0)
        raise ValueError(f"Unsupported done reduction op: {op!r}")
