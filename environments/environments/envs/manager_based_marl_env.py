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

    This class adds reward, termination, and curriculum managers on top of the
    classical manager workflow. Critic inputs are regular observation manager
    groups named ``"critic"`` and are exposed through :meth:`state`.
    """

    critic_observation_group = "critic"
    cfg: ManagerBasedMarlEnvCfg

    def __init__(self, cfg: ManagerBasedMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, actions):
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_pre_step()
        observations, rewards, terminated, truncated, extras = super().step(actions)
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_post_step()
        self._track_done_causes(extras)
        return observations, rewards, terminated, truncated, extras

    def _track_done_causes(self, extras: dict[str, torch.Tensor]) -> None:
        """Write per-term done-cause counts into extras for TensorBoard."""
        for bundle in self._manager_bundles.values():
            tm = bundle.termination_manager
            for term_name in tm.active_terms:
                done = tm.get_term(term_name)
                key = f"done/{term_name}"
                extras[key] = done.float().mean().unsqueeze(0)

    def _create_classical_manager_bundle(self, runtime) -> _ManagerBundle:
        bundle = super()._create_classical_manager_bundle(runtime)
        agent_cfg = runtime.cfg

        bundle.reward_manager = RewardManager(agent_cfg.rewards, bundle.env_view)
        bundle.termination_manager = TerminationManager(agent_cfg.terminations, bundle.env_view)
        bundle.curriculum_manager = (
            CurriculumManager(agent_cfg.curriculum, bundle.env_view) if agent_cfg.curriculum is not None else None
        )

        bundle.env_view.reward_manager = bundle.reward_manager
        bundle.env_view.termination_manager = bundle.termination_manager
        bundle.env_view.curriculum_manager = bundle.curriculum_manager
        return bundle

    def _configure_multi_agent_spaces(self) -> None:
        super()._configure_multi_agent_spaces()
        critic_spaces = self._make_critic_state_spaces()
        if not critic_spaces:
            return
        self.state_spaces.update(critic_spaces)
        self.state_space = next(iter(critic_spaces.values()))
        self.cfg.state_space = self.state_space

    def _make_state_space(self) -> gym.Space | None:
        return None

    def _make_critic_state_spaces(self) -> dict[str, gym.Space]:
        critic_spaces: dict[str, gym.Space] = {}
        for bundle in self._manager_bundles.values():
            group_dims = bundle.observation_manager.group_obs_dim
            if self.critic_observation_group not in group_dims:
                continue
            obs_dim = group_dims[self.critic_observation_group]
            if not isinstance(obs_dim, tuple):
                raise ValueError(
                    f"Observation group {self.critic_observation_group!r} for {bundle.runtime.name!r} must be "
                    "concatenated to be exposed as critic state."
                )
            space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
            for agent_id in bundle.runtime.agent_ids:
                critic_spaces[agent_id] = space
        return critic_spaces

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
        reset_mode = self.cfg.reset_on
        if reset_mode not in ("any", "all"):
            raise ValueError(
                f"Unsupported multi-agent reset mode: {reset_mode!r}. "
                "V2 only supports env-level reset aggregation, not per-agent reset."
            )

        raw_terminated_by_group: list[torch.Tensor] = []
        raw_truncated_by_group: list[torch.Tensor] = []
        per_agent_terminated: dict[str, torch.Tensor] = {}
        per_agent_truncated: dict[str, torch.Tensor] = {}

        for bundle in self._manager_bundles.values():
            bundle.termination_manager.compute()
            raw_terminated = bundle.termination_manager.terminated
            raw_truncated = bundle.termination_manager.time_outs
            if reset_mode == "any":
                raw_terminated_by_group.append(self._reduce_group_done_tensor(raw_terminated, bundle.runtime, op="any"))
                raw_truncated_by_group.append(self._reduce_group_done_tensor(raw_truncated, bundle.runtime, op="any"))
            elif reset_mode == "all":
                raw_terminated_by_group.append(self._reduce_group_done_tensor(raw_terminated, bundle.runtime, op="all"))
                raw_truncated_by_group.append(self._reduce_group_done_tensor(raw_truncated, bundle.runtime, op="all"))
            per_agent_terminated.update(self._fanout_reward_or_done(raw_terminated, bundle.runtime, dtype=torch.bool))
            per_agent_truncated.update(self._fanout_reward_or_done(raw_truncated, bundle.runtime, dtype=torch.bool))

        if reset_mode == "any":
            global_terminated = self._reduce_tensor_list(raw_terminated_by_group, op="any")
            global_truncated = self._reduce_tensor_list(raw_truncated_by_group, op="any")
            terminated = {agent_id: global_terminated for agent_id in self.possible_agents}
            truncated = {agent_id: global_truncated for agent_id in self.possible_agents}
        elif reset_mode == "all":
            global_terminated = self._reduce_tensor_list(raw_terminated_by_group, op="all")
            global_truncated = self._reduce_tensor_list(raw_truncated_by_group, op="all")
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
        """Compute critic observations for the training state channel.

        Computed once (global critic state is agent-independent), then fanned
        out to every possible agent.
        """
        for bundle in self._manager_bundles.values():
            if self.critic_observation_group not in bundle.observation_manager.group_obs_dim:
                continue
            critic_obs = bundle.observation_manager.compute_group(
                self.critic_observation_group, update_history=False
            )
            result = {agent_id: critic_obs for agent_id in self.possible_agents}
            self.state_buf = result
            return result
        return None

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

    def _reduce_group_done_tensor(self, value: torch.Tensor, runtime, *, op: str) -> torch.Tensor:
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
