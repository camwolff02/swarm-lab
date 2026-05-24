# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based multi-agent environment runtime."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)

from .manager_based_ma_env_cfg import ManagerBasedMaEnvCfg, MultiAgentRuntimeSpec, compile_multi_agent_spec


class ManagerBasedMaEnv(ManagerBasedRLEnv):
    """Manager-based multi-agent environment with PettingZoo Parallel-style API.

    Supports heterogeneous agents with per-agent manager bundles for commands,
    actions, observations, terminations, rewards, and curricula. Agent context
    switching via :meth:`_agent_context` enables reuse of the same manager
    attribute namespace across agents during step and computation cycles.
    """

    is_vector_env = True
    cfg: ManagerBasedMaEnvCfg

    def __init__(self, cfg: ManagerBasedMaEnvCfg, render_mode: str | None = None, **kwargs):
        del kwargs
        self.render_mode = render_mode
        self.ma_spec: MultiAgentRuntimeSpec = compile_multi_agent_spec(cfg)
        self.possible_agents = list(self.ma_spec.possible_agents)
        self.agents = list(self.possible_agents)
        self._agent_managers: dict[str, dict[str, Any]] = {}
        self._active_agent_id: str | None = None
        self._obs_group_name = cfg.ma_options.observation_group
        self._active_mask_key = cfg.ma_options.active_agent_mask_key
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.reset_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.bool)
        self.reset_terminated = torch.zeros_like(self.reset_buf)
        self.reset_time_outs = torch.zeros_like(self.reset_buf)
        self.reward_dict: dict[str, torch.Tensor] = {}
        self.terminated_dict: dict[str, torch.Tensor] = {}
        self.time_out_dict: dict[str, torch.Tensor] = {}
        self.obs_dict: dict[str, torch.Tensor] = {}
        self.state_buf: torch.Tensor | dict[str, torch.Tensor] | None = None
        super().__init__(cfg=cfg, render_mode=render_mode)

    @property
    def unwrapped(self) -> "ManagerBasedMaEnv":
        return self

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    @property
    def max_episode_length_s(self) -> float:
        return getattr(self.cfg, "episode_length_s", math.inf)

    @property
    def max_episode_length(self) -> int:
        if not math.isfinite(self.max_episode_length_s):
            return 2**63 - 1
        return math.ceil(self.max_episode_length_s / self.step_dt)

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def load_managers(self):
        """Create one manager bundle per concrete agent."""

        print("[INFO] Event Manager: ", self.event_manager)
        from isaaclab.managers import RecorderManager

        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)

        for agent_id in self.possible_agents:
            spec = self.ma_spec.agents[agent_id]
            profile = spec.profile
            with self._agent_context(agent_id, allow_missing=True):
                command_manager = CommandManager(profile.commands, self)
                action_manager = ActionManager(profile.actions, self)
            self._agent_managers[agent_id] = {
                "command": command_manager,
                "action": action_manager,
                "observation": None,
                "termination": None,
                "reward": None,
                "curriculum": None,
            }

        for agent_id in self.possible_agents:
            spec = self.ma_spec.agents[agent_id]
            profile = spec.profile
            with self._agent_context(agent_id):
                observation_manager = ObservationManager(profile.observations, self)
                termination_manager = TerminationManager(profile.terminations, self) if profile.terminations else None
                reward_manager = RewardManager(profile.rewards, self) if profile.rewards else None
                curriculum_manager = CurriculumManager(profile.curriculum, self) if profile.curriculum else None
            self._agent_managers[agent_id].update(
                {
                    "observation": observation_manager,
                    "termination": termination_manager,
                    "reward": reward_manager,
                    "curriculum": curriculum_manager,
                }
            )

        self.command_manager = self._agent_managers[self.possible_agents[0]]["command"]
        self.action_manager = self._agent_managers[self.possible_agents[0]]["action"]
        self.observation_manager = self._agent_managers[self.possible_agents[0]]["observation"]
        self.termination_manager = self._agent_managers[self.possible_agents[0]]["termination"]
        self.reward_manager = self._agent_managers[self.possible_agents[0]]["reward"]
        self.curriculum_manager = self._agent_managers[self.possible_agents[0]]["curriculum"]

        self._configure_multi_agent_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    @contextlib.contextmanager
    def _agent_context(self, agent_id: str, *, allow_missing: bool = False) -> Iterator[None]:
        old_id = self._active_agent_id
        old_managers = {
            name: getattr(self, name, None)
            for name in (
                "command_manager",
                "action_manager",
                "observation_manager",
                "termination_manager",
                "reward_manager",
                "curriculum_manager",
            )
        }
        self._active_agent_id = agent_id
        bundle = self._agent_managers.get(agent_id)
        if bundle is not None:
            self.command_manager = bundle["command"]
            self.action_manager = bundle["action"]
            self.observation_manager = bundle["observation"]
            self.termination_manager = bundle["termination"]
            self.reward_manager = bundle["reward"]
            self.curriculum_manager = bundle["curriculum"]
        elif not allow_missing:
            raise KeyError(agent_id)
        try:
            yield
        finally:
            self._active_agent_id = old_id
            for name, value in old_managers.items():
                if value is not None:
                    setattr(self, name, value)

    def _configure_multi_agent_spaces(self) -> None:
        self.observation_spaces: dict[str, gym.Space] = {}
        self.action_spaces: dict[str, gym.Space] = {}
        self.state_spaces: dict[str, gym.Space | None] = {}

        for agent_id in self.possible_agents:
            obs_manager = self._agent_managers[agent_id]["observation"]
            action_manager = self._agent_managers[agent_id]["action"]
            obs_dim = obs_manager.group_obs_dim[self._obs_group_name]
            if not isinstance(obs_dim, tuple):
                raise ValueError(f"Observation group {self._obs_group_name!r} for {agent_id!r} must be concatenated.")
            self.observation_spaces[agent_id] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
            self.action_spaces[agent_id] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(action_manager.total_action_dim,), dtype=np.float32
            )

        self.state_space = self._make_state_space()
        for agent_id in self.possible_agents:
            self.state_spaces[agent_id] = self.state_space

    def _make_state_space(self) -> gym.Space | None:
        state_cfg = getattr(self.cfg, "state", None)
        if state_cfg is None or not state_cfg.state_space:
            return None
        if state_cfg.mode == "concat_policy":
            dim = sum(int(np.prod(space.shape)) for space in self.observation_spaces.values())
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        if state_cfg.mode == "observation_group":
            first = self.possible_agents[0]
            obs_manager = self._agent_managers[first]["observation"]
            dim = obs_manager.group_obs_dim[state_cfg.group_name]
            if not isinstance(dim, tuple):
                raise ValueError(f"State observation group {state_cfg.group_name!r} must be concatenated.")
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=dim, dtype=np.float32)
        if isinstance(state_cfg.state_space, gym.Space):
            return state_cfg.state_space
        if isinstance(state_cfg.state_space, int) and state_cfg.state_space > 0:
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_cfg.state_space,), dtype=np.float32)
        return None

    def _get_agent_metadata_maps(self) -> dict[str, Any]:
        return {
            "agent_to_policy": {agent: spec.policy for agent, spec in self.ma_spec.agents.items()},
            "agent_to_team": {agent: spec.team for agent, spec in self.ma_spec.agents.items()},
            "policy_to_agents": self.ma_spec.policies,
            "team_to_agents": self.ma_spec.teams,
            "sets": self.ma_spec.sets,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.seed(seed)
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.recorder_manager.record_pre_reset(env_ids)
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        self.recorder_manager.record_post_reset(env_ids)
        self.obs_dict = self._compute_observations(update_history=True)
        self.agents = list(self.possible_agents)
        return self.obs_dict, self.extras

    def step(self, actions: dict[str, torch.Tensor]):
        actions = {agent: action.to(self.device) for agent, action in actions.items()}
        for agent_id in self.possible_agents:
            action = actions.get(agent_id)
            if action is None:
                action = torch.zeros((self.num_envs, self.action_spaces[agent_id].shape[0]), device=self.device)
            if self._active_mask_key is not None and hasattr(self, self._active_mask_key):
                index = self.possible_agents.index(agent_id)
                mask = getattr(self, self._active_mask_key)[:, index].unsqueeze(-1)
                action = torch.where(mask, action, torch.zeros_like(action))
            with self._agent_context(agent_id):
                self.action_manager.process_action(torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0))

        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.is_rendering
        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self._apply_agent_actions()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self._apply_agent_actions()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.terminated_dict, self.time_out_dict = self._compute_dones()
        self.reset_terminated = torch.stack(list(self.terminated_dict.values()), dim=0).any(dim=0)
        self.reset_time_outs = torch.stack(list(self.time_out_dict.values()), dim=0).any(dim=0)
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_dict = self._compute_rewards()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_dict = self._compute_observations(update_history=False)
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self._compute_commands(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_dict = self._compute_observations(update_history=True)
        self.agents = list(self.possible_agents)
        return self.obs_dict, self.reward_dict, self.terminated_dict, self.time_out_dict, self.extras

    def _apply_agent_actions(self) -> None:
        for agent_id in self.possible_agents:
            with self._agent_context(agent_id):
                self.action_manager.apply_action()

    def _compute_observations(self, *, update_history: bool) -> dict[str, torch.Tensor]:
        obs = {}
        for agent_id in self.possible_agents:
            with self._agent_context(agent_id):
                obs[agent_id] = self.observation_manager.compute(update_history=update_history)[self._obs_group_name]
        return obs

    def _compute_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}
        for agent_id in self.possible_agents:
            manager = self._agent_managers[agent_id]["reward"]
            if manager is None:
                rewards[agent_id] = torch.zeros(self.num_envs, device=self.device)
                continue
            with self._agent_context(agent_id):
                rewards[agent_id] = manager.compute(dt=self.step_dt)
        return rewards

    def _compute_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {}
        truncated = {}
        for agent_id in self.possible_agents:
            manager = self._agent_managers[agent_id]["termination"]
            if manager is None:
                terminated[agent_id] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                truncated[agent_id] = torch.zeros_like(terminated[agent_id])
                continue
            with self._agent_context(agent_id):
                manager.compute()
                terminated[agent_id] = manager.terminated.clone()
                truncated[agent_id] = manager.time_outs.clone()

        reset_mode = self.cfg.ma_options.reset_on
        done_stack = torch.stack(list(terminated.values()), dim=0)
        timeout_stack = torch.stack(list(truncated.values()), dim=0)
        if reset_mode in ("global", "any_agent"):
            global_terminated = done_stack.any(dim=0)
            global_truncated = timeout_stack.any(dim=0)
            terminated = {agent_id: global_terminated.clone() for agent_id in self.possible_agents}
            truncated = {agent_id: global_truncated.clone() for agent_id in self.possible_agents}
        elif reset_mode == "all_agents":
            global_terminated = done_stack.all(dim=0)
            global_truncated = timeout_stack.all(dim=0)
            terminated = {agent_id: global_terminated.clone() for agent_id in self.possible_agents}
            truncated = {agent_id: global_truncated.clone() for agent_id in self.possible_agents}
        elif reset_mode != "per_agent":
            raise ValueError(f"Unsupported multi-agent reset mode: {reset_mode!r}")
        return terminated, truncated

    def _compute_commands(self, *, dt: float) -> None:
        for agent_id in self.possible_agents:
            with self._agent_context(agent_id):
                self.command_manager.compute(dt=dt)

    def _reset_idx(self, env_ids: Sequence[int]):
        self._compute_curriculum(env_ids)
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        metadata_maps = self._get_agent_metadata_maps() if self.cfg.ma_options.expose_agent_metadata_maps else None
        self.extras = {agent: {} for agent in self.possible_agents}
        for agent_id in self.possible_agents:
            if metadata_maps is not None:
                self.extras[agent_id]["ma"] = metadata_maps
            with self._agent_context(agent_id):
                log = {}
                log.update(self.observation_manager.reset(env_ids))
                log.update(self.action_manager.reset(env_ids))
                manager = self._agent_managers[agent_id]["reward"]
                if manager is not None:
                    log.update(manager.reset(env_ids))
                manager = self._agent_managers[agent_id]["curriculum"]
                if manager is not None:
                    log.update(manager.reset(env_ids))
                log.update(self.command_manager.reset(env_ids))
                manager = self._agent_managers[agent_id]["termination"]
                if manager is not None:
                    log.update(manager.reset(env_ids))
                log.update(self.recorder_manager.reset(env_ids))
            self.extras[agent_id]["log"] = log

        event_log = self.event_manager.reset(env_ids)
        for agent_id in self.possible_agents:
            self.extras[agent_id]["event"] = event_log
        self.episode_length_buf[env_ids] = 0

    def _compute_curriculum(self, env_ids: Sequence[int]) -> None:
        for agent_id in self.possible_agents:
            manager = self._agent_managers.get(agent_id, {}).get("curriculum")
            if manager is None:
                continue
            with self._agent_context(agent_id):
                manager.compute(env_ids=env_ids)

    def close(self):
        if not self._is_closed:
            for bundle in getattr(self, "_agent_managers", {}).values():
                bundle.clear()
            super().close()
