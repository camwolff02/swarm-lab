# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classical manager-based multi-agent runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import ActionManager, CommandManager, ObservationManager, RecorderManager

from .manager_based_ma_env_cfg import (
    AgentGroupRuntimeSpec,
    ManagerBasedMaEnvCfg,
    MultiAgentRuntimeSpec,
    compile_multi_agent_spec,
)


class _AgentGroupEnvView:
    """Stable env-like proxy passed to managers for one execution group."""

    def __init__(self, root_env: ManagerBasedMaEnv, runtime: AgentGroupRuntimeSpec):
        object.__setattr__(self, "root", root_env)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "command_manager", None)
        object.__setattr__(self, "action_manager", None)
        object.__setattr__(self, "observation_manager", None)
        object.__setattr__(self, "reward_manager", None)
        object.__setattr__(self, "termination_manager", None)
        object.__setattr__(self, "curriculum_manager", None)

    @property
    def cfg(self) -> ManagerBasedMaEnvCfg:
        return self.root.cfg

    @property
    def scene(self):
        return self.root.scene

    @property
    def sim(self):
        return self.root.sim

    @property
    def device(self):
        return self.root.device

    @property
    def num_envs(self):
        return self.root.num_envs

    @property
    def physics_dt(self):
        return self.root.physics_dt

    @property
    def step_dt(self):
        return self.root.step_dt

    @property
    def common_step_counter(self):
        return self.root.common_step_counter

    @property
    def episode_length_buf(self):
        return self.root.episode_length_buf

    @property
    def reset_buf(self):
        return getattr(self.root, "reset_buf", None)

    @property
    def reset_terminated(self):
        return getattr(self.root, "reset_terminated", None)

    @property
    def reset_time_outs(self):
        return getattr(self.root, "reset_time_outs", None)

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self.runtime.agent_ids

    @property
    def agent_indices(self) -> tuple[int, ...]:
        return self.runtime.agent_indices

    @property
    def entity_names(self) -> tuple[str, ...]:
        return self.runtime.entity_names

    @property
    def num_agents(self) -> int:
        return self.runtime.num_agents

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.root, name)


@dataclass
class _ManagerBundle:
    """One pooled group of managers for one ``AgentGroupRuntimeSpec``."""

    runtime: AgentGroupRuntimeSpec
    env_view: _AgentGroupEnvView
    command_manager: CommandManager | None
    action_manager: ActionManager
    observation_manager: ObservationManager
    reward_manager: Any | None = None
    termination_manager: Any | None = None
    curriculum_manager: Any | None = None
    action_buffer: torch.Tensor | None = None


class ManagerBasedMaEnv(DirectMARLEnv):
    """Manager-backed classical multi-agent env with DirectMARLEnv ABI.

    Args:
        cfg: Manager-based multi-agent environment configuration.
        render_mode: Optional render mode passed through to DirectMARLEnv.
        **kwargs: Ignored compatibility kwargs.

    Notes:
        Rewards and dones are neutral placeholders required by DirectMARLEnv's
        PettingZoo-like ABI. Use ``ManagerBasedMarlEnv`` for RL terms.
    """

    is_vector_env = True
    cfg: ManagerBasedMaEnvCfg

    def __init__(self, cfg: ManagerBasedMaEnvCfg, render_mode: str | None = None, **kwargs):
        del kwargs
        self.render_mode = render_mode
        self.ma_spec: MultiAgentRuntimeSpec = compile_multi_agent_spec(cfg)
        self._populate_direct_cfg_from_spec(cfg)
        self.possible_agents = list(self.ma_spec.possible_agents)
        self.agents = list(self.possible_agents)
        self._obs_group_name = cfg.observation_group
        self._active_mask_key = cfg.active_agent_mask_key
        self._manager_bundles: dict[str, _ManagerBundle] = {}
        self._agent_to_bundle: dict[str, str] = {}
        self._agent_to_local_index: dict[str, int] = {}
        self._agent_to_global_index = {agent: i for i, agent in enumerate(self.possible_agents)}
        self._ma_metadata_maps = self._build_agent_metadata_maps()
        self._zero_rewards: dict[str, torch.Tensor] | None = None
        self._false_dones: dict[str, torch.Tensor] | None = None
        self.obs_dict: dict[str, Any] = {}
        self.state_buf: Any = None
        super().__init__(cfg=cfg, render_mode=render_mode)

    @property
    def unwrapped(self) -> ManagerBasedMaEnv:
        """Return the base environment, matching Gymnasium's convention."""
        return self

    @property
    def num_agents(self) -> int:
        """Number of currently active public agents."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Maximum number of possible public agents."""
        return len(self.possible_agents)

    def observation_space(self, agent: str) -> gym.Space:
        """Return the public observation space for an agent.

        Args:
            agent: PettingZoo-style agent id.

        Returns:
            Gymnasium observation space for the requested agent.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """Return the public action space for an agent.

        Args:
            agent: PettingZoo-style agent id.

        Returns:
            Gymnasium action space for the requested agent.
        """
        return self.action_spaces[agent]

    # ---------------------------------------------------------------------
    # DirectMARLEnv hooks / setup
    # ---------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Hook for task-specific scene setup.

        DirectMARLEnv creates ``self.scene`` before calling this hook, so the
        generic manager-based MA environment must not create another scene here.
        """
        return

    def _configure_env_spaces(self) -> None:
        """Build pooled managers and derive per-agent public spaces."""
        self._build_manager_bundles()
        self._configure_multi_agent_spaces()
        self._zero_rewards = {
            agent_id: torch.zeros(self.num_envs, device=self.device) for agent_id in self.possible_agents
        }
        self._false_dones = {
            agent_id: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for agent_id in self.possible_agents
        }

    def _populate_direct_cfg_from_spec(self, cfg: ManagerBasedMaEnvCfg) -> None:
        """Fill DirectMARLEnvCfg fields that are derived from the compiled agents."""
        cfg.possible_agents = list(self.ma_spec.possible_agents)
        if getattr(cfg, "observation_spaces", None) is None:
            cfg.observation_spaces = {}
        if getattr(cfg, "action_spaces", None) is None:
            cfg.action_spaces = {}
        if getattr(cfg, "state_space", None) is None:
            cfg.state_space = 0

    def _build_manager_bundles(self) -> None:
        """Create one classical manager bundle per execution group."""
        if self._manager_bundles:
            return

        if getattr(self.cfg, "recorders", None) is not None and not hasattr(self, "recorder_manager"):
            self.recorder_manager = RecorderManager(self.cfg.recorders, self)

        runtimes = list(self.ma_spec.execution_groups.values())

        for runtime in runtimes:
            bundle = self._create_classical_manager_bundle(runtime)
            self._manager_bundles[runtime.name] = bundle
            for local_index, agent_id in enumerate(runtime.agent_ids):
                self._agent_to_bundle[agent_id] = runtime.name
                self._agent_to_local_index[agent_id] = local_index

        first_bundle = next(iter(self._manager_bundles.values()))
        self.command_manager = first_bundle.command_manager
        self.action_manager = first_bundle.action_manager
        self.observation_manager = first_bundle.observation_manager

    def _create_classical_manager_bundle(self, runtime: AgentGroupRuntimeSpec) -> _ManagerBundle:
        """Create classical managers for a pooled execution group."""
        agent_cfg = runtime.cfg
        env_view = _AgentGroupEnvView(self, runtime)

        command_manager = CommandManager(agent_cfg.commands, env_view) if agent_cfg.commands is not None else None
        env_view.command_manager = command_manager

        action_manager = ActionManager(agent_cfg.actions, env_view)
        env_view.action_manager = action_manager

        observation_manager = ObservationManager(agent_cfg.observations, env_view)
        env_view.observation_manager = observation_manager

        return _ManagerBundle(
            runtime=runtime,
            env_view=env_view,
            command_manager=command_manager,
            action_manager=action_manager,
            observation_manager=observation_manager,
        )

    def _configure_multi_agent_spaces(self) -> None:
        """Derive public Gymnasium spaces from pooled manager bundles."""
        self.observation_spaces: dict[str, gym.Space] = {}
        self.action_spaces: dict[str, gym.Space] = {}
        self.state_spaces: dict[str, gym.Space | None] = {}

        for bundle in self._manager_bundles.values():
            runtime = bundle.runtime
            obs_shape = self._infer_per_agent_obs_shape(bundle)
            action_shape = self._infer_per_agent_action_shape(bundle)
            bundle.action_buffer = torch.zeros((self.num_envs, runtime.num_agents, action_shape[0]), device=self.device)
            for agent_id in runtime.agent_ids:
                self.observation_spaces[agent_id] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
                )
                self.action_spaces[agent_id] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=action_shape, dtype=np.float32
                )
                self.state_spaces[agent_id] = None

        self.state_space = self._make_state_space()
        if self.state_space is not None:
            for agent_id in self.possible_agents:
                self.state_spaces[agent_id] = self.state_space

        self.cfg.observation_spaces = self.observation_spaces
        self.cfg.action_spaces = self.action_spaces
        self.cfg.state_space = self.state_space if self.state_space is not None else 0

    def _make_state_space(self) -> gym.Space | None:
        return None

    # ---------------------------------------------------------------------
    # DirectMARLEnv runtime hooks
    # ---------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for bundle in self._manager_bundles.values():
            group_action = self._pack_actions_for_bundle(actions, bundle)
            bundle.action_manager.process_action(group_action)

    def _apply_action(self) -> None:
        for bundle in self._manager_bundles.values():
            bundle.action_manager.apply_action()

    def _get_observations(self) -> dict[str, Any]:
        self._compute_commands(dt=self.step_dt)
        observations: dict[str, Any] = {}
        for bundle in self._manager_bundles.values():
            manager_out = bundle.observation_manager.compute(update_history=True)
            group_obs = manager_out[self._obs_group_name]
            observations.update(self._fanout_tensor(group_obs, bundle.runtime))
        self.obs_dict = observations
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        if self._zero_rewards is None:
            self._zero_rewards = {
                agent_id: torch.zeros(self.num_envs, device=self.device) for agent_id in self.possible_agents
            }
        return self._zero_rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self._false_dones is None:
            self._false_dones = {
                agent_id: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                for agent_id in self.possible_agents
            }
        return self._false_dones, self._false_dones

    def _get_states(self) -> Any:
        return None

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        self.scene.reset(env_ids)
        event_log: dict[str, Any] = {}
        if getattr(self.cfg, "events", None) is not None and hasattr(self, "event_manager"):
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
            event_log = self.event_manager.reset(env_ids)

        self.extras = {agent_id: {} for agent_id in self.possible_agents}
        group_logs: dict[str, dict[str, Any]] = {}
        for name, bundle in self._manager_bundles.items():
            log: dict[str, Any] = {}
            log.update(bundle.observation_manager.reset(env_ids))
            log.update(bundle.action_manager.reset(env_ids))
            if bundle.command_manager is not None:
                log.update(bundle.command_manager.reset(env_ids))
            group_logs[name] = log

        recorder_log = self.recorder_manager.reset(env_ids) if hasattr(self, "recorder_manager") else {}
        for agent_id in self.possible_agents:
            bundle_name = self._agent_to_bundle[agent_id]
            self.extras[agent_id]["log"] = group_logs.get(bundle_name, {})
            self.extras[agent_id]["recorder"] = recorder_log
            self.extras[agent_id]["event"] = event_log

        self.episode_length_buf[env_ids] = 0

    # ---------------------------------------------------------------------
    # Pooled helpers
    # ---------------------------------------------------------------------

    def _compute_commands(self, *, dt: float) -> None:
        for bundle in self._manager_bundles.values():
            if bundle.command_manager is not None:
                bundle.command_manager.compute(dt=dt)

    def _pack_actions_for_bundle(self, actions: dict[str, torch.Tensor], bundle: _ManagerBundle) -> torch.Tensor:
        runtime = bundle.runtime
        action_dim = self.action_spaces[runtime.agent_ids[0]].shape[0]
        if bundle.action_buffer is None or bundle.action_buffer.shape[-1] != action_dim:
            bundle.action_buffer = torch.zeros((self.num_envs, runtime.num_agents, action_dim), device=self.device)
        buffer = bundle.action_buffer
        buffer.zero_()
        for local_index, agent_id in enumerate(runtime.agent_ids):
            action = actions.get(agent_id)
            if action is None:
                continue
            action = torch.nan_to_num(action.to(self.device), nan=0.0, posinf=1.0, neginf=-1.0)
            buffer[:, local_index, :].copy_(action.reshape(self.num_envs, action_dim))

        masked = self._apply_active_mask(buffer, runtime)
        return masked.reshape(self.num_envs, runtime.num_agents * action_dim)

    def _apply_active_mask(self, tensor: torch.Tensor, runtime: AgentGroupRuntimeSpec) -> torch.Tensor:
        if self._active_mask_key is None or not hasattr(self, self._active_mask_key):
            return tensor
        mask = getattr(self, self._active_mask_key)
        if mask is None:
            return tensor
        indices = torch.as_tensor(runtime.agent_indices, device=self.device, dtype=torch.long)
        set_mask = mask.index_select(dim=1, index=indices)
        while set_mask.ndim < tensor.ndim:
            set_mask = set_mask.unsqueeze(-1)
        return tensor * set_mask.to(dtype=tensor.dtype)

    def _fanout_tensor(self, value: Any, runtime: AgentGroupRuntimeSpec) -> dict[str, Any]:
        if isinstance(value, dict):
            per_leaf = {key: self._fanout_tensor(item, runtime) for key, item in value.items()}
            return {agent_id: {key: leaf[agent_id] for key, leaf in per_leaf.items()} for agent_id in runtime.agent_ids}
        if not torch.is_tensor(value):
            return {agent_id: value for agent_id in runtime.agent_ids}
        if value.ndim >= 3 and value.shape[1] == runtime.num_agents:
            return {agent_id: value[:, i, ...] for i, agent_id in enumerate(runtime.agent_ids)}
        if value.ndim >= 2 and runtime.num_agents > 1 and value.shape[-1] % runtime.num_agents == 0:
            per_agent_dim = value.shape[-1] // runtime.num_agents
            reshaped = value.reshape(value.shape[0], runtime.num_agents, per_agent_dim)
            return {agent_id: reshaped[:, i, ...] for i, agent_id in enumerate(runtime.agent_ids)}
        return {agent_id: value for agent_id in runtime.agent_ids}

    def _infer_per_agent_action_shape(self, bundle: _ManagerBundle) -> tuple[int, ...]:
        total_dim = int(bundle.action_manager.total_action_dim)
        n = max(bundle.runtime.num_agents, 1)
        if total_dim % n != 0:
            if n == 1:
                return (total_dim,)
            raise ValueError(
                f"Action dimension {total_dim} for group {bundle.runtime.name!r} is not divisible by {n} agents. "
                "ManagerBasedMaEnv builds one manager bundle per concrete agent, so multi-agent action terms must "
                "expose per-agent dimensions."
            )
        return (total_dim // n,)

    def _infer_per_agent_obs_shape(self, bundle: _ManagerBundle) -> tuple[int, ...]:
        obs_dim = bundle.observation_manager.group_obs_dim[self._obs_group_name]
        if not isinstance(obs_dim, tuple):
            raise ValueError(
                f"Observation group {self._obs_group_name!r} for {bundle.runtime.name!r} must be concatenated."
            )
        if len(obs_dim) >= 2 and obs_dim[0] == bundle.runtime.num_agents:
            return tuple(obs_dim[1:])
        if len(obs_dim) == 1 and bundle.runtime.num_agents > 1 and obs_dim[0] % bundle.runtime.num_agents == 0:
            return (obs_dim[0] // bundle.runtime.num_agents,)
        return tuple(obs_dim)

    def _build_agent_metadata_maps(self) -> dict[str, Any]:
        return {
            "agent_to_group": {agent: spec.group_name for agent, spec in self.ma_spec.agents.items()},
            "groups": self.ma_spec.groups,
        }

    def close(self) -> None:
        if not getattr(self, "_is_closed", False):
            for bundle in getattr(self, "_manager_bundles", {}).values():
                bundle.env_view.command_manager = None
                bundle.env_view.action_manager = None
                bundle.env_view.observation_manager = None
                bundle.env_view.reward_manager = None
                bundle.env_view.termination_manager = None
                bundle.env_view.curriculum_manager = None
            self._manager_bundles.clear()
            super().close()
