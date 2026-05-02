"""SKRL Runner hook for the formation-swarm attention model factory."""

from __future__ import annotations

import copy
from typing import Any

from skrl.models.torch import Model
import skrl.utils.runner.torch.runner as skrl_runner_module
from skrl.utils.runner.torch import Runner
from torch.optim.lr_scheduler import ExponentialLR

from .shared_mappo import FormationSharedMAPPO
from environments.tasks.formation_swarm.models import (
    FormationAttentionEncoderCfg,
    FormationDeterministicValue,
    FormationGaussianPolicy,
)

_MODEL_FACTORY_NAME = "formation_swarm_attention"
_ORIGINAL_GENERATE_MODELS = None
_ORIGINAL_GENERATE_AGENT = None


def _encoder_cfg_from_model_config(model_cfg: dict[str, Any]) -> FormationAttentionEncoderCfg:
    return FormationAttentionEncoderCfg(
        num_drones=int(model_cfg.get("num_drones", 3)),
        num_balls=int(model_cfg.get("num_balls", 2)),
        attention_dim=int(model_cfg.get("attention_dim", 32)),
        attention_heads=int(model_cfg.get("attention_heads", 1)),
        initial_log_std=float(model_cfg.get("initial_log_std", 0.0)),
    )


def _generate_models(env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
    device = env.device
    model_cfg = cfg.get("models", {})
    encoder_cfg = _encoder_cfg_from_model_config(model_cfg)
    hidden_units = tuple(int(v) for v in model_cfg.get("hidden_units", [256, 256, 256]))
    first_agent = env.possible_agents[0]
    shared_policy = FormationGaussianPolicy(
        env.observation_spaces[first_agent],
        env.action_spaces[first_agent],
        device,
        encoder_cfg=encoder_cfg,
    )
    shared_value = FormationDeterministicValue(
        env.observation_spaces[first_agent],
        env.action_spaces[first_agent],
        device,
        hidden_units=hidden_units,
    )

    models: dict[str, dict[str, Model]] = {}
    for agent_id in env.possible_agents:
        models[agent_id] = {
            "policy": shared_policy,
            "value": shared_value,
        }

    seen: set[int] = set()
    for agent_models in models.values():
        for role, model in agent_models.items():
            if id(model) in seen:
                continue
            seen.add(id(model))
            model.init_state_dict(role=role)
    return models


def _generate_shared_mappo_agent(self: Runner, env: Any, cfg: dict[str, Any], models: dict[str, dict[str, Model]]) -> FormationSharedMAPPO:
    device = env.device
    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    memory_cfg = copy.deepcopy(cfg.get("memory", {"class": "RandomMemory", "memory_size": -1}))
    memory_class = self._component(memory_cfg.pop("class", "RandomMemory"))
    if memory_cfg["memory_size"] < 0:
        memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
    memories = {
        agent_id: memory_class(num_envs=env.num_envs, device=device, **self._process_cfg(memory_cfg))
        for agent_id in possible_agents
    }

    agent_cfg = self._component("MAPPO_DEFAULT_CONFIG").copy()
    agent_cfg.update(self._process_cfg(cfg["agent"]))
    agent_cfg["state_preprocessor_kwargs"].update(
        {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
    )
    agent_cfg["shared_state_preprocessor_kwargs"].update(
        {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
    )
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
    return FormationSharedMAPPO(
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        shared_observation_spaces=observation_spaces,
        possible_agents=possible_agents,
        cfg=agent_cfg,
        device=device,
    )


def install_formation_swarm_runner_patch() -> None:
    """Install a narrow model factory hook into SKRL's stock Runner."""

    global _ORIGINAL_GENERATE_AGENT, _ORIGINAL_GENERATE_MODELS
    skrl_runner_module.ExponentialLR = ExponentialLR
    if getattr(Runner, "_formation_swarm_patch", False):
        return

    _ORIGINAL_GENERATE_MODELS = Runner._generate_models
    _ORIGINAL_GENERATE_AGENT = Runner._generate_agent

    def _patched_generate_models(self: Runner, env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME:
            return _generate_models(env, cfg)
        return _ORIGINAL_GENERATE_MODELS(self, env, cfg)

    def _patched_generate_agent(
        self: Runner, env: Any, cfg: dict[str, Any], models: dict[str, dict[str, Model]]
    ) -> FormationSharedMAPPO:
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME:
            return _generate_shared_mappo_agent(self, env, cfg, models)
        return _ORIGINAL_GENERATE_AGENT(self, env, cfg, models)

    Runner._generate_models = _patched_generate_models
    Runner._generate_agent = _patched_generate_agent
    Runner._formation_swarm_patch = True
