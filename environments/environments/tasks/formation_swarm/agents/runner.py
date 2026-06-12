"""SKRL Runner hook for the formation-swarm attention model factory.

Patches SKRL's stock Runner to inject shared custom models when the model config
specifies ``factory: formation_swarm_attention``.  The standard SKRL MAPPO agent
handles training; we only override model creation so that all agents share one
policy and one value network.
"""

from __future__ import annotations

from typing import Any

import skrl.utils.runner.torch.runner as skrl_runner_module
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper
from skrl.models.torch import Model
from skrl.utils.runner.torch import Runner
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space
from torch.optim.lr_scheduler import ExponentialLR

from environments.tasks.formation_swarm.models import (
    FormationAttentionEncoderCfg,
    FormationDeterministicValue,
    FormationGaussianPolicy,
)

from .shared_mappo import FormationMAPPO

_MODEL_FACTORY_NAME = "formation_swarm_attention"
_ORIGINAL_GENERATE_MODELS = None
_ORIGINAL_GENERATE_AGENT = None


def _encoder_cfg_from_model_config(model_cfg: dict[str, Any]) -> FormationAttentionEncoderCfg:
    """Encoder cfg from model config."""
    return FormationAttentionEncoderCfg(
        num_drones=int(model_cfg.get("num_drones", 3)),
        num_balls=int(model_cfg.get("num_balls", 2)),
        attention_dim=int(model_cfg.get("attention_dim", 32)),
        attention_heads=int(model_cfg.get("attention_heads", 1)),
        initial_log_std=float(model_cfg.get("initial_log_std", 0.0)),
    )


def _generate_models(env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
    """Generate shared policy and value models for the attention factory."""
    device = env.device
    model_cfg = cfg.get("models", {})
    encoder_cfg = _encoder_cfg_from_model_config(model_cfg)
    hidden_units = tuple(int(v) for v in model_cfg.get("hidden_units", [256, 256, 256]))
    multi_agent = isinstance(env, MultiAgentEnvWrapper)
    if multi_agent:
        # Access spaces via env.unwrapped (the raw Env) instead of wrapper
        # properties, which delegate to self._unwrapped and can fail when
        # gymnasium's OrderEnforcing wrapper sits between the SKRL wrapper
        # and the DirectMARLEnv.
        raw = env.unwrapped
        possible_agents = raw.possible_agents
        observation_spaces = raw.observation_spaces
        state_spaces = raw.state_spaces
        action_spaces = raw.action_spaces
    else:
        possible_agents = ["agent"]
        observation_spaces = {"agent": env.observation_space}
        state_spaces = {"agent": env.state_space}
        action_spaces = {"agent": env.action_space}
    first_agent = possible_agents[0]
    shared_policy = FormationGaussianPolicy(
        observation_spaces[first_agent],
        action_spaces[first_agent],
        device,
        encoder_cfg=encoder_cfg,
        clip_log_std=bool(model_cfg.get("clip_log_std", False)),
        min_log_std=float(model_cfg.get("min_log_std", -20.0)),
        max_log_std=float(model_cfg.get("max_log_std", 2.0)),
    )
    shared_value = FormationDeterministicValue(
        state_spaces.get(first_agent) or observation_spaces[first_agent],
        action_spaces[first_agent],
        device,
        hidden_units=hidden_units,
    )

    models: dict[str, dict[str, Model]] = {}
    for agent_id in possible_agents:
        models[agent_id] = {
            "policy": shared_policy,
            "value": shared_value,
        }
    return models


def install_formation_swarm_runner_patch() -> None:
    """Install a narrow model factory hook into SKRL's stock Runner."""
    global _ORIGINAL_GENERATE_MODELS, _ORIGINAL_GENERATE_AGENT
    skrl_runner_module.ExponentialLR = ExponentialLR
    if getattr(Runner, "_formation_swarm_patch", False):
        return

    _ORIGINAL_GENERATE_MODELS = Runner._generate_models
    _ORIGINAL_GENERATE_AGENT = Runner._generate_agent
    _original_component = Runner._component

    def _patched_generate_models(self: Runner, env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME:
            return _generate_models(env, cfg)
        return _ORIGINAL_GENERATE_MODELS(self, env, cfg)

    def _patched_generate_agent(self: Runner, env: Any, cfg: dict[str, Any], models: dict[str, dict[str, Model]]):
        return _ORIGINAL_GENERATE_AGENT(self, env, cfg, models)

    def _patched_component(self: Runner, name: str):
        if name.lower() in {"mappo", "formationsharedmappo"}:
            return FormationMAPPO
        return _original_component(self, name)

    Runner._generate_models = _patched_generate_models
    Runner._generate_agent = _patched_generate_agent
    Runner._component = _patched_component
    Runner._formation_swarm_patch = True
    _install_multi_agent_state_patch()


def _install_multi_agent_state_patch() -> None:
    """Allow SKRL's IsaacLab multi-agent wrapper to consume per-agent state dicts."""
    import skrl.envs.wrappers.torch.isaaclab_envs as isaaclab_envs

    wrapper_cls = isaaclab_envs.IsaacLabMultiAgentWrapper
    if getattr(wrapper_cls, "_formation_swarm_state_patch", False):
        return

    def _state(self):
        try:
            state = self._env.state()
        except AttributeError:
            state = self._unwrapped.state()
        if state is None:
            return {uid: None for uid in self.possible_agents}
        if isinstance(state, dict):
            return {
                uid: flatten_tensorized_space(tensorize_space(self.state_spaces[uid], state[uid]))
                if state.get(uid) is not None
                else None
                for uid in self.possible_agents
            }
        state = flatten_tensorized_space(tensorize_space(next(iter(self.state_spaces.values())), state))
        return {uid: state for uid in self.possible_agents}

    wrapper_cls.state = _state
    wrapper_cls._formation_swarm_state_patch = True
