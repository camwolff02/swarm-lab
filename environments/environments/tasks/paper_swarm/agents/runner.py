"""SKRL Runner hook for the paper_swarm attention model factory.

Patches SKRL's stock Runner to inject shared custom models (PaperGaussianPolicy
and PaperDeterministicValue) when the model config specifies
``factory: paper_swarm_attention``.  The standard SKRL MAPPO/IPPO agent handles
training; we only override model creation so that all agents share one policy
and one value network.
"""

from __future__ import annotations

from typing import Any

import skrl.utils.runner.torch.runner as skrl_runner_module
from skrl.models.torch import Model
from skrl.utils.runner.torch import Runner
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space
from torch.optim.lr_scheduler import ExponentialLR

from environments.tasks.paper_swarm.models import (
    PaperAttentionEncoderCfg,
    PaperDeterministicValue,
    PaperGaussianPolicy,
)

from .shared_mappo import PaperMAPPO

_MODEL_FACTORY_NAME = "paper_swarm_attention"
_ORIGINAL_GENERATE_MODELS = None
_ORIGINAL_GENERATE_AGENT = None


def _encoder_cfg_from_model_config(model_cfg: dict[str, Any]) -> PaperAttentionEncoderCfg:
    """Encoder cfg from model config."""
    return PaperAttentionEncoderCfg(
        num_drones=int(model_cfg.get("num_drones", 8)),
        max_neighbors=int(model_cfg.get("max_neighbors", 7)),
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
    first_agent = env.possible_agents[0]
    state_spaces = getattr(env, "state_spaces", {})
    shared_policy = PaperGaussianPolicy(
        env.observation_spaces[first_agent],
        env.action_spaces[first_agent],
        device,
        encoder_cfg=encoder_cfg,
        clip_log_std=bool(model_cfg.get("clip_log_std", False)),
        min_log_std=float(model_cfg.get("min_log_std", -20.0)),
        max_log_std=float(model_cfg.get("max_log_std", 2.0)),
    )
    shared_value = PaperDeterministicValue(
        state_spaces.get(first_agent) or env.observation_spaces[first_agent],
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
    return models


def _install_runner_patch() -> None:
    """Install a narrow model factory hook into SKRL's stock Runner."""
    global _ORIGINAL_GENERATE_MODELS, _ORIGINAL_GENERATE_AGENT
    skrl_runner_module.ExponentialLR = ExponentialLR
    if getattr(Runner, "_paper_swarm_patch", False):
        return

    _ORIGINAL_GENERATE_MODELS = Runner._generate_models
    _ORIGINAL_GENERATE_AGENT = Runner._generate_agent

    # Capture the raw function before replacing it so the patched version
    # can delegate non-mappo lookups without infinite recursion.
    _original_component = Runner._component

    def _patched_generate_models(self: Runner, env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME:
            return _generate_models(env, cfg)
        return _ORIGINAL_GENERATE_MODELS(self, env, cfg)

    def _patched_generate_agent(self: Runner, env: Any, cfg: dict[str, Any], models: dict[str, dict[str, Model]]):
        return _ORIGINAL_GENERATE_AGENT(self, env, cfg, models)

    def _patched_component(self: Runner, name: str):
        if name.lower() == "mappo":
            return PaperMAPPO
        return _original_component(self, name)

    Runner._generate_models = _patched_generate_models
    Runner._generate_agent = _patched_generate_agent
    Runner._component = _patched_component
    Runner._paper_swarm_patch = True
    _install_multi_agent_state_patch()


def _install_multi_agent_state_patch() -> None:
    """Allow SKRL's IsaacLab multi-agent wrapper to consume per-agent state dicts."""
    import skrl.envs.wrappers.torch.isaaclab_envs as isaaclab_envs

    wrapper_cls = isaaclab_envs.IsaacLabMultiAgentWrapper
    if getattr(wrapper_cls, "_paper_swarm_state_patch", False):
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
    wrapper_cls._paper_swarm_state_patch = True
