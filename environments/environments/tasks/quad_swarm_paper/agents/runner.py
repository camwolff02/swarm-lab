"""skrl Runner integration for the quad swarm paper model factory."""

from __future__ import annotations

from typing import Any

from skrl.models.torch import Model
from skrl.utils.runner.torch import Runner

from environments.tasks.quad_swarm_paper.models.quad_swarm_encoder import QuadSwarmEncoderCfg
from environments.tasks.quad_swarm_paper.models.quad_swarm_skrl_models import (
    QuadSwarmDeterministicValue,
    QuadSwarmGaussianPolicy,
)

_MODEL_FACTORY_NAME = "quad_swarm_paper_attention"
_ORIGINAL_GENERATE_MODELS = None


def _generate_quad_swarm_models(env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
    """Build shared paper policy/value modules for skrl's stock Runner."""

    device = env.device
    possible_agents = env.possible_agents
    model_cfg = cfg.get("models", {})
    encoder_cfg = QuadSwarmEncoderCfg(
        self_obs_dim=int(model_cfg.get("self_obs_dim", 19)),
        neighbor_obs_dim=int(model_cfg.get("neighbor_obs_dim", 12)),
        obstacle_obs_dim=int(model_cfg.get("obstacle_obs_dim", 9)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
    )
    share_parameters = bool(model_cfg.get("share_parameters", True))

    def make_pair(agent_id: str) -> dict[str, Model]:
        return {
            "policy": QuadSwarmGaussianPolicy(
                env.observation_spaces[agent_id],
                env.action_spaces[agent_id],
                device,
                encoder_cfg=encoder_cfg,
            ),
            "value": QuadSwarmDeterministicValue(
                env.observation_spaces[agent_id],
                env.action_spaces[agent_id],
                device,
                encoder_cfg=encoder_cfg,
            ),
        }

    if share_parameters:
        shared_pair = make_pair(possible_agents[0])
        models = {agent_id: dict(shared_pair) for agent_id in possible_agents}
    else:
        models = {agent_id: make_pair(agent_id) for agent_id in possible_agents}

    initialized: set[int] = set()
    for agent_models in models.values():
        for role, model in agent_models.items():
            model_id = id(model)
            if model_id not in initialized:
                model.init_state_dict(role=role)
                initialized.add(model_id)
    return models


def install_quad_swarm_runner_patch() -> None:
    """Install a narrow quad-swarm model factory hook into skrl's stock Runner."""

    global _ORIGINAL_GENERATE_MODELS
    if getattr(Runner, "_quad_swarm_paper_patch", False):
        return

    _ORIGINAL_GENERATE_MODELS = Runner._generate_models

    def _generate_models(self: Runner, env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME:
            return _generate_quad_swarm_models(env, cfg)
        return _ORIGINAL_GENERATE_MODELS(self, env, cfg)

    Runner._generate_models = _generate_models
    Runner._quad_swarm_paper_patch = True
