"""skrl Runner integration for the quad swarm paper model factory.

The paper uses a homogeneous decentralized policy, but stock skrl IPPO builds
one optimizer per named agent. Reusing the exact same module instances across
agents therefore means multiple Adam optimizers mutate the same parameters with
separate optimizer state. The default factory path intentionally builds one
policy/value pair per drone for stable stock-runner training. Explicit
``share_parameters=True`` is still available for evaluation experiments, but it
is not the safe default training path.
"""

from __future__ import annotations

import warnings
from typing import Any

from skrl.models.torch import Model
from skrl.utils import set_seed
from skrl.utils.runner.torch import Runner

from environments.tasks.quad_swarm_paper.agents.shared_ippo import (
    SharedIPPOAgent,
    SharedIPPOTrainer,
    encoder_cfg_from_model_config,
    shared_homogeneous_ippo_enabled,
)
from environments.tasks.quad_swarm_paper.models.quad_swarm_skrl_models import (
    QuadSwarmDeterministicValue,
    QuadSwarmGaussianPolicy,
)

_MODEL_FACTORY_NAME = "quad_swarm_paper_attention"
_ORIGINAL_RUNNER_INIT = None
_ORIGINAL_GENERATE_MODELS = None


def _generate_quad_swarm_models(env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
    """Build paper policy/value modules for skrl's stock Runner."""
    if shared_homogeneous_ippo_enabled(cfg):
        raise NotImplementedError(
            "training.shared_homogeneous_ippo=True cannot use stock skrl IPPO. "
            "Use the quad swarm Runner constructor path, which installs the dedicated "
            "shared-policy/shared-optimizer trainer before stock model generation."
        )

    device = env.device
    possible_agents = env.possible_agents
    model_cfg = cfg.get("models", {})
    encoder_cfg = encoder_cfg_from_model_config(model_cfg)
    share_parameters = bool(model_cfg.get("share_parameters", False))
    if share_parameters:
        warnings.warn(
            "share_parameters=True reuses module instances across skrl IPPO agents. "
            "Stock skrl IPPO creates one optimizer per agent, so this is unsafe for training "
            "unless a custom shared-optimizer update path is used.",
            RuntimeWarning,
            stacklevel=2,
        )

    def make_pair(agent_id: str) -> dict[str, Model]:
        """Make pair."""
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
    global _ORIGINAL_GENERATE_MODELS, _ORIGINAL_RUNNER_INIT
    if getattr(Runner, "_quad_swarm_paper_patch", False):
        return

    _ORIGINAL_RUNNER_INIT = Runner.__init__
    _ORIGINAL_GENERATE_MODELS = Runner._generate_models

    def _init(self: Runner, env: Any, cfg: dict[str, Any]) -> None:
        """Init."""
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME and shared_homogeneous_ippo_enabled(cfg):
            self._env = env
            self._cfg = cfg
            set_seed(self._cfg.get("seed", None))
            self._cfg["agent"]["rewards_shaper"] = None
            self._agent = SharedIPPOAgent(env, self._cfg)
            self._models = {"shared": {"policy": self._agent.policy, "value": self._agent.value}}
            self._trainer = SharedIPPOTrainer(env, self._agent, self._cfg.get("trainer", {}))
            return
        _ORIGINAL_RUNNER_INIT(self, env, cfg)

    def _generate_models(self: Runner, env: Any, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        """Generate models."""
        if cfg.get("models", {}).get("factory") == _MODEL_FACTORY_NAME:
            return _generate_quad_swarm_models(env, cfg)
        return _ORIGINAL_GENERATE_MODELS(self, env, cfg)

    Runner.__init__ = _init
    Runner._generate_models = _generate_models
    Runner._quad_swarm_paper_patch = True
