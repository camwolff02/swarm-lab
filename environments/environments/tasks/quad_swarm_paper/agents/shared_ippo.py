"""Phase 1 shared-IPPO ownership and collation plumbing.

This module intentionally does not implement PPO updates. It provides the
single shared policy/value objects, one optimizer per role, and shape-safe
helpers needed by the later shared trainer. Stock skrl multi-agent IPPO keeps
per-agent optimizer/update streams, so shared mode must not be routed through
that path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from skrl.models.torch import Model

from environments.tasks.quad_swarm_paper.models.quad_swarm_encoder import QuadSwarmEncoderCfg
from environments.tasks.quad_swarm_paper.models.quad_swarm_skrl_models import (
    QuadSwarmDeterministicValue,
    QuadSwarmGaussianPolicy,
)

SHARED_HOMOGENEOUS_IPPO_KEY = "shared_homogeneous_ippo"


@dataclass(frozen=True)
class SharedIPPOComponents:
    """Shared model/optimizer ownership for the homogeneous decentralized policy."""

    agent_ids: tuple[str, ...]
    policy: QuadSwarmGaussianPolicy
    value: QuadSwarmDeterministicValue
    policy_optimizer: torch.optim.Optimizer
    value_optimizer: torch.optim.Optimizer


def shared_homogeneous_ippo_enabled(cfg: Mapping[str, Any]) -> bool:
    """Return whether the explicit shared-training mode is requested."""

    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, Mapping):
        return False
    return bool(training_cfg.get(SHARED_HOMOGENEOUS_IPPO_KEY, False))


def encoder_cfg_from_model_config(model_cfg: Mapping[str, Any]) -> QuadSwarmEncoderCfg:
    """Build the paper encoder config from the skrl YAML model section."""

    return QuadSwarmEncoderCfg(
        self_obs_dim=int(model_cfg.get("self_obs_dim", 19)),
        neighbor_obs_dim=int(model_cfg.get("neighbor_obs_dim", 12)),
        obstacle_obs_dim=int(model_cfg.get("obstacle_obs_dim", 9)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        initial_log_std=float(model_cfg.get("initial_log_std", -1.0)),
        init_policy_to_hover=bool(model_cfg.get("init_policy_to_hover", True)),
    )


def build_shared_ippo_components(env: Any, cfg: Mapping[str, Any]) -> SharedIPPOComponents:
    """Create exactly one policy, one value function, and one optimizer per role."""

    agent_ids = tuple(env.possible_agents)
    if not agent_ids:
        raise ValueError("Shared IPPO requires at least one agent in env.possible_agents.")

    _validate_homogeneous_spaces(env, agent_ids)
    model_cfg = cfg.get("models", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    agent_cfg = cfg.get("agent", {})
    if not isinstance(agent_cfg, Mapping):
        agent_cfg = {}

    encoder_cfg = encoder_cfg_from_model_config(model_cfg)
    reference_agent = agent_ids[0]
    policy = QuadSwarmGaussianPolicy(
        env.observation_spaces[reference_agent],
        env.action_spaces[reference_agent],
        env.device,
        encoder_cfg=encoder_cfg,
    )
    value = QuadSwarmDeterministicValue(
        env.observation_spaces[reference_agent],
        env.action_spaces[reference_agent],
        env.device,
        encoder_cfg=encoder_cfg,
    )
    policy.init_state_dict(role="policy")
    value.init_state_dict(role="value")

    learning_rate = float(agent_cfg.get("learning_rate", 1.0e-4))
    return SharedIPPOComponents(
        agent_ids=agent_ids,
        policy=policy,
        value=value,
        # Unlike stock skrl IPPO, these optimizers are owned once globally,
        # not duplicated for each named drone.
        policy_optimizer=torch.optim.Adam(policy.parameters(), lr=learning_rate),
        value_optimizer=torch.optim.Adam(value.parameters(), lr=learning_rate),
    )


def stack_agent_observations(obs_dict: Mapping[str, torch.Tensor], agent_ids: Sequence[str]) -> torch.Tensor:
    """Stack per-agent observation dicts into ``[num_envs, num_agents, obs_dim]``.

    The ordering is always the explicit ``agent_ids`` order, expected to be
    ``env.possible_agents`` for the quad swarm task.
    """

    tensors = _ordered_agent_tensors(obs_dict, agent_ids, label="observations")
    return torch.stack(tensors, dim=1)


def flatten_agent_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten ``[num_envs, num_agents, ...]`` to ``[num_envs * num_agents, ...]``."""

    if tensor.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions [E, N, ...], got shape {tuple(tensor.shape)}.")
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def unflatten_agent_batch(tensor: torch.Tensor, *, num_envs: int, agent_ids: Sequence[str]) -> torch.Tensor:
    """Unflatten ``[num_envs * num_agents, ...]`` to ``[num_envs, num_agents, ...]``."""

    num_agents = len(agent_ids)
    if num_envs <= 0 or num_agents <= 0:
        raise ValueError("num_envs and agent_ids must describe a non-empty [E, N] batch.")
    expected = num_envs * num_agents
    if tensor.shape[0] != expected:
        raise ValueError(f"Expected leading dimension {expected}, got {tensor.shape[0]}.")
    return tensor.reshape(num_envs, num_agents, *tensor.shape[1:])


def unstack_agent_actions(action_tensor: torch.Tensor, agent_ids: Sequence[str]) -> dict[str, torch.Tensor]:
    """Convert ``[num_envs, num_agents, act_dim]`` actions back to the env dict API."""

    if action_tensor.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions [E, N, ...], got shape {tuple(action_tensor.shape)}.")
    if action_tensor.shape[1] != len(agent_ids):
        raise ValueError(
            f"Action tensor agent dimension {action_tensor.shape[1]} does not match {len(agent_ids)} agent ids."
        )
    return {agent_id: action_tensor[:, index].contiguous() for index, agent_id in enumerate(agent_ids)}


def flatten_rollout_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten pooled rollout tensors ``[T, E, N, ...]`` to ``[T * E * N, ...]``."""

    if tensor.ndim < 4:
        raise ValueError(f"Expected at least 4 dimensions [T, E, N, ...], got shape {tuple(tensor.shape)}.")
    return tensor.reshape(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], *tensor.shape[3:])


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    """Return optimizer-owned parameter object ids in param-group order."""

    return tuple(id(param) for group in optimizer.param_groups for param in group["params"])


def assert_optimizer_owns_model_once(model: Model, optimizer: torch.optim.Optimizer) -> None:
    """Validate that an optimizer owns each model parameter exactly once."""

    model_param_ids = tuple(id(param) for param in model.parameters())
    optimizer_param_ids = optimizer_parameter_ids(optimizer)
    if len(optimizer_param_ids) != len(set(optimizer_param_ids)):
        raise ValueError("Optimizer contains duplicate parameter references.")
    if set(optimizer_param_ids) != set(model_param_ids):
        raise ValueError("Optimizer parameter ownership does not match the model parameters.")


def _ordered_agent_tensors(
    tensor_dict: Mapping[str, torch.Tensor], agent_ids: Sequence[str], *, label: str
) -> list[torch.Tensor]:
    if not agent_ids:
        raise ValueError(f"Cannot collate {label} without agent ids.")

    missing = [agent_id for agent_id in agent_ids if agent_id not in tensor_dict]
    if missing:
        raise KeyError(f"Missing {label} for agents: {missing}.")

    tensors = [tensor_dict[agent_id] for agent_id in agent_ids]
    reference_shape = tuple(tensors[0].shape)
    for agent_id, tensor in zip(agent_ids[1:], tensors[1:], strict=True):
        if tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"Expected all {label} tensors to have shape {reference_shape}; "
                f"{agent_id} has shape {tuple(tensor.shape)}."
            )
    return tensors


def _validate_homogeneous_spaces(env: Any, agent_ids: Sequence[str]) -> None:
    reference_agent = agent_ids[0]
    reference_obs_shape = tuple(env.observation_spaces[reference_agent].shape)
    reference_action_shape = tuple(env.action_spaces[reference_agent].shape)
    for agent_id in agent_ids[1:]:
        obs_shape = tuple(env.observation_spaces[agent_id].shape)
        action_shape = tuple(env.action_spaces[agent_id].shape)
        if obs_shape != reference_obs_shape or action_shape != reference_action_shape:
            raise ValueError(
                "Shared homogeneous IPPO requires identical per-agent observation/action spaces; "
                f"{agent_id} has obs {obs_shape}, action {action_shape}, expected "
                f"obs {reference_obs_shape}, action {reference_action_shape}."
            )
