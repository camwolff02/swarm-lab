from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from environments.tasks.quad_swarm_paper.agents.runner import install_quad_swarm_runner_patch
from environments.tasks.quad_swarm_paper.agents.shared_ippo import (
    assert_optimizer_owns_model_once,
    build_shared_ippo_components,
    flatten_agent_batch,
    flatten_rollout_tensor,
    stack_agent_observations,
    unflatten_agent_batch,
    unstack_agent_actions,
)
from skrl.utils.runner.torch import Runner


def _fake_env(num_agents: int = 3) -> SimpleNamespace:
    agent_ids = [f"drone_{index}" for index in range(num_agents)]
    return SimpleNamespace(
        device="cpu",
        possible_agents=agent_ids,
        observation_spaces={
            agent_id: gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(40,))
            for agent_id in agent_ids
        },
        action_spaces={agent_id: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent_id in agent_ids},
    )


def test_shared_observation_collation_preserves_possible_agent_order() -> None:
    agent_ids = ("drone_0", "drone_1", "drone_2")
    observations = {
        "drone_0": torch.full((2, 4), 0.0),
        "drone_1": torch.full((2, 4), 1.0),
        "drone_2": torch.full((2, 4), 2.0),
    }

    stacked = stack_agent_observations(observations, agent_ids)
    flattened = flatten_agent_batch(stacked)
    restored = unflatten_agent_batch(flattened, num_envs=2, agent_ids=agent_ids)

    assert stacked.shape == (2, 3, 4)
    assert torch.equal(stacked[:, 0], observations["drone_0"])
    assert torch.equal(stacked[:, 1], observations["drone_1"])
    assert torch.equal(stacked[:, 2], observations["drone_2"])
    assert flattened.shape == (6, 4)
    assert torch.equal(restored, stacked)


def test_shared_action_unstacking_preserves_env_action_api() -> None:
    agent_ids = ("drone_0", "drone_1", "drone_2")
    actions = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    action_dict = unstack_agent_actions(actions, agent_ids)

    assert tuple(action_dict) == agent_ids
    assert torch.equal(action_dict["drone_0"], actions[:, 0])
    assert torch.equal(action_dict["drone_1"], actions[:, 1])
    assert torch.equal(action_dict["drone_2"], actions[:, 2])


def test_shared_rollout_flattening_pools_time_env_and_agent_axes() -> None:
    rollout = torch.arange(5 * 2 * 3 * 4, dtype=torch.float32).reshape(5, 2, 3, 4)

    flattened = flatten_rollout_tensor(rollout)

    assert flattened.shape == (30, 4)
    assert torch.equal(flattened[0], rollout[0, 0, 0])
    assert torch.equal(flattened[1], rollout[0, 0, 1])
    assert torch.equal(flattened[3], rollout[0, 1, 0])


def test_shared_components_construct_one_model_and_one_optimizer_per_role() -> None:
    cfg = {
        "models": {
            "hidden_size": 32,
            "attention_heads": 4,
            "self_obs_dim": 19,
            "neighbor_obs_dim": 12,
            "obstacle_obs_dim": 9,
        },
        "agent": {"learning_rate": 1.0e-4},
    }

    components = build_shared_ippo_components(_fake_env(), cfg)

    assert components.agent_ids == ("drone_0", "drone_1", "drone_2")
    assert components.policy is not components.value
    assert components.policy_optimizer is not components.value_optimizer
    assert_optimizer_owns_model_once(components.policy, components.policy_optimizer)
    assert_optimizer_owns_model_once(components.value, components.value_optimizer)


def test_stock_runner_rejects_shared_homogeneous_mode_until_custom_trainer_exists() -> None:
    install_quad_swarm_runner_patch()
    cfg = {
        "models": {
            "factory": "quad_swarm_paper_attention",
            "share_parameters": True,
            "hidden_size": 32,
            "attention_heads": 4,
            "self_obs_dim": 19,
            "neighbor_obs_dim": 12,
            "obstacle_obs_dim": 9,
        },
        "training": {"shared_homogeneous_ippo": True},
    }
    runner = object.__new__(Runner)

    with pytest.raises(NotImplementedError, match="cannot use stock skrl IPPO"):
        runner._generate_models(_fake_env(), cfg)
