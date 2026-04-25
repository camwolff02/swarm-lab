from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

from environments.tasks.quad_swarm_paper.agents.runner import install_quad_swarm_runner_patch
from environments.tasks.quad_swarm_paper.models.quad_swarm_encoder import QuadSwarmEncoder, QuadSwarmEncoderCfg
from environments.tasks.quad_swarm_paper.models.quad_swarm_skrl_models import (
    QuadSwarmDeterministicValue,
    QuadSwarmGaussianPolicy,
)


def test_quad_swarm_encoder_slices_observation_and_returns_stable_latent_size() -> None:
    cfg = QuadSwarmEncoderCfg(hidden_size=32, attention_heads=4)
    encoder = QuadSwarmEncoder(cfg)

    latent = encoder(torch.zeros(5, cfg.observation_dim))

    assert latent.shape == (5, cfg.output_dim)


def test_quad_swarm_skrl_models_forward_shapes() -> None:
    encoder_cfg = QuadSwarmEncoderCfg(hidden_size=32, attention_heads=4)
    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(40,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    observations = torch.zeros(7, 40)

    policy = QuadSwarmGaussianPolicy(observation_space, action_space, "cpu", encoder_cfg=encoder_cfg)
    value = QuadSwarmDeterministicValue(observation_space, action_space, "cpu", encoder_cfg=encoder_cfg)

    means, policy_outputs = policy.compute({"observations": observations})
    values, value_outputs = value.compute({"observations": observations})

    assert means.shape == (7, 4)
    assert policy_outputs["log_std"].shape == (7, 4)
    assert values.shape == (7, 1)
    assert value_outputs == {}


def test_stock_skrl_runner_hook_reuses_shared_module_instances() -> None:
    install_quad_swarm_runner_patch()
    env = SimpleNamespace(
        device="cpu",
        possible_agents=["drone_0", "drone_1"],
        observation_spaces={
            "drone_0": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(40,)),
            "drone_1": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(40,)),
        },
        action_spaces={
            "drone_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
            "drone_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
        },
    )
    cfg = {
        "models": {
            "factory": "quad_swarm_paper_attention",
            "share_parameters": True,
            "hidden_size": 32,
            "attention_heads": 4,
            "self_obs_dim": 19,
            "neighbor_obs_dim": 12,
            "obstacle_obs_dim": 9,
        }
    }
    runner = object.__new__(Runner)

    models = runner._generate_models(env, cfg)

    assert models["drone_0"] is not models["drone_1"]
    assert models["drone_0"]["policy"] is models["drone_1"]["policy"]
    assert models["drone_0"]["value"] is models["drone_1"]["value"]
