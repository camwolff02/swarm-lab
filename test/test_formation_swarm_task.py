from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import torch

from environments.tasks.formation_swarm import paper_spec as spec
from environments.tasks.formation_swarm.agents.runner import _generate_models
from environments.tasks.formation_swarm.env import _formation_cost, _laplacian, _other_drone_observation, _pairwise_without_self
from environments.tasks.formation_swarm.models import FormationAttentionEncoder, FormationAttentionEncoderCfg


def test_formation_laplacian_cost_is_zero_for_target_formation():
    formation = torch.tensor(spec.FORMATION, dtype=torch.float32)
    desired = _laplacian(formation, normalize=True)

    cost = _formation_cost(formation.unsqueeze(0), desired, normalize=True)

    assert torch.allclose(cost, torch.zeros_like(cost))


def test_pairwise_without_self_shape():
    values = torch.zeros(2, spec.NUM_DRONES, spec.NUM_DRONES, 3)

    without_self = _pairwise_without_self(values)

    assert without_self.shape == (2, spec.NUM_DRONES, spec.NUM_DRONES - 1, 3)


def test_other_drone_observation_matches_paper_order():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    velocities = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.4, 0.0]]])

    obs = _other_drone_observation(positions, velocities).reshape(1, spec.NUM_DRONES, spec.NUM_DRONES - 1, spec.OTHER_OBS_DIM)

    assert torch.allclose(obs[0, 0, 0], torch.tensor([-1.0, 0.0, 0.0, 1.0, -0.2, 0.0, 0.0]))
    assert torch.allclose(obs[0, 0, 1], torch.tensor([0.0, -2.0, 0.0, 2.0, 0.0, -0.4, 0.0]))


def test_attention_encoder_shape():
    cfg = FormationAttentionEncoderCfg()
    encoder = FormationAttentionEncoder(cfg)

    latent = encoder(torch.zeros(4, cfg.observation_dim))

    assert latent.shape == (4, cfg.output_dim)


def test_model_factory_uses_shared_actor_and_observation_critic():
    obs_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(spec.OBS_DIM,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(spec.ACTION_DIM,))
    env = SimpleNamespace(
        device="cpu",
        possible_agents=[f"drone_{index}" for index in range(spec.NUM_DRONES)],
        observation_spaces={f"drone_{index}": obs_space for index in range(spec.NUM_DRONES)},
        action_spaces={f"drone_{index}": action_space for index in range(spec.NUM_DRONES)},
    )

    models = _generate_models(env, {"models": {"factory": "formation_swarm_attention"}})

    assert models["drone_0"]["policy"] is models["drone_1"]["policy"]
    assert models["drone_0"]["value"] is models["drone_1"]["value"]
    assert models["drone_0"]["value"].observation_space.shape == obs_space.shape


def test_obstacle_presence_broadcast_shape():
    batch = 64
    active_ball = torch.zeros(batch, 1, spec.NUM_BALLS, dtype=torch.bool)
    col_dist = torch.ones(batch, spec.NUM_DRONES, spec.STATIC_OBSTACLES)

    column_near = (col_dist < (spec.SOFT_OBS_SAFE_DISTANCE + 1.0)).any(dim=(1, 2)).unsqueeze(-1)
    obstacle_present = active_ball.any(dim=-1) | column_near

    assert obstacle_present.shape == (batch, 1)
