"""skrl model wrappers for the quad swarm paper encoder."""

from __future__ import annotations

import gymnasium as gym
import torch
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch import nn

from .quad_swarm_encoder import QuadSwarmEncoder, QuadSwarmEncoderCfg


def _space_size(space: gym.Space) -> int:
    if not hasattr(space, "shape") or space.shape is None:
        raise ValueError(f"Expected a Box-like space with a shape, got {space}.")
    size = 1
    for dim in space.shape:
        size *= int(dim)
    return size


class QuadSwarmGaussianPolicy(GaussianMixin, Model):
    """Gaussian policy with a paper-style attention encoder."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        *,
        encoder_cfg: QuadSwarmEncoderCfg = QuadSwarmEncoderCfg(),
        clip_actions: bool = True,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions)
        self.encoder = QuadSwarmEncoder(encoder_cfg)
        self.mean_head = nn.Linear(encoder_cfg.output_dim, _space_size(action_space))
        self.log_std_parameter = nn.Parameter(torch.zeros(_space_size(action_space)))

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del role
        latent = self.encoder(inputs["observations"])
        mean = self.mean_head(latent)
        return mean, {"log_std": self.log_std_parameter.expand_as(mean)}


class QuadSwarmDeterministicValue(DeterministicMixin, Model):
    """Local value function with a paper-style attention encoder."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        *,
        encoder_cfg: QuadSwarmEncoderCfg = QuadSwarmEncoderCfg(),
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.encoder = QuadSwarmEncoder(encoder_cfg)
        self.value_head = nn.Linear(encoder_cfg.output_dim, 1)

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del role
        return self.value_head(self.encoder(inputs["observations"])), {}
