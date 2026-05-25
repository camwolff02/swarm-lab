"""SKRL model wrappers for the paper_swarm attention policy."""

from __future__ import annotations

import gymnasium as gym
import torch
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch import nn

from .encoder import PaperAttentionEncoder, PaperAttentionEncoderCfg


def _space_size(space: gym.Space) -> int:
    """Space size."""
    size = 1
    for dim in space.shape:
        size *= int(dim)
    return size


def _states(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Policy observation — prefers 'observations' (86-dim policy) over 'states' (232-dim critic)."""
    return inputs.get("observations", inputs.get("states"))


def _critic_states(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Critic state — prefers 'states' (232-dim centralized) over 'observations' (86-dim policy)."""
    return inputs.get("states", inputs.get("observations"))


class PaperGaussianPolicy(GaussianMixin, Model):
    """Gaussian CTBR policy with the paper attention encoder."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        *,
        encoder_cfg: PaperAttentionEncoderCfg = PaperAttentionEncoderCfg(),
        clip_actions: bool = False,
    ) -> None:
        """Initialize the PaperGaussianPolicy instance."""
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions)
        self.encoder = PaperAttentionEncoder(encoder_cfg)
        self.mean_head = nn.Linear(encoder_cfg.output_dim, _space_size(action_space))
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        self.log_std_parameter = nn.Parameter(
            torch.full((_space_size(action_space),), float(encoder_cfg.initial_log_std))
        )

    def compute(
        self, inputs: dict[str, torch.Tensor], role: str = ""
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the policy distribution."""
        del role
        mean = self.mean_head(self.encoder(_states(inputs)))
        return mean, {"log_std": self.log_std_parameter.expand_as(mean)}


class PaperDeterministicValue(DeterministicMixin, Model):
    """Centralized value model for SKRL MAPPO shared states."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        *,
        hidden_units: tuple[int, ...] = (256, 256, 256),
    ) -> None:
        """Initialize the PaperDeterministicValue instance."""
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        layers: list[nn.Module] = []
        dims = (_space_size(observation_space), *hidden_units)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ELU(), nn.LayerNorm(out_dim)))
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the value for the current inputs."""
        del role
        return self.net(_critic_states(inputs)), {}
