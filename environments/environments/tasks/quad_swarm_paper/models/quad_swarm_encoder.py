"""Paper-style encoder for quad swarm observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import PaperMultiheadAttention


@dataclass(frozen=True)
class QuadSwarmEncoderCfg:
    """Shape and hidden-size configuration for the paper encoder."""

    self_obs_dim: int = 19
    neighbor_obs_dim: int = 12
    obstacle_obs_dim: int = 9
    hidden_size: int = 256
    attention_heads: int = 4
    initial_log_std: float = -1.0
    init_policy_to_hover: bool = True

    @property
    def observation_dim(self) -> int:
        return self.self_obs_dim + self.neighbor_obs_dim + self.obstacle_obs_dim

    @property
    def output_dim(self) -> int:
        return 2 * self.hidden_size


def _two_layer_mlp(input_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
    )


class QuadSwarmEncoder(nn.Module):
    """Encode ``[self, neighbor, obstacle]`` observations with the paper attention layout."""

    def __init__(self, cfg: QuadSwarmEncoderCfg = QuadSwarmEncoderCfg()) -> None:
        super().__init__()
        self.cfg = cfg
        self.self_embed_layer = _two_layer_mlp(cfg.self_obs_dim, cfg.hidden_size)
        self.neighbor_embed_layer = _two_layer_mlp(cfg.neighbor_obs_dim, cfg.hidden_size)
        self.obstacle_embed_layer = _two_layer_mlp(cfg.obstacle_obs_dim, cfg.hidden_size)
        self.attention_layer = PaperMultiheadAttention(cfg.hidden_size, cfg.attention_heads)
        self.feed_forward = nn.Sequential(nn.Linear(3 * cfg.hidden_size, cfg.output_dim), nn.Tanh())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.cfg.observation_dim:
            raise ValueError(
                f"Expected observation dim {self.cfg.observation_dim}, got {observations.shape[-1]}."
            )

        self_end = self.cfg.self_obs_dim
        neighbor_end = self_end + self.cfg.neighbor_obs_dim
        obs_self = observations[..., :self_end]
        obs_neighbor = observations[..., self_end:neighbor_end]
        obs_obstacle = observations[..., neighbor_end:]

        self_embed = self.self_embed_layer(obs_self)
        neighbor_embed = self.neighbor_embed_layer(obs_neighbor).view(observations.shape[0], 1, -1)
        obstacle_embed = self.obstacle_embed_layer(obs_obstacle).view(observations.shape[0], 1, -1)

        attn_embed = torch.cat((neighbor_embed, obstacle_embed), dim=1)
        attn_embed, _ = self.attention_layer(attn_embed)
        embeddings = torch.cat((self_embed, attn_embed.reshape(observations.shape[0], -1)), dim=-1)
        return self.feed_forward(embeddings)
