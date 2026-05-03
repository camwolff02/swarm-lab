"""Attention encoder matching the released Xie et al. architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .. import paper_spec as spec


@dataclass(frozen=True)
class FormationAttentionEncoderCfg:
    """Shape and model-size configuration for the formation encoder."""

    num_drones: int = spec.NUM_DRONES
    num_balls: int = spec.NUM_BALLS
    self_obs_dim: int = spec.SELF_OBS_DIM
    other_obs_dim: int = spec.OTHER_OBS_DIM
    dynamic_obs_dim: int = spec.DYNAMIC_OBS_DIM
    static_obs_dim: int = spec.STATIC_SDF_DIM
    attention_dim: int = spec.ATTENTION_DIM
    attention_heads: int = spec.ATTENTION_HEADS
    hidden_units: tuple[int, ...] = spec.MLP_HIDDEN
    initial_log_std: float = spec.INITIAL_LOG_STD

    @property
    def other_count(self) -> int:
        """Return the configured value."""
        return self.num_drones - 1

    @property
    def observation_dim(self) -> int:
        """Return the configured value."""
        return (
            self.self_obs_dim
            + self.other_count * self.other_obs_dim
            + self.num_balls * self.dynamic_obs_dim
            + self.static_obs_dim
        )

    @property
    def output_dim(self) -> int:
        """Return the configured value."""
        return self.hidden_units[-1]


class _SplitEmbedding(nn.Module):
    """_SplitEmbedding API surface."""
    def __init__(self, cfg: FormationAttentionEncoderCfg) -> None:
        """Initialize the _SplitEmbedding instance."""
        super().__init__()
        self.cfg = cfg
        self.self_embed = nn.Linear(cfg.self_obs_dim, cfg.attention_dim)
        self.other_embed = nn.Linear(cfg.other_obs_dim, cfg.attention_dim)
        self.ball_embed = nn.Linear(cfg.dynamic_obs_dim, cfg.attention_dim)
        self.static_embed = nn.Linear(cfg.static_obs_dim, cfg.attention_dim)
        self.norm = nn.LayerNorm(cfg.attention_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Run the module forward pass."""
        cfg = self.cfg
        self_end = cfg.self_obs_dim
        other_end = self_end + cfg.other_count * cfg.other_obs_dim
        ball_end = other_end + cfg.num_balls * cfg.dynamic_obs_dim
        obs_self = observations[:, :self_end].view(-1, 1, cfg.self_obs_dim)
        obs_others = observations[:, self_end:other_end].view(-1, cfg.other_count, cfg.other_obs_dim)
        obs_balls = observations[:, other_end:ball_end].view(-1, cfg.num_balls, cfg.dynamic_obs_dim)
        obs_static = observations[:, ball_end:].view(-1, 1, cfg.static_obs_dim)
        tokens = torch.cat(
            (
                self.self_embed(obs_self),
                self.other_embed(obs_others),
                self.ball_embed(obs_balls),
                self.static_embed(obs_static),
            ),
            dim=1,
        )
        return self.norm(tokens)


class FormationAttentionEncoder(nn.Module):
    """Self-attention plus self-query cross-attention encoder from the source implementation."""

    def __init__(self, cfg: FormationAttentionEncoderCfg = FormationAttentionEncoderCfg()) -> None:
        """Initialize the FormationAttentionEncoder instance."""
        super().__init__()
        self.cfg = cfg
        self.split_embed = _SplitEmbedding(cfg)
        self.self_attn = nn.MultiheadAttention(cfg.attention_dim, cfg.attention_heads, batch_first=True)
        self.self_norm1 = nn.LayerNorm(cfg.attention_dim)
        self.self_norm2 = nn.LayerNorm(cfg.attention_dim)
        self.self_linear1 = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.self_linear2 = nn.Linear(cfg.attention_dim, cfg.attention_dim)

        self.cross_attn = nn.MultiheadAttention(cfg.attention_dim, cfg.attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.attention_dim)
        self.norm2 = nn.LayerNorm(cfg.attention_dim)
        self.linear1 = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.linear2 = nn.Linear(cfg.attention_dim, cfg.attention_dim)

        dims = (2 * cfg.attention_dim, *cfg.hidden_units)
        layers: list[nn.Module] = [nn.LayerNorm(dims[0])]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ELU(), nn.LayerNorm(out_dim)))
        self.trunk = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Run the module forward pass."""
        if observations.shape[-1] != self.cfg.observation_dim:
            raise ValueError(f"Expected observation dim {self.cfg.observation_dim}, got {observations.shape[-1]}.")

        x = self.split_embed(observations)
        x2 = self.self_norm1(x)
        x = x + self.self_attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.self_linear2(F.gelu(self.self_linear1(self.self_norm2(x))))

        x2 = self.norm1(x)
        attended = x[:, [0]] + self.cross_attn(x2[:, [0]], x2[:, 1:], x2[:, 1:], need_weights=False)[0]
        x = torch.cat((x[:, :1], attended), dim=1)
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return self.trunk(x.reshape(x.shape[0], -1))
