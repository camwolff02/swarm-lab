"""Attention encoder for the paper_swarm waypoint-navigation task.

Architecture matches the formation_swarm encoder (self-attention + self-query
cross-attention) using PyTorch's builtin nn.MultiheadAttention.

Token layout:
    [self] [neighbor_0 ... neighbor_K] [sdf] [goal]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PaperAttentionEncoderCfg:
    """Shape and model-size configuration for the paper_swarm attention encoder.

    The observation layout expected by the encoder is:

        [self: self_obs_dim] [neighbors: max_neighbors * other_obs_dim]
        [sdf: static_sdf_dim] [goal: goal_obs_dim]

    where the *self* block ends with the active-flag and last-action aux features.
    """

    num_drones: int = 8
    max_neighbors: int = 7
    self_obs_dim: int = 34
    other_obs_dim: int = 6
    static_sdf_dim: int = 9
    goal_obs_dim: int = 6
    attention_dim: int = 32
    attention_heads: int = 1
    hidden_units: tuple[int, ...] = (256, 256, 256)
    initial_log_std: float = 0.0

    @property
    def other_count(self) -> int:
        """Return the configured value."""
        return self.max_neighbors

    @property
    def observation_dim(self) -> int:
        """Return the configured value."""
        return (
            self.self_obs_dim
            + self.max_neighbors * self.other_obs_dim
            + self.static_sdf_dim
            + self.goal_obs_dim
        )

    @property
    def output_dim(self) -> int:
        """Return the configured value."""
        return self.hidden_units[-1]


class _SplitEmbedding(nn.Module):
    """Split flat observation into four semantically distinct token groups."""

    def __init__(self, cfg: PaperAttentionEncoderCfg) -> None:
        """Initialize the _SplitEmbedding instance."""
        super().__init__()
        self.cfg = cfg
        self.self_embed = nn.Linear(cfg.self_obs_dim, cfg.attention_dim)
        self.other_embed = nn.Linear(cfg.other_obs_dim, cfg.attention_dim)
        self.sdf_embed = nn.Linear(cfg.static_sdf_dim, cfg.attention_dim)
        self.goal_embed = nn.Linear(cfg.goal_obs_dim, cfg.attention_dim)
        self.norm = nn.LayerNorm(cfg.attention_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Convert flat observation to token sequence (B, num_tokens, attention_dim)."""
        cfg = self.cfg
        self_end = cfg.self_obs_dim
        other_end = self_end + cfg.max_neighbors * cfg.other_obs_dim
        sdf_end = other_end + cfg.static_sdf_dim

        obs_self = observations[:, :self_end].view(-1, 1, cfg.self_obs_dim)
        obs_others = observations[:, self_end:other_end].view(-1, cfg.max_neighbors, cfg.other_obs_dim)
        obs_sdf = observations[:, other_end:sdf_end].view(-1, 1, cfg.static_sdf_dim)
        obs_goal = observations[:, sdf_end:].view(-1, 1, cfg.goal_obs_dim)

        tokens = torch.cat(
            (
                self.self_embed(obs_self),
                self.other_embed(obs_others),
                self.sdf_embed(obs_sdf),
                self.goal_embed(obs_goal),
            ),
            dim=1,
        )
        return self.norm(tokens)


class PaperAttentionEncoder(nn.Module):
    """Self-attention + self-query cross-attention encoder using builtin PyTorch attention."""

    def __init__(self, cfg: PaperAttentionEncoderCfg = PaperAttentionEncoderCfg()) -> None:
        """Initialize the PaperAttentionEncoder instance."""
        super().__init__()
        self.cfg = cfg
        self.split_embed = _SplitEmbedding(cfg)

        # Self-attention block
        self.self_attn = nn.MultiheadAttention(cfg.attention_dim, cfg.attention_heads, batch_first=True)
        self.self_norm1 = nn.LayerNorm(cfg.attention_dim)
        self.self_norm2 = nn.LayerNorm(cfg.attention_dim)
        self.self_linear1 = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.self_linear2 = nn.Linear(cfg.attention_dim, cfg.attention_dim)

        # Cross-attention block (query=self-token, key/value=other tokens)
        self.cross_attn = nn.MultiheadAttention(cfg.attention_dim, cfg.attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.attention_dim)
        self.norm2 = nn.LayerNorm(cfg.attention_dim)
        self.linear1 = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.linear2 = nn.Linear(cfg.attention_dim, cfg.attention_dim)

        # Output trunk: concat attended self-token + original self → MLP
        dims = (cfg.attention_dim * 2, *cfg.hidden_units)
        layers: list[nn.Module] = [nn.LayerNorm(dims[0])]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ELU(), nn.LayerNorm(out_dim)))
        self.trunk = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Process observation through attention encoder.

        Args:
            observations: Flat observation tensor (B, observation_dim).

        Returns:
            Encoded feature vector (B, output_dim).
        """
        if observations.shape[-1] != self.cfg.observation_dim:
            raise ValueError(
                f"Expected observation dim {self.cfg.observation_dim}, got {observations.shape[-1]}."
            )

        # Tokenize
        x = self.split_embed(observations)  # (B, num_tokens, attention_dim)

        # Self-attention block (Pre-LN residual)
        x2 = self.self_norm1(x)
        x = x + self.self_attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.self_linear2(F.gelu(self.self_linear1(self.self_norm2(x))))

        # Self-query cross-attention block
        x2 = self.norm1(x)
        attended = x[:, [0]] + self.cross_attn(x2[:, [0]], x2[:, 1:], x2[:, 1:], need_weights=False)[0]
        x = torch.cat((x[:, :1], attended), dim=1)
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))

        # Flatten all tokens and pass through trunk
        return self.trunk(x.reshape(x.shape[0], -1))
