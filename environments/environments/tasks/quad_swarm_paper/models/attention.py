"""Paper attention wrapper built on PyTorch scaled-dot-product attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PaperMultiheadAttention(nn.Module):
    """Release-shaped multi-head attention using PyTorch's SDPA kernel.

    The released implementation uses ``n_head=4`` and ``d_k=d_v=d_model``.
    ``nn.MultiheadAttention`` instead splits ``embed_dim`` across heads, so this
    wrapper keeps the release projection shapes while delegating scaled
    dot-product attention itself to PyTorch.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int | None = None) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim) if head_dim is not None else int(embed_dim)
        self.w_qs = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.w_ks = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.w_vs = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.fc = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1.0e-6)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size, sequence_length, _ = tokens.shape
        q = self.w_qs(tokens).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_ks(tokens).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_vs(tokens).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        attended = self.fc(attended)
        return self.layer_norm(tokens + attended), None
