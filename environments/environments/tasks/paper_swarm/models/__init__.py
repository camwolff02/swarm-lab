"""Model package for the paper_swarm task."""

from __future__ import annotations

from .encoder import PaperAttentionEncoder, PaperAttentionEncoderCfg
from .skrl_models import PaperDeterministicValue, PaperGaussianPolicy

__all__ = [
    "PaperAttentionEncoder",
    "PaperAttentionEncoderCfg",
    "PaperDeterministicValue",
    "PaperGaussianPolicy",
]
