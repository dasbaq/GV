"""
CrossAttentionFusion — 3-way 고정 fusion.

LC + Param + Σ-curve token을 self-attention으로 융합.
Image 모달리티는 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    각 모달리티 feature를 token으로 취급해 Self-attention으로 융합 (3-way 고정).

    입력 : h_lc, h_params, h_sigma  각 [B, d_model]
    출력 : [B, d_model]  fused representation
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h_lc: torch.Tensor,
        h_params: torch.Tensor,
        h_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_lc, h_params, h_sigma : [B, d_model]

        Returns
        -------
        [B, d_model]
        """
        # stack → [B, 3, d_model]
        seq = torch.stack([h_lc, h_params, h_sigma], dim=1)

        # Self-attention (cross-modal)
        attn_out, _ = self.attn(seq, seq, seq)   # [B, 3, d_model]
        seq = self.norm1(seq + self.drop(attn_out))

        # Feed-forward
        seq = self.norm2(seq + self.drop(self.ff(seq)))

        # Mean pool over tokens → [B, d_model]
        return seq.mean(dim=1)
