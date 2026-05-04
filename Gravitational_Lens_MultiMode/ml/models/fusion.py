"""
CrossAttentionFusion — 가변 모달리티 fusion.

Mode 1·2: LC + Param + Σ  (3-way)
Mode 3:   LC + Param + Σ + Image (4-way, use_image=True)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    각 모달리티 feature를 token으로 취급해 Cross-attention으로 융합.

    입력 : h_lc, h_params, h_sigma  각 [B, d_model]
           h_img                    [B, d_model]  (옵션)
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
        h_img: Optional[torch.Tensor] = None,
        use_image: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_lc, h_params, h_sigma : [B, d_model]
        h_img                   : [B, d_model]  (use_image=True일 때만 사용)
        use_image               : bool

        Returns
        -------
        [B, d_model]
        """
        tokens = [h_lc, h_params, h_sigma]
        if use_image and h_img is not None:
            tokens.append(h_img)

        # stack → [B, n_tokens, d_model]
        seq = torch.stack(tokens, dim=1)

        # Self-attention (cross-modal)
        attn_out, _ = self.attn(seq, seq, seq)   # [B, n_tokens, d_model]
        seq = self.norm1(seq + self.drop(attn_out))

        # Feed-forward
        seq = self.norm2(seq + self.drop(self.ff(seq)))

        # Mean pool over tokens → [B, d_model]
        return seq.mean(dim=1)
