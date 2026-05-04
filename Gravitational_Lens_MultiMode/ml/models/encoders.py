"""
4종 입력 인코더.

LightCurveEncoder   : [B, 2, T] + mask → [B, d_model]
ParamEncoder        : [B, P]            → [B, d_model]
SigmaCurveEncoder   : [B, 1, S]         → [B, d_model]
ImageEncoder        : [B, 1, H, W]      → ([B, d_model], skip_features)
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn

from ml.utils.mask import masked_mean_pool


# --------------------------------------------------------------------------- #
# 공통 1D CNN 블록                                                              #
# --------------------------------------------------------------------------- #
class _Conv1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# LightCurveEncoder                                                             #
# --------------------------------------------------------------------------- #
class LightCurveEncoder(nn.Module):
    """
    입력 : [B, 2, T] (channel 0=flux, channel 1=noise), mask [B, T]
    출력 : [B, d_model]
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _Conv1DBlock(2, 32, 7),
            _Conv1DBlock(32, 64, 5),
            _Conv1DBlock(64, 64, 3, dilation=2),
            _Conv1DBlock(64, d_model, 3),
        )
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, lc: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(lc)                    # [B, d_model, T]
        x = x.permute(0, 2, 1)              # [B, T, d_model]
        x = masked_mean_pool(x, mask)        # [B, d_model]
        return self.drop(x)


# --------------------------------------------------------------------------- #
# ParamEncoder                                                                  #
# --------------------------------------------------------------------------- #
class ParamEncoder(nn.Module):
    """
    입력 : [B, P]
    출력 : [B, d_model]
    """

    def __init__(self, in_dim: int, d_model: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)


# --------------------------------------------------------------------------- #
# SigmaCurveEncoder                                                             #
# --------------------------------------------------------------------------- #
class SigmaCurveEncoder(nn.Module):
    """
    입력 : [B, 1, S]
    출력 : [B, d_model]
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _Conv1DBlock(1, 32, 9),
            _Conv1DBlock(32, 64, 5),
            _Conv1DBlock(64, d_model, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        x = self.conv(sigma)           # [B, d_model, S]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        return self.drop(x)


# --------------------------------------------------------------------------- #
# ImageEncoder (2D CNN with skip connections for U-Net decoder)                #
# --------------------------------------------------------------------------- #
class _Conv2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """
    입력 : [B, 2, H, W]
    출력 : (global_feat [B, d_model], skip_features List[Tensor])
           skip_features는 Mode3Head U-Net 디코더로 전달.
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        # 인코더 레벨: 2→32→64→128→d_model
        self.enc1 = _Conv2DBlock(2, 32)
        self.enc2 = _Conv2DBlock(32, 64)
        self.enc3 = _Conv2DBlock(64, 128)
        self.enc4 = _Conv2DBlock(128, d_model)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s1 = self.enc1(img)                    # [B, 32, H, W]
        s2 = self.enc2(self.pool(s1))          # [B, 64, H/2, W/2]
        s3 = self.enc3(self.pool(s2))          # [B, 128, H/4, W/4]
        s4 = self.enc4(self.pool(s3))          # [B, d_model, H/8, W/8]

        global_feat = self.global_pool(s4).flatten(1)  # [B, d_model]
        global_feat = self.drop(global_feat)

        return global_feat, [s1, s2, s3, s4]
