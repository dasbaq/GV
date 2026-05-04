"""
Mode별 분기 헤드.

Mode1Head : MLP → (h0_correction [B], log_sigma [B])
Mode2Head : MLP → (dm_correction [B, max_dm_dim], log_sigma [B, max_dm_dim])
Mode3Head : U-Net 디코더 → residual map [B, 1, H, W]
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_SIGMA_MIN = -5.0
_LOG_SIGMA_MAX =  2.0


def _clamp_log_sigma(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)


# --------------------------------------------------------------------------- #
# Mode 1 Head                                                                   #
# --------------------------------------------------------------------------- #
class Mode1Head(nn.Module):
    """
    MLP → (H0_correction: scalar, log_σ: scalar) per sample.
    출력 shapes: ([B], [B])
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1,
                 in_dim: int | None = None) -> None:
        super().__init__()
        in_dim = in_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.out_val = nn.Linear(32, 1)
        self.out_log_sigma = nn.Linear(32, 1)

    def forward(self, fused: torch.Tensor) -> dict:
        h = self.net(fused)
        val       = self.out_val(h).squeeze(-1)           # [B]
        log_sigma = _clamp_log_sigma(
            self.out_log_sigma(h).squeeze(-1)
        )                                                  # [B]
        return {"h0_correction": val, "log_sigma": log_sigma}


# --------------------------------------------------------------------------- #
# Mode 2 Head                                                                   #
# --------------------------------------------------------------------------- #
class Mode2Head(nn.Module):
    """
    MLP → (dm_correction [B, max_dm_dim], log_σ [B, max_dm_dim])
    """

    def __init__(self, d_model: int = 128, max_dm_dim: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.max_dm_dim = max_dm_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.out_val      = nn.Linear(64, max_dm_dim)
        self.out_log_sigma= nn.Linear(64, max_dm_dim)

    def forward(self, fused: torch.Tensor) -> dict:
        h = self.net(fused)
        val       = self.out_val(h)                          # [B, max_dm_dim]
        log_sigma = _clamp_log_sigma(self.out_log_sigma(h))  # [B, max_dm_dim]
        return {"dm_correction": val, "log_sigma": log_sigma}


# --------------------------------------------------------------------------- #
# Mode 3 Head — U-Net style decoder                                             #
# --------------------------------------------------------------------------- #
class _UpBlock(nn.Module):
    """업샘플링 + skip connection 융합 블록."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 크기 보정 (홀수 해상도 처리)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Mode3Head(nn.Module):
    """
    U-Net 디코더. fused feature [B, d_model] + image skip features → residual [B, 1, H, W].

    skip_features 순서(ImageEncoder 출력 순): [s1, s2, s3, s4]
      s1: [B, 32,      H,   W]
      s2: [B, 64,      H/2, W/2]
      s3: [B, 128,     H/4, W/4]
      s4: [B, d_model, H/8, W/8]
    """

    def __init__(self, d_model: int = 128, image_size: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.image_size = image_size
        h8 = image_size // 8

        # fused 벡터 → 공간 텐서 [B, d_model, H/8, W/8]
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model * h8 * h8),
            nn.GELU(),
        )
        self.h8 = h8
        self.d_model = d_model

        # 디코더 레벨 (skip_ch는 ImageEncoder와 맞춤)
        self.up3 = _UpBlock(d_model, 128,     128)   # → H/4
        self.up2 = _UpBlock(128,     64,      64)    # → H/2
        self.up1 = _UpBlock(64,      32,      32)    # → H
        self.final = nn.Conv2d(32, 1, 1)

        self.drop = nn.Dropout2d(dropout)

    def forward(
        self,
        fused: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> dict:
        """
        Parameters
        ----------
        fused        : [B, d_model]
        skip_features: [s1, s2, s3, s4]  from ImageEncoder

        Returns
        -------
        dict with key "source_residual" : [B, 1, H, W]
        """
        B = fused.size(0)
        s1, s2, s3, s4 = skip_features

        # project → 공간 텐서
        x = self.project(fused)                          # [B, d_model*h8*h8]
        x = x.view(B, self.d_model, self.h8, self.h8)   # [B, d_model, H/8, W/8]

        # skip_features 없는 경우(Mode 1·2에서 dummy image 쓸 때) 처리
        # → s4와 채널 수 맞추기
        if x.shape[1] != s4.shape[1]:
            x = F.interpolate(x, size=s4.shape[2:], mode="bilinear",
                              align_corners=False)

        x = self.drop(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        out = self.final(x)                              # [B, 1, H, W]
        return {"source_residual": out}
