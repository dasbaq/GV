"""
4종 입력 인코더.

LightCurveEncoder   : [B, 2, T] + mask → [B, d_model]
ParamEncoder        : [B, P]            → [B, d_model]
SigmaCurveEncoder   : [B, 1, S]         → [B, d_model]
ImageEncoder        : [B, 1, H, W]      → [B, d_model]  (I_obs 단일채널, v0.6 복구)
"""

from __future__ import annotations

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
# ImageEncoder                                                                  #
# --------------------------------------------------------------------------- #
class _Conv2DStage(nn.Module):
    """stride-1 2×Conv2d + BN + GELU + MaxPool2d(2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """관측 이미지 인코더 (v0.6 복구, Mode 3 없음).

    입력  : [B, 1, H, W]  — I_obs 단일 채널 (H=W=image_size, 기본 64px)
    출력  : [B, d_model]

    단위 / 가정:
        - 픽셀값 정규화: 호출 전 [0, 1] 범위 가정 (dataset에서 처리).
        - SIE 표준 근사에서 얻은 관측 이미지. S_approx (truth-adjacent) 미사용.
        - Mode 3(Source 복원) head 없음 — Mode 1/2 correction 전용.

    아키텍처:
        4-stage 2D CNN (enc1~enc4):
          [B,1,64,64] → [B,32,32,32] → [B,64,16,16] → [B,128,8,8] → [B,d_model,4,4]
        AdaptiveAvgPool2d(1) → Flatten → Dropout → [B, d_model]
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.enc1 = _Conv2DStage(1, 32)
        self.enc2 = _Conv2DStage(32, 64)
        self.enc3 = _Conv2DStage(64, 128)
        self.enc4 = _Conv2DStage(128, d_model)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : [B, 1, H, W]  — I_obs, 픽셀값 [0, 1]

        Returns
        -------
        [B, d_model]
        """
        x = self.enc1(img)              # [B, 32, H/2, W/2]
        x = self.enc2(x)               # [B, 64, H/4, W/4]
        x = self.enc3(x)               # [B, 128, H/8, W/8]
        x = self.enc4(x)               # [B, d_model, H/16, W/16]
        x = self.pool(x).flatten(1)    # [B, d_model]
        return self.drop(x)
