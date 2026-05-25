"""
Mode별 분기 헤드.

Mode1Head : MLP → (h0_correction [B], log_sigma [B])
Mode2Head : MLP → (dm_correction [B, max_dm_dim], log_sigma [B, max_dm_dim])
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
