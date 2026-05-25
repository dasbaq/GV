"""
MultiModalErrorCorrector — 조립체.

공유 인코더 + CrossAttentionFusion + Mode별 분기 헤드.
target_mode에 따라 해당 head만 호출.

입력 모달리티: LC + Param + Σ-curve (3-way 고정).
Mode 3(Source 복원)과 Image 입력은 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.encoders import (
    LightCurveEncoder, ParamEncoder, SigmaCurveEncoder,
)
from ml.models.fusion import CrossAttentionFusion
from ml.models.heads import Mode1Head, Mode2Head


class MultiModalErrorCorrector(nn.Module):
    """
    Parameters (cfg keys)
    ---------------------
    d_model           : int   (default 128)
    n_heads           : int   (default 4)
    dropout           : float (default 0.1)
    mode2_max_dm_dim  : int   (default 4)
    param_in_dim      : int   — norm_cfg 키 수 + 5 (approx + mode one-hot)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_model  = cfg.get("d_model", 128)
        n_heads  = cfg.get("n_heads", 4)
        dropout  = cfg.get("dropout", 0.1)
        max_dm   = cfg.get("mode2_max_dm_dim", 4)
        param_dim= cfg.get("param_in_dim", 20)  # scalar schema + 2 approx + 3 mode

        self.lc_enc    = LightCurveEncoder(d_model, dropout)
        self.par_enc   = ParamEncoder(param_dim, d_model, dropout)
        self.sig_enc   = SigmaCurveEncoder(d_model, dropout)

        self.fusion    = CrossAttentionFusion(d_model, n_heads, dropout)

        self.head1     = Mode1Head(d_model, dropout, in_dim=d_model * 2)
        self.head2     = Mode2Head(d_model, max_dm, dropout)

    def forward(
        self,
        lc:          torch.Tensor,          # [B, 2, T]
        lc_mask:     torch.Tensor,          # [B, T]
        params:      torch.Tensor,          # [B, P]
        sigma_curve: torch.Tensor,          # [B, 1, S]
        target_mode: torch.Tensor,          # [B]  int 1/2
    ) -> dict:
        """
        target_mode에 따라 해당 head만 호출.
        다른 mode는 None 반환.

        Returns
        -------
        {
            "mode1": dict or None,
            "mode2": dict or None,
            "fused": Tensor [B, d_model]
        }
        """
        modes_in_batch = target_mode.unique().tolist()

        # 공유 인코더 (항상 실행)
        h_lc  = self.lc_enc(lc, lc_mask)       # [B, d_model]
        h_par = self.par_enc(params)             # [B, d_model]
        h_sig = self.sig_enc(sigma_curve)        # [B, d_model]

        # Fusion — 3-way 고정
        fused = self.fusion(h_lc, h_par, h_sig)  # [B, d_model]

        out: dict = {"mode1": None, "mode2": None, "fused": fused}

        # Mode별 head 호출 (sub-batch 분리)
        for mode_id in modes_in_batch:
            mode_id = int(mode_id)
            mask = (target_mode == mode_id)            # [B]
            fused_sub = fused[mask]                    # [n_sub, d_model]

            if mode_id == 1:
                head1_in = torch.cat([fused_sub, h_lc[mask]], dim=1)
                out["mode1"] = self.head1(head1_in)
            elif mode_id == 2:
                out["mode2"] = self.head2(fused_sub)

        return out
