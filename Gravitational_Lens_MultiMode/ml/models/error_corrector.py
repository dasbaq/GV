"""
MultiModalErrorCorrector — 조립체.

공유 인코더 + CrossAttentionFusion + Mode별 분기 헤드.
target_mode에 따라 해당 head만 호출.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from ml.models.encoders import (
    LightCurveEncoder, ParamEncoder, SigmaCurveEncoder, ImageEncoder
)
from ml.models.fusion import CrossAttentionFusion
from ml.models.heads import Mode1Head, Mode2Head, Mode3Head


class MultiModalErrorCorrector(nn.Module):
    """
    Parameters (cfg keys)
    ---------------------
    d_model           : int   (default 128)
    n_heads           : int   (default 4)
    dropout           : float (default 0.1)
    mode2_max_dm_dim  : int   (default 4)
    image_size        : int   (default 128)
    param_in_dim      : int   — norm_cfg 키 수 + 5 (approx + mode one-hot)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_model  = cfg.get("d_model", 128)
        n_heads  = cfg.get("n_heads", 4)
        dropout  = cfg.get("dropout", 0.1)
        max_dm   = cfg.get("mode2_max_dm_dim", 4)
        img_size = cfg.get("image_size", 128)
        param_dim= cfg.get("param_in_dim", 12)  # 7 물리/ratio + 2 approx + 3 mode

        self.lc_enc    = LightCurveEncoder(d_model, dropout)
        self.par_enc   = ParamEncoder(param_dim, d_model, dropout)
        self.sig_enc   = SigmaCurveEncoder(d_model, dropout)
        self.img_enc   = ImageEncoder(d_model, dropout)

        self.fusion    = CrossAttentionFusion(d_model, n_heads, dropout)

        self.head1     = Mode1Head(d_model, dropout, in_dim=d_model * 3)
        self.head2     = Mode2Head(d_model, max_dm, dropout)
        self.head3     = Mode3Head(d_model, img_size, dropout)

    def forward(
        self,
        lc:          torch.Tensor,          # [B, 2, T]
        lc_mask:     torch.Tensor,          # [B, T]
        params:      torch.Tensor,          # [B, P]
        sigma_curve: torch.Tensor,          # [B, 1, S]
        image:       torch.Tensor,          # [B, 1, H, W]
        use_image:   torch.Tensor,          # [B]  bool or float
        target_mode: torch.Tensor,          # [B]  int 1/2/3
    ) -> dict:
        """
        target_mode에 따라 해당 head만 호출.
        다른 mode는 None 반환.

        Returns
        -------
        {
            "mode1": dict or None,
            "mode2": dict or None,
            "mode3": dict or None,
            "fused": Tensor [B, d_model]
        }
        """
        B = lc.size(0)
        modes_in_batch = target_mode.unique().tolist()

        # 공유 인코더 (항상 실행)
        h_lc  = self.lc_enc(lc, lc_mask)       # [B, d_model]
        h_par = self.par_enc(params)             # [B, d_model]
        h_sig = self.sig_enc(sigma_curve)        # [B, d_model]

        # Image 인코더 — Mode 3이 포함된 경우만 실제 실행, 나머지는 zeros skip
        any_image = bool(use_image.any().item())
        if any_image:
            h_img, skip_feats = self.img_enc(image)
        else:
            h_img = torch.zeros(B, self.img_enc.d_model, device=lc.device)
            # 더미 skip features (Mode3Head에서 안 쓰이도록 head에서 use_image 체크)
            H8 = self.head3.h8
            skip_feats = [
                torch.zeros(B, 32,  h_lc.shape[0] and self.head3.image_size,
                            self.head3.image_size, device=lc.device),
                torch.zeros(B, 64,  self.head3.image_size // 2,
                            self.head3.image_size // 2, device=lc.device),
                torch.zeros(B, 128, self.head3.image_size // 4,
                            self.head3.image_size // 4, device=lc.device),
                torch.zeros(B, self.img_enc.d_model, H8, H8, device=lc.device),
            ]

        # Fusion — use_image는 batch 내 any_image 기준
        fused = self.fusion(h_lc, h_par, h_sig,
                            h_img=h_img, use_image=any_image)  # [B, d_model]

        out: dict = {"mode1": None, "mode2": None, "mode3": None, "fused": fused}

        # Mode별 head 호출 (sub-batch 분리)
        for mode_id in modes_in_batch:
            mode_id = int(mode_id)
            mask = (target_mode == mode_id)            # [B]
            fused_sub = fused[mask]                    # [n_sub, d_model]

            if mode_id == 1:
                head1_in = torch.cat([fused_sub, h_lc[mask], h_img[mask]], dim=1)
                out["mode1"] = self.head1(head1_in)
            elif mode_id == 2:
                out["mode2"] = self.head2(fused_sub)
            elif mode_id == 3:
                # skip_features를 sub-batch로 슬라이싱
                sub_skips = [s[mask] for s in skip_feats]
                out["mode3"] = self.head3(fused_sub, sub_skips)

        return out
