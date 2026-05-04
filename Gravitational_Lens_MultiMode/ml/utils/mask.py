"""패딩 마스크 유틸리티."""

from __future__ import annotations
import torch


def make_lc_mask(n_valid: int, max_len: int) -> torch.BoolTensor:
    """
    True = valid position, False = padding.
    shape: [max_len]
    """
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[:n_valid] = True
    return mask


def masked_mean_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    feat  : [B, T, d_model] or [B, d_model, T]
    mask  : [B, T]  True=valid
    반환  : [B, d_model]
    """
    if feat.dim() == 3 and feat.shape[1] != mask.shape[1]:
        # [B, d_model, T] → [B, T, d_model]
        feat = feat.permute(0, 2, 1)
    mask_f = mask.float().unsqueeze(-1)          # [B, T, 1]
    summed = (feat * mask_f).sum(dim=1)          # [B, d_model]
    count  = mask_f.sum(dim=1).clamp(min=1.0)   # [B, 1]
    return summed / count
