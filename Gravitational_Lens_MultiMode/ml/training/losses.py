"""
composite_loss — Mode별 task loss + physics + calibration.

Mode 1: MSE(H0) + Gaussian NLL
Mode 2: 마스킹된 MSE(dm_params) + Gaussian NLL
Mode 3: 픽셀 MSE + (1 - SSIM)  [torch 구현, skimage 미사용]
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# SSIM (torch 기반, skimage 미사용)                                             #
# --------------------------------------------------------------------------- #
def _ssim_loss(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 11, C1: float = 0.01**2,
               C2: float = 0.03**2) -> torch.Tensor:
    """
    1 - SSIM(pred, target).  입력: [B, 1, H, W].
    반환: 스칼라

    AMP/fp16 안전: autocast를 비활성화하고 내부 계산을 float32로 강제한다.
    autocast 하에서는 conv2d가 입력 dtype과 무관하게 fp16으로 다운캐스트되어
    분모 eps(1e-8)가 underflow하고, conv 누적오차로 분산이 음수가 되면 ratio가
    inf/nan이 된다(v0.3/v0.3.1 학습 NaN의 발원지). 분산은 0으로 클램프한다.
    """
    device_type = "cuda" if pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        pred = pred.float()
        target = target.float()
        # 가우시안 커널
        sigma = 1.5
        coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel = g[:, None] * g[None, :]                     # [w, w]
        kernel = kernel.unsqueeze(0).unsqueeze(0)            # [1, 1, w, w]
        pad = window_size // 2

        mu_x  = F.conv2d(pred,   kernel, padding=pad)
        mu_y  = F.conv2d(target, kernel, padding=pad)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        # 분산/공분산은 수치오차로 음수가 될 수 있으므로 클램프
        sig_x  = (F.conv2d(pred   * pred,   kernel, padding=pad) - mu_x2).clamp_min(0.0)
        sig_y  = (F.conv2d(target * target, kernel, padding=pad) - mu_y2).clamp_min(0.0)
        sig_xy = F.conv2d(pred   * target, kernel, padding=pad) - mu_xy

        ssim_map = ((2*mu_xy + C1) * (2*sig_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


# --------------------------------------------------------------------------- #
# Gaussian NLL                                                                  #
# --------------------------------------------------------------------------- #
def _gaussian_nll(pred: torch.Tensor, target: torch.Tensor,
                  log_sigma: torch.Tensor) -> torch.Tensor:
    """
    NLL = 0.5 * [log(2π) + 2*log_sigma + ((pred-target)/exp(log_sigma))^2]
    pred, target, log_sigma: 동일 shape.
    반환: 스칼라

    AMP/fp16 안전: autocast를 비활성화하고 float32로 계산한다. autocast 하에서는
    exp(2*log_sigma)가 fp16 overflow하거나 var가 underflow해 (pred-target)^2/var가
    inf가 될 수 있다. 2*log_sigma를 안전범위로 클램프하고 var에 fp32 floor를 둔다.
    """
    device_type = "cuda" if pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        pred = pred.float()
        target = target.float()
        two_log_sigma = (2.0 * log_sigma.float()).clamp(-30.0, 30.0)
        var = two_log_sigma.exp().clamp_min(1e-8)
        return (0.5 * ((pred - target) ** 2 / var + two_log_sigma)).mean()


# --------------------------------------------------------------------------- #
# composite_loss                                                                #
# --------------------------------------------------------------------------- #
def composite_loss(
    pred: dict,
    batch: dict,
    weights: dict,
) -> dict:
    """
    Parameters
    ----------
    pred    : MultiModalErrorCorrector.forward() 출력
    batch   : DataLoader 배치 dict
    weights : {mode1, mode2, mode3, physics, calibration, ssim}

    Returns
    -------
    dict: {total, mode1_task, mode2_task, mode3_task,
           mode1_cal, mode2_cal, physics, ssim}
    """
    device = batch["target"].device
    zero   = torch.tensor(0.0, device=device)

    target_mode = batch["target_mode"]              # [B]
    target      = batch["target"]                   # [B, max_label_dim]
    target_dim  = batch["target_dim"]               # [B]
    target_img  = batch["target_image"]             # [B, H, W]

    w_m1  = weights.get("mode1", 1.0)
    w_m2  = weights.get("mode2", 1.0)
    w_m3  = weights.get("mode3", 0.5)
    w_phy = weights.get("physics", 0.1)
    w_cal = weights.get("calibration", 0.1)
    w_ssim= weights.get("ssim", 0.1)

    loss_m1_task = zero.clone()
    loss_m2_task = zero.clone()
    loss_m3_task = zero.clone()
    loss_m1_cal  = zero.clone()
    loss_m2_cal  = zero.clone()
    loss_ssim    = zero.clone()
    loss_physics = zero.clone()

    # ---- Mode 1 ----
    mask1 = (target_mode == 1)
    if mask1.any() and pred["mode1"] is not None:
        p1 = pred["mode1"]
        t1 = target[mask1, 0]                  # H0 error label
        v1 = p1["h0_correction"]               # [n1]
        ls1= p1["log_sigma"]

        loss_m1_task = F.mse_loss(v1, t1)
        loss_m1_cal  = _gaussian_nll(v1, t1, ls1)

    # ---- Mode 2 ----
    mask2 = (target_mode == 2)
    if mask2.any() and pred["mode2"] is not None:
        p2  = pred["mode2"]
        v2  = p2["dm_correction"]              # [n2, max_dm_dim]
        t2  = target[mask2, :v2.shape[1]]     # [n2, max_dm_dim]  (target 차원 맞춤)
        ls2 = p2["log_sigma"]
        dims= target_dim[mask2]                # [n2]

        # 차원 마스킹: 각 샘플의 실제 dm_dim까지만 손실 계산
        max_k = v2.shape[1]
        dim_mask = torch.arange(max_k, device=device).unsqueeze(0) < dims.unsqueeze(1)
        diff2    = (v2 - t2) ** 2 * dim_mask.float()
        loss_m2_task = diff2.sum() / (dim_mask.float().sum() + 1e-8)

        nll2 = _gaussian_nll(
            v2[dim_mask], t2[dim_mask], ls2[dim_mask]
        )
        loss_m2_cal = nll2

    # ---- Mode 3 ----
    mask3 = (target_mode == 3)
    if mask3.any() and pred["mode3"] is not None:
        p3    = pred["mode3"]
        t_img = target_img[mask3].unsqueeze(1)      # [n3, 1, H, W]
        v3    = p3["source_residual"]               # [n3, 1, H, W]

        loss_m3_task = F.mse_loss(v3, t_img)
        loss_ssim    = _ssim_loss(v3, t_img)

    # ---- Physics penalty: D_Δt 일관성 (Mode 1·2 공통) ----
    # 간단 구현: Mode 1 예측 H0 보정값과 Mode 2 DM 파라미터 보정값의
    # 크기 패널티 (실제 일관성 검증은 full physics 필요 — 여기선 L2 regularization 대리)
    if pred["mode1"] is not None:
        loss_physics = loss_physics + 0.5 * (pred["mode1"]["h0_correction"] ** 2).mean()
    if pred["mode2"] is not None:
        loss_physics = loss_physics + 0.5 * (pred["mode2"]["dm_correction"] ** 2).mean()

    total = (
        w_m1  * (loss_m1_task + w_cal * loss_m1_cal)
      + w_m2  * (loss_m2_task + w_cal * loss_m2_cal)
      + w_m3  * (loss_m3_task + w_ssim * loss_ssim)
      + w_phy * loss_physics
    )

    return {
        "total":      total,
        "mode1_task": loss_m1_task,
        "mode2_task": loss_m2_task,
        "mode3_task": loss_m3_task,
        "mode1_cal":  loss_m1_cal,
        "mode2_cal":  loss_m2_cal,
        "physics":    loss_physics,
        "ssim":       loss_ssim,
    }
