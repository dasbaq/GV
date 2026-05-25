"""
composite_loss — Mode별 task loss + physics + calibration.

Mode 1: MSE(H0) + Gaussian NLL
Mode 2: 마스킹된 MSE(dm_params) + Gaussian NLL

Mode 3(Source 복원)과 SSIM loss는 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from core.physics.config import constants


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


def _physics_d_dt_consistency_loss(physics_pred: dict, batch: dict) -> torch.Tensor:
    """Mode 1/2 D_Δt consistency penalty in fp32.

    Units: H0 and Mode 1 correction are [km/s/Mpc], ``dt_lc`` is [days],
    ``theta_E`` is [arcsec], Fermat-potential differences are [rad²], and
    distances are [Mpc]. SIE 표준 근사 가정: Mode 2 uses only the fixed SIE
    parameter order ``[theta_E, q, position_angle, sigma_v]``. The Mode 2
    Fermat-potential update is a first-order scaling
    ``Δφ_corrected ≈ Δφ_SIE * (theta_E_corrected/theta_E_approx)^2``; this is
    a consistency regularizer, not a replacement for the full solver.

    If Mode 2 labels are all-zero placeholder data, ``mode2_label_available``
    masks those rows and the loss degrades to exactly zero.
    """

    mode1 = physics_pred.get("mode1")
    mode2 = physics_pred.get("mode2")
    if mode1 is None or mode2 is None:
        ref = batch["target"].float()
        return ref.new_tensor(0.0)

    required = (
        "h0_approx",
        "dt_lc",
        "theta_E_approx",
        "dphi_sie_rad2",
        "mode1_target_mean",
        "mode1_target_scale",
        "mode2_target_mean",
        "mode2_target_scale",
        "mode2_label_available",
    )
    if not all(key in batch for key in required):
        ref = batch["target"].float()
        return ref.new_tensor(0.0)

    device_type = "cuda" if batch["target"].is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        c = constants()
        c_m_s = float(c["c_m_s"])
        day_s = float(c["day_s"])
        mpc_m = float(c["Mpc_m"])

        h0_approx = batch["h0_approx"].float()
        dt_lc = batch["dt_lc"].float()
        theta_e = batch["theta_E_approx"].float()
        dphi_sie = batch["dphi_sie_rad2"].float()
        label_available = batch["mode2_label_available"].bool()

        h0_corr = (
            mode1["h0_correction"].float() * batch["mode1_target_scale"].float()
            + batch["mode1_target_mean"].float()
        )
        mode2_mean = batch["mode2_target_mean"].float()
        mode2_scale = batch["mode2_target_scale"].float()
        theta_corr = mode2["dm_correction"].float()[:, 0] * mode2_scale[:, 0] + mode2_mean[:, 0]

        h0_corrected = h0_approx + h0_corr
        theta_corrected = theta_e + theta_corr
        ddt_approx = (c_m_s * day_s * dt_lc) / (dphi_sie * mpc_m)
        ddt_mode1 = ddt_approx * h0_approx / h0_corrected.clamp_min(1.0e-6)
        dphi_mode2 = dphi_sie * (theta_corrected / theta_e.clamp_min(1.0e-6)).pow(2)
        ddt_mode2 = (c_m_s * day_s * dt_lc) / (dphi_mode2 * mpc_m)

        valid = (
            label_available
            & torch.isfinite(ddt_mode1)
            & torch.isfinite(ddt_mode2)
            & (dt_lc > 0.0)
            & (dphi_sie > 0.0)
            & (theta_e > 0.0)
            & (theta_corrected > 0.0)
            & (h0_approx > 0.0)
            & (h0_corrected > 0.0)
            & (ddt_mode1 > 0.0)
            & (ddt_mode2 > 0.0)
        )
        if not valid.any():
            return h0_approx.new_tensor(0.0)
        diff = torch.log(ddt_mode1[valid]) - torch.log(ddt_mode2[valid])
        return diff.pow(2).mean()


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
    weights : {mode1, mode2, physics, calibration}

    Returns
    -------
    dict: {total, mode1_task, mode2_task, mode1_cal, mode2_cal, physics}
    """
    device = batch["target"].device
    zero   = torch.tensor(0.0, device=device)

    target_mode = batch["target_mode"]              # [B]
    target      = batch["target"]                   # [B, max_label_dim]
    target_dim  = batch["target_dim"]               # [B]

    w_m1  = weights.get("mode1", 1.0)
    w_m2  = weights.get("mode2", 1.0)
    w_phy = weights.get("physics", 0.1)
    w_cal = weights.get("calibration", 0.1)

    loss_m1_task = zero.clone()
    loss_m2_task = zero.clone()
    loss_m1_cal  = zero.clone()
    loss_m2_cal  = zero.clone()
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

    # ---- Physics penalty: Mode 1/2 D_Δt 일관성 ----
    if pred.get("physics") is not None:
        loss_physics = _physics_d_dt_consistency_loss(pred["physics"], batch)

    total = (
        w_m1  * (loss_m1_task + w_cal * loss_m1_cal)
      + w_m2  * (loss_m2_task + w_cal * loss_m2_cal)
      + w_phy * loss_physics
    )

    return {
        "total":      total,
        "mode1_task": loss_m1_task,
        "mode2_task": loss_m2_task,
        "mode1_cal":  loss_m1_cal,
        "mode2_cal":  loss_m2_cal,
        "physics":    loss_physics,
    }
