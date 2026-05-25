"""AMP/fp16 안전성 회귀 테스트 — NLL이 autocast 하에서 finite한지 검증.

배경: Phase 4 v0.3/v0.3.1 학습이 epoch1부터 NaN으로 붕괴했고, 발원지는
``torch.amp.autocast`` 안에서 fp16으로 계산되던 ``_ssim_loss``였다.
SSIM loss는 v0.5에서 삭제됨 (DECISIONS.md [2026-05-25] 참조).
이 테스트는 NLL과 physics loss가 extreme inputs 하에서도 finite한지 고정한다.
"""

from __future__ import annotations

import torch

from ml.training.losses import _gaussian_nll, _physics_d_dt_consistency_loss


def _autocast_cpu():
    return torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16)


def test_nll_finite_on_extreme_log_sigma_under_autocast():
    pred = torch.randn(64)
    target = torch.randn(64)
    for ls in (-5.0, -20.0, 2.0, 20.0):
        log_sigma = torch.full((64,), ls)
        with _autocast_cpu():
            loss = _gaussian_nll(pred, target, log_sigma)
        assert torch.isfinite(loss), f"NLL nan/inf at log_sigma={ls}"


def test_nll_matches_closed_form_on_normal_inputs():
    pred = torch.zeros(3)
    target = torch.zeros(3)
    log_sigma = torch.zeros(3)  # var = 1
    loss = _gaussian_nll(pred, target, log_sigma)
    # 0.5 * (0/1 + 0) = 0
    assert abs(float(loss)) < 1e-6


def _physics_batch(label_available=True):
    return {
        "target": torch.zeros(2, 6),
        "h0_approx": torch.full((2,), 70.0),
        "dt_lc": torch.tensor([20.0, 30.0]),
        "theta_E_approx": torch.ones(2),
        "dphi_sie_rad2": torch.tensor([1.0e-11, 1.5e-11]),
        "mode1_target_mean": torch.zeros(2),
        "mode1_target_scale": torch.ones(2),
        "mode2_target_mean": torch.zeros(2, 4),
        "mode2_target_scale": torch.ones(2, 4),
        "mode2_label_available": torch.full((2,), bool(label_available), dtype=torch.bool),
    }


def test_physics_loss_zero_for_zero_mode2_labels():
    pred = {
        "mode1": {"h0_correction": torch.tensor([1.0, -2.0])},
        "mode2": {"dm_correction": torch.ones(2, 4)},
    }
    loss = _physics_d_dt_consistency_loss(pred, _physics_batch(label_available=False))
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_physics_loss_finite_under_autocast():
    pred = {
        "mode1": {"h0_correction": torch.tensor([1.0, -2.0])},
        "mode2": {"dm_correction": torch.tensor([[0.01, 0.0, 0.0, 0.0], [-0.02, 0.0, 0.0, 0.0]])},
    }
    with _autocast_cpu():
        loss = _physics_d_dt_consistency_loss(pred, _physics_batch(label_available=True))
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_physics_loss_ignores_missing_mode1_or_mode2_payload():
    loss = _physics_d_dt_consistency_loss({"mode1": None, "mode2": None}, _physics_batch())
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
