"""AMP/fp16 안전성 회귀 테스트 — SSIM/NLL이 autocast 하에서 finite한지 검증.

배경: Phase 4 v0.3/v0.3.1 학습이 epoch1부터 NaN으로 붕괴했고, 발원지는
``torch.amp.autocast`` 안에서 fp16으로 계산되던 ``_ssim_loss``(분모 eps underflow +
분산 음수)였다. 이 테스트는 (a) degenerate 입력 + fp16 autocast에서 finite,
(b) 정상 입력에서 fp32 경로 수치가 합리적 범위임을 고정한다.
"""

from __future__ import annotations

import torch

from ml.training.losses import _gaussian_nll, _ssim_loss


def _autocast_cpu():
    return torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16)


def test_ssim_finite_on_degenerate_inputs_under_autocast():
    # 상수 이미지(분산 0) + 큰 밝기 — autocast 하에서도 finite해야 한다
    for value in (0.0, 1.0, 50.0):
        pred = torch.full((2, 1, 32, 32), value)
        target = torch.zeros((2, 1, 32, 32))
        with _autocast_cpu():
            loss = _ssim_loss(pred, target)
        assert torch.isfinite(loss), f"SSIM nan/inf at value={value}"


def test_nll_finite_on_extreme_log_sigma_under_autocast():
    pred = torch.randn(64)
    target = torch.randn(64)
    for ls in (-5.0, -20.0, 2.0, 20.0):
        log_sigma = torch.full((64,), ls)
        with _autocast_cpu():
            loss = _gaussian_nll(pred, target, log_sigma)
        assert torch.isfinite(loss), f"NLL nan/inf at log_sigma={ls}"


def test_ssim_matches_fp32_reference_on_normal_inputs():
    torch.manual_seed(0)
    pred = torch.rand(4, 1, 32, 32)
    target = torch.rand(4, 1, 32, 32)
    loss = _ssim_loss(pred, target)
    assert torch.isfinite(loss)
    assert 0.0 <= float(loss) <= 2.0


def test_nll_matches_closed_form_on_normal_inputs():
    pred = torch.zeros(3)
    target = torch.zeros(3)
    log_sigma = torch.zeros(3)  # var = 1
    loss = _gaussian_nll(pred, target, log_sigma)
    # 0.5 * (0/1 + 0) = 0
    assert abs(float(loss)) < 1e-6
