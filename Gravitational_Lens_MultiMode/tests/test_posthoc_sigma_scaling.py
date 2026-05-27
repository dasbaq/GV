from __future__ import annotations

import numpy as np
import pytest

from scripts.phase4_v0_2_round import apply_mode1_sigma_scale, metrics


def test_mode1_sigma_scale_changes_only_sigma() -> None:
    y_true = np.array([70.0, 72.0, 74.0], dtype=np.float64)
    model_h0 = np.array([69.0, 73.5, 74.5], dtype=np.float64)
    pred_sigma = np.array([1.0, 2.0, 4.0], dtype=np.float64)

    before_metrics = metrics(y_true, model_h0)
    unscaled, scaled = apply_mode1_sigma_scale(pred_sigma, 1.47)
    after_metrics = metrics(y_true, model_h0)

    np.testing.assert_allclose(unscaled, pred_sigma)
    np.testing.assert_allclose(scaled, pred_sigma * 1.47)
    assert after_metrics == before_metrics


def test_mode1_sigma_scale_default_preserves_sigma() -> None:
    pred_sigma = np.array([0.5, 1.5, 3.0], dtype=np.float64)

    unscaled, scaled = apply_mode1_sigma_scale(pred_sigma, 1.0)

    np.testing.assert_allclose(unscaled, pred_sigma)
    np.testing.assert_allclose(scaled, pred_sigma)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_mode1_sigma_scale_rejects_invalid_values(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        apply_mode1_sigma_scale(np.array([1.0], dtype=np.float64), scale)
