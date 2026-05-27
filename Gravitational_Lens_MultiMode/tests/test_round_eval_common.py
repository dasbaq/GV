from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from ml.training.round_eval import evaluate_mode1_h0_on_loader, mode2_correction_availability


class _FixedMode1Model(torch.nn.Module):
    def __init__(self, pred_scaled: torch.Tensor, log_sigma: torch.Tensor) -> None:
        super().__init__()
        self.pred_scaled = pred_scaled
        self.log_sigma = log_sigma
        self.offset = 0

    def forward(self, **kwargs):
        b = int(kwargs["target_mode"].shape[0])
        lo = self.offset
        hi = lo + b
        self.offset = hi
        return {
            "mode1": {
                "h0_correction": self.pred_scaled[lo:hi].to(kwargs["target_mode"].device),
                "log_sigma": self.log_sigma[lo:hi].to(kwargs["target_mode"].device),
            },
            "mode2": None,
            "fused": None,
        }


def _write_eval_h5(path: Path, *, mode2_nonzero: bool = False) -> None:
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_systems"] = 3
        truth = f.create_group("true_values")
        truth.create_dataset("H0_true", data=np.array([71.0, 74.0, 78.0], dtype=np.float32))
        approx = f.create_group("approx_outputs")
        approx.create_dataset("H0_approx", data=np.array([70.0, 72.0, 75.0], dtype=np.float32))
        corr = f.create_group("correction_targets")
        corr.create_dataset("mode1_H0_correction", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        mode2 = np.zeros((3, 4), dtype=np.float32)
        if mode2_nonzero:
            mode2[1, 0] = 0.1
        corr.create_dataset("mode2_dm_correction", data=mode2)


def _loader():
    batch = {
        "lc": torch.zeros(3, 2, 4),
        "lc_mask": torch.ones(3, 4, dtype=torch.bool),
        "params": torch.zeros(3, 20),
        "sigma_curve": torch.zeros(3, 1, 4),
        "image": torch.zeros(3, 1, 64, 64),   # v0.6: I_obs zero tensor
        "target_mode": torch.ones(3, dtype=torch.long),
    }
    return [batch]


def test_shared_round_eval_inverts_scaler_and_computes_h0(tmp_path):
    h5_path = tmp_path / "eval.h5"
    _write_eval_h5(h5_path)
    model = _FixedMode1Model(
        pred_scaled=torch.tensor([0.0, 1.0, 2.0]),
        log_sigma=torch.zeros(3),
    )
    scaler = {"mode1": {"mean": 1.0, "scale": 1.0}}
    result = evaluate_mode1_h0_on_loader(
        model=model,
        loader=_loader(),
        device=torch.device("cpu"),
        path=h5_path,
        ids=np.arange(3),
        scaler=scaler,
        bootstrap_n=4,
        output_path=None,
        label="unit",
        training_summary={},
        infra={},
        checkpoint_display="ckpt",
        data_display="data",
        scaler_display="scaler",
    )
    metrics = result["best"]["mode1"]["h0"]["model"]
    assert metrics["RMSE"] == 0.0
    assert result["best"]["mode1"]["correction_prediction"]["positive_fraction"] == 1.0
    assert result["best"]["mode1"]["log_sigma_calibration"]["coverage_abs_residual_le_1sigma"] == 1.0


def test_mode2_availability_reports_zero_placeholders(tmp_path):
    h5_path = tmp_path / "eval.h5"
    _write_eval_h5(h5_path, mode2_nonzero=False)
    result = mode2_correction_availability(h5_path)
    assert result["available_for_real_mode2_training"] is False
    assert "데이터 선행 필요" in result["decision"]


def test_mode2_availability_detects_nonzero_targets(tmp_path):
    h5_path = tmp_path / "eval.h5"
    _write_eval_h5(h5_path, mode2_nonzero=True)
    result = mode2_correction_availability(h5_path)
    assert result["available_for_real_mode2_training"] is True
