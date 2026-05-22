"""Phase 5 Mode 1 ML 보정 결합 테스트.

(b) obs_to_features 어댑터가 dataset.__getitem__의 Mode 1 입력을 그대로 재현하는지,
(c) 학습된 v0.4 corrector + scaler로 H0_corrected = H0_approx + correction이 닫히는지,
graceful skip 동작을 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "mock" / "phase4_v0_4.h5"
CKPT = ROOT / "data" / "checkpoints" / "phase4_v0_4_imgres_best.pt"
SCALER = ROOT / "data" / "target_scaler_phase4_v0_4.pkl"
CONFIG = ROOT / "config" / "ml.yaml"


def _cfg() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


@pytest.mark.skipif(not CATALOG.exists(), reason="v0.4 catalog not present")
def test_adapter_matches_dataset_getitem():
    from ml.training.dataset import LensCorrectionDataset
    from inversion.obs_to_features import build_corrector_inputs, system_spec_from_hdf5

    cfg = _cfg()
    pn = cfg["data"]["param_normalization"]
    ds = LensCorrectionDataset(
        [CATALOG],
        split="train",
        modes=(1,),
        approx_levels=(1,),
        image_size=int(cfg["data"]["image_size"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
        max_len=int(cfg["data"]["max_lc_len"]),
        param_norm=pn,
        seed=int(cfg["seed"]),
    )
    path, sys_idx, al, mode = ds._index[0]
    assert al == 1 and mode == 1
    sample = ds[0]

    spec = system_spec_from_hdf5(path, sys_idx)
    got = build_corrector_inputs(
        spec,
        param_norm=pn,
        image_size=int(cfg["data"]["image_size"]),
        max_len=int(cfg["data"]["max_lc_len"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
        approx_level=1,
        target_mode=1,
    )
    # dataset 출력(배치 차원 없음) vs 어댑터(배치 1) 비교
    for key in ("lc", "params", "sigma_curve", "image"):
        a = got[key][0].numpy()
        b = sample[key].numpy()
        assert a.shape == b.shape, f"{key} shape {a.shape} != {b.shape}"
        assert np.allclose(a, b, atol=1e-6), f"{key} mismatch (max {np.abs(a-b).max():.2e})"
    assert bool(got["use_image"][0]) == bool(sample["use_image"])
    assert int(got["target_mode"][0]) == int(sample["target_mode"])


@pytest.mark.skipif(
    not (CATALOG.exists() and CKPT.exists() and SCALER.exists()),
    reason="v0.4 catalog/checkpoint/scaler not present",
)
def test_correction_closes_with_v0_4_checkpoint():
    from inversion.obs_to_features import (
        build_corrector_inputs,
        load_corrector,
        load_target_scaler,
        system_spec_from_hdf5,
    )

    model, cfg = load_corrector(CKPT, CONFIG)
    scaler = load_target_scaler(SCALER)
    spec = system_spec_from_hdf5(CATALOG, 0)
    inputs = build_corrector_inputs(
        spec,
        param_norm=cfg["data"]["param_normalization"],
        image_size=int(cfg["data"]["image_size"]),
        max_len=int(cfg["data"]["max_lc_len"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
    )
    with torch.no_grad():
        out = model(**inputs)
    assert out["mode1"] is not None
    s = scaler["mode1"]
    pred_scaled = float(out["mode1"]["h0_correction"].item())
    correction = pred_scaled * float(s["scale"]) + float(s["mean"])
    h0_corrected = float(spec["H0_approx"]) + correction
    # round eval과 동일한 역변환식: model_h0 = H0_approx + pred*scale + mean
    assert np.isfinite(correction)
    assert np.isfinite(h0_corrected)
    # correction은 v0.4 분포(mean~29.6)와 같은 부호/스케일 영역이어야 함
    assert 0.0 < correction < 80.0, f"correction out of range: {correction}"


def test_apply_correction_graceful_skip_without_features():
    from pipelines.run_mode1 import _apply_ml_correction

    # checkpoint가 없으면 skip
    h0, info = _apply_ml_correction(
        70.0,
        apply_correction=True,
        checkpoint=None,
        feature_spec=None,
        scaler_path=None,
        config_path=None,
    )
    assert h0 == 70.0 and info["applied"] is False

    # checkpoint는 있으나 feature_spec이 없으면 skip(reason 명시)
    if CKPT.exists():
        h0b, infob = _apply_ml_correction(
            70.0,
            apply_correction=True,
            checkpoint=CKPT,
            feature_spec=None,
            scaler_path=SCALER if SCALER.exists() else None,
            config_path=CONFIG,
        )
        assert h0b == 70.0 and infob["applied"] is False
        assert "unavailable" in infob["reason"]
