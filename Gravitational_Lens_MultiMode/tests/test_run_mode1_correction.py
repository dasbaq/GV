"""Phase 5 Mode 1 ML 보정 결합 테스트.

(a) obs_to_features 어댑터가 dataset.__getitem__의 Mode 1 입력을 그대로 재현하는지,
(b) v0.5 호환 checkpoint로 H0_corrected = H0_approx + correction이 닫히는지,
(c) graceful skip 동작,
(d) v0.4 checkpoint가 v0.5 모델과 비호환임을 단언(예상된 동작 문서화)을 검증한다.

Image 입력 모달리티는 삭제됨 (DECISIONS.md [2026-05-25] 참조).
v0.4 checkpoint는 Mode1Head in_dim=384(d_model×3)이라 v0.5(in_dim=256, d_model×2)와
비호환이다. Kaggle v0.5 재학습 완료 후 새 checkpoint로 교체 예정.
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

# v0.5 호환 checkpoint는 Kaggle v0.5 재학습 후 이 경로에 저장될 예정
CKPT_V05 = ROOT / "data" / "checkpoints" / "phase4_v0_5_imgres_best.pt"


def _checkpoint_is_v05_compatible(path: Path) -> bool:
    """head1.net.0.weight shape가 v0.5 in_dim=256인지 확인한다."""
    if not path.exists():
        return False
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        w = sd.get("head1.net.0.weight")
        return w is not None and w.shape[1] == 256
    except Exception:
        return False


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
        max_len=int(cfg["data"]["max_lc_len"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
        approx_level=1,
        target_mode=1,
    )
    # dataset 출력(배치 차원 없음) vs 어댑터(배치 1) 비교
    for key in ("lc", "params", "sigma_curve"):
        a = got[key][0].numpy()
        b = sample[key].numpy()
        assert a.shape == b.shape, f"{key} shape {a.shape} != {b.shape}"
        assert np.allclose(a, b, atol=1e-6), f"{key} mismatch (max {np.abs(a-b).max():.2e})"
    assert int(got["target_mode"][0]) == int(sample["target_mode"])


@pytest.mark.skipif(
    not (CATALOG.exists() and SCALER.exists() and _checkpoint_is_v05_compatible(CKPT_V05)),
    reason=(
        "v0.5 호환 checkpoint 없음 — Kaggle v0.5 재학습 후 "
        "data/checkpoints/phase4_v0_5_imgres_best.pt 로 교체 예정. "
        "v0.4 checkpoint(head1 in_dim=384)는 v0.5 모델(in_dim=256)과 비호환 "
        "(test_v0_4_checkpoint_incompatible_with_v0_5 참조)."
    ),
)
def test_correction_closes_with_v0_5_checkpoint():
    """v0.5 재학습 checkpoint로 H0_corrected = H0_approx + correction이 닫히는지 검증."""
    from inversion.obs_to_features import (
        build_corrector_inputs,
        load_corrector,
        load_target_scaler,
        system_spec_from_hdf5,
    )

    model, cfg = load_corrector(CKPT_V05, CONFIG)
    scaler = load_target_scaler(SCALER)
    spec = system_spec_from_hdf5(CATALOG, 0)
    inputs = build_corrector_inputs(
        spec,
        param_norm=cfg["data"]["param_normalization"],
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


@pytest.mark.skipif(not CKPT.exists(), reason="v0.4 checkpoint not present")
def test_v0_4_checkpoint_incompatible_with_v0_5_model():
    """v0.4 checkpoint가 v0.5 모델과 비호환임을 단언한다.

    v0.4: Mode1Head in_dim = d_model×3 = 384 (fused + h_lc + h_img 연결)
    v0.5: Mode1Head in_dim = d_model×2 = 256 (fused + h_lc 연결, image 삭제)

    이 테스트가 통과(RuntimeError 발생)하는 동안 v0.5 재학습이 필요하다.
    새 checkpoint 교체 후 test_correction_closes_with_v0_5_checkpoint가 활성화된다.
    """
    import yaml
    from ml.models.error_corrector import MultiModalErrorCorrector

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    mc = dict(cfg["model"])
    mc["param_in_dim"] = len(cfg["data"]["param_normalization"]) + 5
    model = MultiModalErrorCorrector(mc)

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)

    # v0.4 checkpoint에는 삭제된 키들이 남아 있어야 한다
    assert any(k.startswith("img_enc.") for k in sd), \
        "expected img_enc.* keys in v0.4 checkpoint"
    assert any(k.startswith("head3.") for k in sd), \
        "expected head3.* keys in v0.4 checkpoint"

    # head1.net.0.weight in_dim이 v0.4(384)여야 한다
    w = sd.get("head1.net.0.weight")
    assert w is not None and w.shape[1] == 384, \
        f"expected v0.4 head1 in_dim=384, got {w.shape if w is not None else None}"

    # v0.5 모델에 strict=True로 로드하면 RuntimeError가 발생해야 한다
    with pytest.raises(RuntimeError, match="size mismatch|Unexpected key"):
        model.load_state_dict(sd, strict=True)


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
