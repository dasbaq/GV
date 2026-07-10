"""Phase 5 Mode 1 ML 보정 결합 테스트.

(a) obs_to_features 어댑터가 dataset.__getitem__의 Mode 1 입력을 그대로 재현하는지,
(b) v0.6 호환 checkpoint로 H0_corrected = H0_approx + correction이 닫히는지,
(c) graceful skip 동작,
(d) v0.5 checkpoint가 v0.6 모델과 비호환임을 단언(예상된 동작 문서화)을 검증한다.

v0.6: Image 입력 (I_obs 1ch) 복구. Mode1Head in_dim=d_model×3=384.
v0.5 checkpoint (head1 in_dim=256)는 v0.6 모델(in_dim=384)과 비호환.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "mock" / "phase4_v0_4.h5"
CKPT_V04 = ROOT / "data" / "checkpoints" / "phase4_v0_4_imgres_best.pt"
CKPT_V05 = ROOT / "data" / "checkpoints" / "phase4_v0_5_imgres_best.pt"
SCALER = ROOT / "data" / "target_scaler_phase4_v0_4.pkl"
CONFIG = ROOT / "config" / "ml.yaml"

# v0.6 호환 checkpoint — Kaggle v0.6 재학습 후 이 경로에 저장될 예정
CKPT_V06 = ROOT / "data" / "checkpoints" / "phase4_v0_6_imgres_best.pt"


def _checkpoint_is_v06_compatible(path: Path) -> bool:
    """head1.net.0.weight shape가 v0.6 in_dim=384이고 img_enc가 있는지 확인한다."""
    if not path.exists():
        return False
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        w = sd.get("head1.net.0.weight")
        has_img_enc = any(k.startswith("img_enc.") for k in sd)
        no_head3 = not any(k.startswith("head3.") for k in sd)
        return (
            w is not None
            and w.shape[1] == 384
            and has_img_enc
            and no_head3
        )
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
        image_size=int(cfg["data"].get("image_size", 64)),
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
        image_size=int(cfg["data"].get("image_size", 64)),
        approx_level=1,
        target_mode=1,
    )
    # dataset 출력(배치 차원 없음) vs 어댑터(배치 1) 비교
    for key in ("lc", "params", "sigma_curve", "image"):
        a = got[key][0].numpy()
        b = sample[key].numpy()
        assert a.shape == b.shape, f"{key} shape {a.shape} != {b.shape}"
        assert np.allclose(a, b, atol=1e-6), f"{key} mismatch (max {np.abs(a-b).max():.2e})"
    assert int(got["target_mode"][0]) == int(sample["target_mode"])


@pytest.mark.skipif(
    not (CATALOG.exists() and SCALER.exists() and _checkpoint_is_v06_compatible(CKPT_V06)),
    reason=(
        "v0.6 호환 checkpoint 없음 — Kaggle v0.6 재학습 후 "
        "data/checkpoints/phase4_v0_6_imgres_best.pt 로 교체 예정. "
        "v0.5 checkpoint(head1 in_dim=256)는 v0.6 모델(in_dim=384)과 비호환."
    ),
)
def test_correction_closes_with_v0_6_checkpoint():
    """v0.6 재학습 checkpoint로 H0_corrected = H0_approx + correction이 닫히는지 검증."""
    from inversion.obs_to_features import (
        build_corrector_inputs,
        load_corrector,
        load_target_scaler,
        system_spec_from_hdf5,
    )

    model, cfg = load_corrector(CKPT_V06, CONFIG)
    scaler = load_target_scaler(SCALER)
    spec = system_spec_from_hdf5(CATALOG, 0)
    inputs = build_corrector_inputs(
        spec,
        param_norm=cfg["data"]["param_normalization"],
        max_len=int(cfg["data"]["max_lc_len"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
        image_size=int(cfg["data"].get("image_size", 64)),
    )
    with torch.no_grad():
        out = model(**inputs)
    assert out["mode1"] is not None
    s = scaler["mode1"]
    pred_scaled = float(out["mode1"]["h0_correction"].item())
    correction = pred_scaled * float(s["scale"]) + float(s["mean"])
    h0_corrected = float(spec["H0_approx"]) + correction
    assert np.isfinite(correction)
    assert np.isfinite(h0_corrected)
    assert 0.0 < correction < 80.0, f"correction out of range: {correction}"


@pytest.mark.skipif(not CKPT_V05.exists(), reason="v0.5 checkpoint not present")
def test_v0_5_checkpoint_incompatible_with_v0_6_model():
    """v0.5 checkpoint가 v0.6 모델과 비호환임을 단언한다.

    v0.5: Mode1Head in_dim = d_model×2 = 256 (image 없음)
    v0.6: Mode1Head in_dim = d_model×3 = 384 (image 복구)

    이 테스트가 통과(RuntimeError 발생)하는 동안 v0.6 재학습이 필요하다.
    """
    import yaml
    from ml.models.error_corrector import MultiModalErrorCorrector

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    mc = dict(cfg["model"])
    mc["param_in_dim"] = len(cfg["data"]["param_normalization"]) + 5
    model = MultiModalErrorCorrector(mc)

    sd = torch.load(CKPT_V05, map_location="cpu", weights_only=True)

    # v0.5 checkpoint에는 img_enc 키가 없어야 한다
    assert not any(k.startswith("img_enc.") for k in sd), \
        "v0.5 checkpoint should NOT have img_enc.* keys"

    # head1.net.0.weight in_dim이 v0.5(256)여야 한다
    w = sd.get("head1.net.0.weight")
    assert w is not None and w.shape[1] == 256, \
        f"expected v0.5 head1 in_dim=256, got {w.shape if w is not None else None}"

    # v0.6 모델에 strict=True로 로드하면 RuntimeError가 발생해야 한다
    with pytest.raises(RuntimeError, match="size mismatch|Missing key|Unexpected key"):
        model.load_state_dict(sd, strict=True)


def test_apply_correction_graceful_skip_without_features(tmp_path: Path):
    from pipelines.run_mode1 import _apply_ml_correction
    from tests.test_run_mode1_e2e import _mode1_delay_cfg, _write_forward_observation_h5
    from inversion.delay_extraction import extract_delay_from_observation
    from inversion.mode1_h0 import invert_h0
    from inversion.observation_io import from_hdf5
    from inversion.sie_fit import fit_sie_to_images

    input_path, dt_true = _write_forward_observation_h5(tmp_path / "mode1_skip_smoke.h5")
    observation = from_hdf5(input_path)
    delay = extract_delay_from_observation(
        observation,
        _mode1_delay_cfg(dt_true),
        is_mock=True,
        return_grid=False,
    )
    sie_fit = fit_sie_to_images(
        observation.image_positions,
        observation.z_lens,
        observation.z_source,
        cosmology={"H0": 70.0},
    )
    h0_approx = float(
        invert_h0(
            np.array([delay["dt_obs_days"]], dtype=float),
            np.array([sie_fit["dphi_rad2"]], dtype=float),
            observation.z_lens,
            observation.z_source,
            approx_level=0,
            n_bootstrap=10,
        )["H0"]
    )

    h0, info = _apply_ml_correction(
        h0_approx,
        apply_correction=True,
        checkpoint=None,
        scaler=SCALER,
        config=CONFIG,
        device="cpu",
        correction_approx_level=1,
        mode1_sigma_scale=1.0,
        observation=observation,
        input_path=input_path,
        system_index=0,
        delay=delay,
        sie_fit=sie_fit,
    )
    assert h0 == h0_approx and info["applied"] is False

    if CKPT_V05.exists():
        h0b, infob = _apply_ml_correction(
            h0_approx,
            apply_correction=True,
            checkpoint=CKPT_V05,
            scaler=ROOT / "data" / "missing_scaler.pkl",
            config=CONFIG,
            device="cpu",
            correction_approx_level=1,
            mode1_sigma_scale=1.0,
            observation=observation,
            input_path=input_path,
            system_index=0,
            delay=delay,
            sie_fit=sie_fit,
        )
        assert h0b == h0_approx and infob["applied"] is False
        assert "scaler not found" in infob["reason"]


@pytest.mark.skipif(not CKPT_V04.exists(), reason="v0.4 checkpoint not present")
def test_legacy_checkpoint_gets_actionable_compatibility_error(tmp_path: Path) -> None:
    from pipelines.run_mode1 import run_mode1
    from tests.test_run_mode1_e2e import _mode1_delay_cfg, _write_forward_observation_h5

    input_path, dt_true = _write_forward_observation_h5(tmp_path / "mode1_obs.h5")
    with pytest.raises(RuntimeError, match="phase4_v0_6 checkpoint"):
        run_mode1(
            input_path,
            approx_level=0,
            apply_correction=True,
            correction_checkpoint=CKPT_V04,
            correction_scaler=SCALER,
            correction_device="cpu",
            delay_config=_mode1_delay_cfg(dt_true),
        )
