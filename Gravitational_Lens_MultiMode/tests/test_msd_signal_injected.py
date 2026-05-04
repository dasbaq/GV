from pathlib import Path

import h5py
import numpy as np


# Canonical MSD injection checks:
#   - delta_t_obs/delta_t_sie ~= 1 - kappa_ext  (exact, r=1.0)
#   - mean_I_obs_preclip/mean_I_sie ~= 1/(1 - kappa_ext)  (exact, r=1.0)
# postclip ratio is a clip-impact sanity check (>0.99).
DATA_PATH = Path("data/mock/real_phase3_v2_1.h5")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _load_perturbations() -> dict[str, np.ndarray]:
    with h5py.File(DATA_PATH, "r") as f:
        p = f["perturbations"]
        return {
            "kappa_ext": p["kappa_ext"][:],
            "delta_t_sie": p["delta_t_sie"][:],
            "delta_t_obs": p["delta_t_obs"][:],
            "mean_I_sie": p["mean_I_sie"][:],
            "mean_I_obs_preclip": p["mean_I_obs_preclip"][:],
            "mean_I_obs": p["mean_I_obs"][:],
            "clip_fraction": p["clip_fraction"][:],
        }


def test_msd_time_delay_scaling_exact():
    data = _load_perturbations()
    expected = 1.0 - data["kappa_ext"]
    ratio = data["delta_t_obs"] / data["delta_t_sie"]
    r = _corr(ratio, expected)
    print(f"time_delay corr={r:.4f}")
    assert r > 0.999, f"time-delay scaling corr={r:.4f}, expected > 0.999"
    assert np.allclose(ratio, expected, atol=1e-6), "delta_t_obs/delta_t_sie differs from 1-kappa_ext"


def test_msd_image_brightness_preclip_scaling():
    data = _load_perturbations()
    expected = 1.0 / (1.0 - data["kappa_ext"])
    ratio = data["mean_I_obs_preclip"] / data["mean_I_sie"]
    r = _corr(ratio, expected)
    print(f"image preclip corr={r:.4f}")
    assert r > 0.999, f"preclip image brightness corr={r:.4f}, expected > 0.999"


def test_msd_image_brightness_postclip_morphology_controlled():
    data = _load_perturbations()
    expected = 1.0 / (1.0 - data["kappa_ext"])
    ratio = data["mean_I_obs"] / data["mean_I_sie"]
    r = _corr(ratio, expected)
    print(f"image postclip ratio corr={r:.4f}")
    # 0.99로 설정한 이유: clip_fraction이 ~0.06%인 한도 내에서 극소수 포화
    # 픽셀이 ratio corr을 ~0.997 부근으로 떨어뜨림. preclip probe(r=1.0000)가
    # canonical 검증이고, 이 테스트는 clip 영향이 학습 신호를 지우지 않음을
    # 확인하는 sanity check.
    assert r > 0.99, f"postclip ratio corr={r:.4f}, expected > 0.99"


def test_msd_clip_fraction_sanity():
    data = _load_perturbations()
    clip_mean = float(data["clip_fraction"].mean())
    clip_max = float(data["clip_fraction"].max())
    print(f"clip_fraction stats: mean={clip_mean:.4f}, max={clip_max:.4f}")
    assert clip_mean < 0.5, f"clip_fraction mean={clip_mean:.4f}, expected < 0.5"


def test_morphology_dominates_raw_mean_diagnostic():
    data = _load_perturbations()
    sie_cv = float(data["mean_I_sie"].std() / data["mean_I_sie"].mean())
    msd_scale = 1.0 / (1.0 - data["kappa_ext"])
    msd_scale_range = float(msd_scale.max() - msd_scale.min())
    print(f"morphology CV(mean_I_sie)={sie_cv:.4f}, MSD scale range={msd_scale_range:.4f}")
    assert sie_cv > msd_scale_range, (
        "raw mean probe can detect MSD without morphology control; "
        "ratio-probe assumption should be revisited."
    )
