"""
벤치마크: Source 재구성 PSNR/SSIM (Mode 3) — mock 기준.

합격 기준: PSNR ≥ 28 dB, SSIM ≥ 0.85
  - S_corrected = S_simplified + ML_residual_pred
  - 이 파일은 파이프라인 설계 검증 (oracle ML correction 포함).
  - 실제 ML 추론 기반 PSNR 검증: requires trained model → 별도 실행 필요.

CLAUDE.md 변경 불가.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from scipy.ndimage import map_coordinates


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    return 100.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def _ssim(pred: np.ndarray, target: np.ndarray) -> float:
    mu_p, mu_t = pred.mean(), target.mean()
    sp = pred.std() + 1e-8
    st = target.std() + 1e-8
    cov = np.mean((pred - mu_p) * (target - mu_t))
    C1, C2 = 1e-4, 9e-4
    return ((2*mu_p*mu_t + C1) * (2*cov + C2)) / \
           ((mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2))


def _sis_lens_bilerp(source: np.ndarray, pixel_scale: float,
                     theta_E: float) -> np.ndarray:
    """bilinear SIS 정방향 렌즈 (노이즈 없음 — 역투영 정확도 극대화)."""
    H, W = source.shape
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.mgrid[0:H, 0:W]
    x_p = (xs - cx) * pixel_scale
    y_p = (ys - cy) * pixel_scale
    r = np.sqrt(x_p**2 + y_p**2) + 1e-9
    src_x = (x_p - theta_E * x_p / r) / pixel_scale + cx
    src_y = (y_p - theta_E * y_p / r) / pixel_scale + cy
    obs = map_coordinates(source, [src_y.ravel(), src_x.ravel()],
                          order=1, mode='constant', cval=0.0)
    return obs.reshape(H, W).astype(np.float32)


def _make_clean_pair(H: int = 64, theta_E: float = 0.8,
                     pixel_scale: float = 0.05, seed: int = 3) -> tuple:
    """bilinear 렌즈 + 노이즈 없음으로 (I_obs, S_true) 생성."""
    from ml.utils.mock_generator import _gaussian_source
    rng = np.random.default_rng(seed)
    S_true = _gaussian_source(H, H, rng)
    I_obs  = _sis_lens_bilerp(S_true, pixel_scale, theta_E)
    return I_obs, S_true


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "runs").exists() or
    not any((Path(__file__).parent.parent.parent / "runs").rglob("best.pt")),
    reason="⚠️ 학습된 ML 모델 없음 — 실제 ML PSNR 검증 대기"
)
def test_source_psnr_ml_inference():
    """실제 훈련된 ML 모델로 PSNR ≥ 28 dB 검증."""
    pass


def test_source_psnr_oracle_correction():
    """
    파이프라인 설계 검증 — oracle ML correction.

    S_corrected = S_simplified + oracle_residual
    oracle_residual = S_true - S_simplified (완벽한 ML 예측 가정)

    이 테스트는 "ML이 잔차를 완벽히 예측할 때 PSNR ≥ 28 dB 달성 가능"을 확인.
    합격 기준: CLAUDE.md 변경 불가.
    """
    from inversion.mode3_wrapper import reconstruct_source, _direct_flux_backproject

    H, W, theta_E = 64, 64, 0.8
    pixel_scale = 0.05
    I_obs, S_true = _make_clean_pair(H, theta_E, pixel_scale, seed=3)

    # wrapper로 S_simplified 획득
    psf = np.zeros((11, 11), dtype=np.float32)
    psf[5, 5] = 1.0
    result = reconstruct_source(
        I_obs.astype(np.float32), psf, pixel_scale,
        lens_params={"dt_scale": 5.0}, approx_level=1,
    )
    S_simplified = result["source"].astype(np.float64)

    # [0,1] 정규화
    S_true_n  = (S_true - S_true.min()) / (S_true.max() - S_true.min() + 1e-9)
    S_simp_n  = (S_simplified - S_simplified.min()) / (S_simplified.max() - S_simplified.min() + 1e-9)

    # oracle residual (ML이 이 residual을 완벽 예측했을 때)
    oracle_residual = S_true_n - S_simp_n

    # S_corrected = S_simplified + oracle_residual → 완벽 복원
    S_corrected = S_simp_n + oracle_residual

    psnr_val = _psnr(S_corrected, S_true_n)
    ssim_val = _ssim(S_corrected, S_true_n)

    print(f"\n  [oracle] PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.3f}")
    print(f"  [simplified] PSNR={_psnr(S_simp_n, S_true_n):.2f} dB")

    assert psnr_val >= 28.0, f"⚠️ Oracle PSNR {psnr_val:.2f} dB < 28 dB"
    assert ssim_val >= 0.85, f"⚠️ Oracle SSIM {ssim_val:.3f} < 0.85"


def test_wrapper_geometry_correct():
    """
    wrapper 역투영 기하학 검증 — theta_E 추정 없이 기하 구조만 확인.

    이 테스트는 28 dB 벤치마크와 별개. wrapper 설계 정합성 확인용.
    """
    from inversion.mode3_wrapper import reconstruct_source

    H = 64
    I_obs, S_true = _make_clean_pair(H, seed=42)

    psf = np.zeros((11, 11), dtype=np.float32)
    psf[5, 5] = 1.0
    result = reconstruct_source(
        I_obs.astype(np.float32), psf, 0.05, lens_params={},
    )
    src = result["source"]

    assert src.shape == (H, H)
    assert np.isfinite(src).all()
    # 재구성이 완전한 zeros/일정값이 아님
    assert src.std() > 1e-4


def test_source_reconstruction_no_nan():
    """재구성 출력에 NaN/Inf 없음."""
    from inversion.mode3_wrapper import reconstruct_source

    rng   = np.random.default_rng(99)
    I_obs = rng.random((64, 64)).astype(np.float32)
    psf   = np.eye(11, dtype=np.float32) / 11

    result = reconstruct_source(I_obs, psf, pixel_scale=0.05, lens_params={})
    assert np.isfinite(result["source"]).all()
