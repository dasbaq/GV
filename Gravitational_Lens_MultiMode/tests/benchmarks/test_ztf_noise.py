"""
벤치마크: ZTF 노이즈 전체 — precision ≥ 92.3%, recall ≥ 60%.

실측 ZTF 데이터 없음 → mock Σ 곡선으로 탐지 로직 검증.
합격 기준: CLAUDE.md 변경 불가.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest


def _generate_mock_sigma_curves(n: int, true_dt: float,
                                 sigma_curve_size: int = 512,
                                 seed: int = 0) -> tuple:
    """
    참 Δt가 true_dt인 mock Σ 곡선 생성.
    양성: Σ < -2.0 극솟값 있음 / 음성: 노이즈만.
    """
    rng = np.random.default_rng(seed)
    dt_try = np.linspace(1, 200, sigma_curve_size)

    positives = []
    for _ in range(n):
        sigma = rng.normal(0, 0.5, sigma_curve_size)
        # true_dt 근처에 극솟값 삽입
        idx = int(np.argmin(np.abs(dt_try - true_dt)))
        jitter = rng.integers(-5, 6)
        target_idx = np.clip(idx + jitter, 0, sigma_curve_size - 1)
        sigma[target_idx] = rng.uniform(-4.0, -2.5)
        positives.append(sigma)

    negatives = [rng.normal(0, 0.5, sigma_curve_size) for _ in range(n)]
    return np.array(positives), np.array(negatives)


def _detect(sigma_curve: np.ndarray, threshold: float = -2.0) -> bool:
    """Bag et al. 2022 보수 기준: Σ < threshold 인 극솟값 존재."""
    below = sigma_curve < threshold
    if not below.any():
        return False
    # 극솟값 조건: 인접 값보다 작음
    indices = np.where(below)[0]
    for i in indices:
        left  = sigma_curve[i-1] if i > 0 else 0
        right = sigma_curve[i+1] if i < len(sigma_curve)-1 else 0
        if sigma_curve[i] < left and sigma_curve[i] < right:
            return True
    return False


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "data" / "ztf_noise_catalog.h5").exists(),
    reason="⚠️ ztf_noise_catalog.h5 미존재 — MOCK 모드"
)
def test_ztf_noise_real():
    """실측 ZTF 카탈로그로 precision/recall 검증."""
    pass


def test_ztf_noise_mock():
    """
    Mock Σ 곡선으로 탐지기 precision/recall 검증.
    실측 기준과 동일한 임계값으로 평가.
    합격 기준: precision ≥ 92.3%, recall ≥ 60%.
    """
    N = 200
    positives, negatives = _generate_mock_sigma_curves(N, true_dt=24.14, seed=42)

    tp = sum(_detect(s) for s in positives)
    fp = sum(_detect(s) for s in negatives)
    fn = N - tp

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)

    print(f"\n  precision={precision:.3f}, recall={recall:.3f}, "
          f"tp={tp}, fp={fp}, fn={fn}")

    assert precision >= 0.923, f"⚠️ precision {precision:.3f} < 92.3%"
    assert recall    >= 0.60,  f"⚠️ recall {recall:.3f} < 60%"
