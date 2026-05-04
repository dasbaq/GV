"""
Mode 1 — H₀ 역산 솔버.

입력: 관측 시간 지연 Δt_obs [days], 페르마 포텐셜 Δφ [arcsec²], 적색편이.
출력: H₀ [km/s/Mpc] + 불확도.

근사 레벨(approx_level)은 Mode와 직교 축.
  level 1 (FAST):  해석 근사 — flat ΛCDM, 선형 팽창 근사
  level 2 (TURBO): level 1 + D_Δt를 H₀=70 기준 룩업 스케일로 단순화

단위: Δt [days], H₀ [km/s/Mpc], D [Mpc], c [km/s]
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

# --------------------------------------------------------------------------- #
# 물리 상수 로드 (config/physics.yaml)                                          #
# --------------------------------------------------------------------------- #
def _load_physics() -> dict:
    import yaml
    cfg_path = Path(__file__).parent.parent / "config" / "physics.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_PHYS: dict[str, float] = {}

def _phys() -> dict[str, float]:
    global _PHYS
    if not _PHYS:
        _PHYS = _load_physics()
    return _PHYS


# --------------------------------------------------------------------------- #
# 거리 계산 (core/cosmology/distances.py 없으면 stub 사용)                      #
# --------------------------------------------------------------------------- #
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src_py"))
    from core.cosmology.distances import angular_diameter_distance as _add_ext
    _HAS_DISTANCES = True
except ImportError:
    _HAS_DISTANCES = False


def _angular_diameter_distance_approx(z: float, H0: float) -> float:
    """
    평탄 ΛCDM 해석 근사 (Ω_m=0.3, Ω_Λ=0.7).
    단위: [Mpc]  approx_level=1
    """
    ph = _phys()
    c = ph["c_km_s"]
    # Pen (1999) 근사: D_A ≈ (2c/H0) * [1 - 1/sqrt(1+z)] / (1+z)  for flat ΛCDM
    # 더 정확한 적분 근사 (Simpson 5점)
    def integrand(zp: float) -> float:
        Om = 0.3
        return 1.0 / np.sqrt(Om * (1 + zp) ** 3 + (1 - Om))

    n = 100
    zz = np.linspace(0.0, z, n + 1)
    fz = np.array([integrand(zi) for zi in zz])
    dh = c / H0
    comoving = dh * np.trapezoid(fz, zz)
    return comoving / (1.0 + z)


def _d_delta_t(H0: float, z_lens: float, z_source: float,
               approx_level: int) -> float:
    """
    D_Δt = (1+z_L) * D_L * D_S / D_LS  [Mpc]
    approx_level=1: 해석 근사
    approx_level=2: H₀=70 기준 스케일 팩터 룩업 (더 빠름, 오차 < 5%)
    """
    if _HAS_DISTANCES and approx_level == 0:
        D_L = _add_ext(z_lens, H0)
        D_S = _add_ext(z_source, H0)
        D_LS = _add_ext(z_source, H0, z_lens=z_lens)
    else:
        D_L = _angular_diameter_distance_approx(z_lens, H0)
        D_S = _angular_diameter_distance_approx(z_source, H0)
        # D_LS: 렌즈~소스 사이 (간소화: 비율로 근사)
        D_LS = _angular_diameter_distance_approx(z_source - z_lens, H0) * (
            1.0 + z_lens
        ) / (1.0 + z_source)

    if approx_level == 2:
        # H₀=70 기준으로 미리 계산 후 선형 스케일
        scale_ref = _d_delta_t(70.0, z_lens, z_source, approx_level=1)
        return scale_ref * 70.0 / H0

    return (1.0 + z_lens) * D_L * D_S / D_LS


def _h0_from_dt(dt_obs: np.ndarray, fermat_potential: np.ndarray,
                z_lens: float, z_source: float,
                approx_level: int, cosmology_kwargs: dict) -> float:
    """비선형 방정식  Δt = (1+z_L)/c * D_Δt(H₀) * Δφ  을 H₀에 대해 풀기."""
    ph = _phys()
    c = ph["c_km_s"]
    days_s = ph["days_s"]
    Mpc_km = ph["Mpc_km"]

    def residuals(h0_vec: np.ndarray) -> np.ndarray:
        H0 = float(h0_vec[0])
        if H0 <= 0:
            return np.full(len(dt_obs), 1e6)
        ddt = _d_delta_t(H0, z_lens, z_source, approx_level)
        # Δt_model [days]
        dt_model = ((1.0 + z_lens) * ddt * Mpc_km / c) * fermat_potential / days_s
        return dt_obs - dt_model

    result = least_squares(
        residuals,
        x0=[70.0],
        bounds=([50.0], [90.0]),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
    )
    return float(result.x[0])


def invert_h0(
    dt_obs: np.ndarray,
    fermat_potential: np.ndarray,
    z_lens: float,
    z_source: float,
    cosmology_kwargs: dict | None = None,
    approx_level: int = 1,
    n_bootstrap: int = 200,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """
    Mode 1 — H₀ 역산.

    Parameters
    ----------
    dt_obs : ndarray, shape [n_pairs]
        관측 시간 지연 [days]
    fermat_potential : ndarray, shape [n_pairs]
        페르마 포텐셜 차이 Δφ [arcsec²]
    z_lens : float
        렌즈 적색편이
    z_source : float
        소스 적색편이
    cosmology_kwargs : dict, optional
        추가 우주론 파라미터 (현재 미사용, 인터페이스 호환용)
    approx_level : int
        1=FAST(해석 근사), 2=TURBO(스케일 근사). 0=EXACT는 distances.py 필요.
    n_bootstrap : int
        불확도 추정 부트스트랩 반복 횟수
    rng_seed : int
        재현성 시드

    Returns
    -------
    dict with keys:
        H0              : float  [km/s/Mpc]
        H0_uncertainty  : float  [km/s/Mpc] (1σ bootstrap)
        approx_level    : int
        n_pairs         : int
    """
    dt_obs = np.asarray(dt_obs, dtype=float)
    fermat_potential = np.asarray(fermat_potential, dtype=float)

    if cosmology_kwargs is None:
        cosmology_kwargs = {}

    assert dt_obs.shape == fermat_potential.shape, "dt_obs / fermat_potential shape 불일치"
    assert z_source > z_lens + 0.05, f"z_source({z_source}) <= z_lens({z_lens})+0.05"

    h0_best = _h0_from_dt(dt_obs, fermat_potential, z_lens, z_source,
                          approx_level, cosmology_kwargs)

    # 부트스트랩 불확도
    rng = np.random.default_rng(rng_seed)
    n = len(dt_obs)
    h0_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            h0_samples[i] = _h0_from_dt(
                dt_obs[idx], fermat_potential[idx],
                z_lens, z_source, approx_level, cosmology_kwargs,
            )
        except Exception:
            h0_samples[i] = h0_best

    h0_unc = float(np.std(h0_samples))

    return {
        "H0": h0_best,
        "H0_uncertainty": h0_unc,
        "approx_level": approx_level,
        "n_pairs": int(n),
    }
