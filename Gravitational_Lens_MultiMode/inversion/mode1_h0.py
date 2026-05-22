"""
Mode 1 — H₀ 역산 솔버.

입력: 관측 시간 지연 Δt_obs [days], 페르마 포텐셜 Δφ [rad²], 적색편이.
출력: H₀ [km/s/Mpc] + 불확도.

근사 레벨(approx_level)은 Mode와 직교 축.
  level 0 (EXACT): core.physics.distances 적분 거리 — flat ΛCDM
  level 1 (FAST):  해석 근사 — flat ΛCDM, 선형 팽창 근사
  level 2 (TURBO): level 1 + D_Δt를 H₀=70 기준 룩업 스케일로 단순화

단위: Δt [days], Δφ [rad²], H₀ [km/s/Mpc], D [Mpc], c [km/s].
SIE 표준 근사 가정: 단일 렌즈 평면, κ_ext=0, smooth SIE mass profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from core.physics.distances import (
    angular_diameter_distance,
    angular_diameter_distance_between,
)

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
def _angular_diameter_distance_approx(z: float, H0: float) -> float:
    """
    평탄 ΛCDM 해석 근사 (Ω_m=0.3, Ω_Λ=0.7).
    단위: return [Mpc], z dimensionless, H0 [km/s/Mpc].
    SIE 표준 근사 가정: Mode 1의 SIE Δφ와 결합되는 보조 거리 근사.
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
    Reduced time-delay distance D_L * D_S / D_LS [Mpc].

    Units: H0 [km/s/Mpc], redshifts dimensionless, return [Mpc].
    SIE 표준 근사 가정: fitted SIE Fermat-potential difference Δφ is in
    [rad²], and callers multiply this reduced distance by ``(1 + z_lens)``.
    approx_level=0 uses ``core.physics.distances`` integral distances,
    approx_level=1 uses the local analytic distance approximation, and
    approx_level=2 reuses level 1 at H0=70 with linear H0 scaling.
    """
    if approx_level == 0:
        cosmo = {"H0": float(H0)}
        D_L = angular_diameter_distance(z_lens, cosmo)
        D_S = angular_diameter_distance(z_source, cosmo)
        D_LS = angular_diameter_distance_between(z_lens, z_source, cosmo)
    else:
        D_L = _angular_diameter_distance_approx(z_lens, H0)
        D_S = _angular_diameter_distance_approx(z_source, H0)
        # D_LS from approximate comoving distances. EXACT(level 0) uses
        # core.physics.distances.angular_diameter_distance_between instead.
        D_C_L = D_L * (1.0 + z_lens)
        D_C_S = D_S * (1.0 + z_source)
        D_LS = (D_C_S - D_C_L) / (1.0 + z_source)

    if approx_level == 2:
        # H₀=70 기준으로 미리 계산 후 선형 스케일
        scale_ref = _d_delta_t(70.0, z_lens, z_source, approx_level=1)
        return scale_ref * 70.0 / H0

    return D_L * D_S / D_LS


def _h0_from_dt(dt_obs: np.ndarray, fermat_potential: np.ndarray,
                z_lens: float, z_source: float,
                approx_level: int, cosmology_kwargs: dict) -> float:
    """Solve Δt = (1+z_L) D_red(H0) Δφ / c for H0.

    Units: ``dt_obs`` [days], ``fermat_potential`` [rad²], reduced distance
    [Mpc], H0 [km/s/Mpc]. SIE 표준 근사 가정: Δφ comes from the fixed SIE
    standard approximation with single plane and κ_ext=0.
    """
    ph = _phys()
    c = ph["c_km_s"]
    days_s = ph["days_s"]
    Mpc_km = ph["Mpc_km"]

    def residuals(h0_vec: np.ndarray) -> np.ndarray:
        H0 = float(h0_vec[0])
        if H0 <= 0:
            return np.full(len(dt_obs), 1e6)
        d_reduced = _d_delta_t(H0, z_lens, z_source, approx_level)
        # Δt_model [days]
        dt_model = ((1.0 + z_lens) * d_reduced * Mpc_km / c) * fermat_potential / days_s
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
    fermat_potential_rad2: np.ndarray,
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
    fermat_potential_rad2 : ndarray, shape [n_pairs]
        페르마 포텐셜 차이 Δφ [rad²]. arcsec² 입력은 허용하지 않는다.
    z_lens : float
        렌즈 적색편이
    z_source : float
        소스 적색편이
    cosmology_kwargs : dict, optional
        추가 우주론 파라미터 (현재 미사용, 인터페이스 호환용)
    approx_level : int
        0=EXACT(core.physics.distances), 1=FAST(해석 근사),
        2=TURBO(스케일 근사).
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

    SIE 표준 근사 가정
    ----------------
    Δφ는 고정 SIE 표준 근사에서 산출된 [rad²] 값이며, 단일 평면,
    κ_ext=0, smooth SIE mass profile을 가정한다.
    """
    dt_obs = np.asarray(dt_obs, dtype=float)
    fermat_potential_rad2 = np.asarray(fermat_potential_rad2, dtype=float)

    if cosmology_kwargs is None:
        cosmology_kwargs = {}

    assert dt_obs.shape == fermat_potential_rad2.shape, "dt_obs / fermat_potential_rad2 shape 불일치"
    assert z_source > z_lens + 0.05, f"z_source({z_source}) <= z_lens({z_lens})+0.05"
    if approx_level not in (0, 1, 2):
        raise ValueError("approx_level must be one of 0, 1, or 2")
    if not np.isfinite(dt_obs).all() or not np.isfinite(fermat_potential_rad2).all():
        raise ValueError("dt_obs and fermat_potential_rad2 must be finite")
    if np.any(dt_obs <= 0):
        raise ValueError("dt_obs must be positive [days]")
    if np.any(fermat_potential_rad2 <= 0):
        raise ValueError("fermat_potential_rad2 must be positive [rad²]")

    h0_best = _h0_from_dt(dt_obs, fermat_potential_rad2, z_lens, z_source,
                          approx_level, cosmology_kwargs)

    # 부트스트랩 불확도
    rng = np.random.default_rng(rng_seed)
    n = len(dt_obs)
    n_boot = max(int(n_bootstrap), 0)
    h0_samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            h0_samples[i] = _h0_from_dt(
                dt_obs[idx], fermat_potential_rad2[idx],
                z_lens, z_source, approx_level, cosmology_kwargs,
            )
        except Exception:
            h0_samples[i] = h0_best

    h0_unc = 0.0 if n_boot <= 0 else float(np.std(h0_samples))

    return {
        "H0": h0_best,
        "H0_uncertainty": h0_unc,
        "approx_level": approx_level,
        "n_pairs": int(n),
    }
