"""
Mode 2 — 암흑물질 분포 파라미터 역산 솔버.

입력: 관측 (Δt, θ_i, μ_i), 고정 H₀ / 적색편이, 렌즈 모델 선택.
출력: DM 파라미터 벡터 + 불확도.

렌즈 모델별 파라미터 차원:
  SIS  : [σ_v]                  (1)
  NFW  : [log10_M200, c_nfw]    (2)
  SIE  : [σ_v, q]               (2)
  POINT: [log10_M]              (1)

approx_level (Mode와 직교 축):
  1=FAST : 해석 렌즈 방정식 (SIS는 완전 해석, NFW는 근사 프로파일)
  2=TURBO: 모든 모델 SIS로 유효 σ_v 근사

단위: θ [arcsec], μ [무차원], Δt [days], σ_v [km/s], M [M☉]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# 물리 상수 로드                                                                #
# --------------------------------------------------------------------------- #
def _load_physics() -> dict:
    import yaml
    cfg_path = Path(__file__).parent.parent / "config" / "physics.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_PHYS: dict = {}

def _phys() -> dict:
    global _PHYS
    if not _PHYS:
        _PHYS = _load_physics()
    return _PHYS


# --------------------------------------------------------------------------- #
# 렌즈 모델 — 관측량 예측                                                       #
# --------------------------------------------------------------------------- #
LENS_PARAM_DIM = {"SIS": 1, "NFW": 2, "SIE": 2, "POINT": 1}
LENS_PARAM_BOUNDS = {
    "SIS":   [(150.0, 350.0)],
    "NFW":   [(12.0, 14.0), (3.0, 15.0)],
    "SIE":   [(150.0, 350.0), (0.2, 1.0)],
    "POINT": [(8.0, 12.0)],
}


def _sis_einstein_radius(sigma_v: float, z_lens: float, z_source: float,
                         H0: float) -> float:
    """θ_E [arcsec] for SIS.  approx_level=1"""
    ph = _phys()
    c = ph["c_km_s"]
    Mpc_km = ph["Mpc_km"]
    arcsec = ph["arcsec_rad"]
    # θ_E = 4π(σ_v/c)² * D_LS/D_S
    # D 비율만 필요 — approx: D_LS/D_S ≈ (z_S - z_L)/z_S (선형 근사)
    Dls_Ds = (z_source - z_lens) / z_source
    theta_E = 4 * np.pi * (sigma_v / c) ** 2 * Dls_Ds / arcsec
    return theta_E


def _predict_observables_sis(params: np.ndarray, theta_obs: np.ndarray,
                              z_lens: float, z_source: float,
                              H0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    SIS 렌즈 모델로 (θ_pred, μ_pred) 계산.
    params = [sigma_v]
    theta_obs shape [n_images, 2]
    반환: theta_pred [n_images, 2], mu_pred [n_images]
    """
    sigma_v = float(params[0])
    theta_E = _sis_einstein_radius(sigma_v, z_lens, z_source, H0)

    r = np.linalg.norm(theta_obs, axis=1)        # [n_images]
    # SIS magnification: μ = |r / (r - θ_E)|
    eps = 1e-9
    mu_pred = np.abs(r / (r - theta_E + eps))
    # SIS 상 위치: 여기선 관측 위치 자체를 theta_pred로 (역산 대상은 모델 파라미터)
    theta_pred = theta_obs.copy()
    return theta_pred, mu_pred


def _predict_observables_nfw(params: np.ndarray, theta_obs: np.ndarray,
                              z_lens: float, z_source: float,
                              H0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    NFW 근사 — Bartelmann (1996) convergence 기반 magnification 근사.
    params = [log10_M200, c_nfw]
    """
    log10_M200, c_nfw = float(params[0]), float(params[1])
    M200 = 10 ** log10_M200  # M_sun

    ph = _phys()
    G = ph["G"]
    Mpc_m = ph["Mpc_m"]
    M_sun = ph["M_sun"]
    arcsec = ph["arcsec_rad"]
    c_km = ph["c_km_s"]

    # 스케일 반경 r_s 근사 [Mpc]
    rho_crit = 3 * (H0 * 1e3 / Mpc_m) ** 2 / (8 * np.pi * G) / (M_sun / Mpc_m ** 3)
    r200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)  # Mpc
    r_s = r200 / c_nfw

    # Einstein 반경 근사 (NFW → SIS 유효 σ_v)
    sigma_eff = np.sqrt(G * M200 * M_sun / (2 * r200 * Mpc_m)) / 1e3  # km/s
    sigma_eff = np.clip(sigma_eff, 50, 500)
    theta_E = _sis_einstein_radius(sigma_eff, z_lens, z_source, H0)

    r = np.linalg.norm(theta_obs, axis=1)
    eps = 1e-9
    mu_pred = np.abs(r / (r - theta_E + eps))
    return theta_obs.copy(), mu_pred


def _predict_observables_sie(params: np.ndarray, theta_obs: np.ndarray,
                              z_lens: float, z_source: float,
                              H0: float) -> tuple[np.ndarray, np.ndarray]:
    """SIE = SIS with axis ratio q.  params = [sigma_v, q]"""
    sigma_v, q = float(params[0]), float(params[1])
    q = np.clip(q, 0.1, 1.0)
    # SIE는 SIS에 비해 효과적 θ_E를 q로 조정
    theta_E = _sis_einstein_radius(sigma_v, z_lens, z_source, H0) * np.sqrt(q)
    r = np.linalg.norm(theta_obs, axis=1)
    eps = 1e-9
    mu_pred = np.abs(r / (r - theta_E + eps))
    return theta_obs.copy(), mu_pred


def _predict_observables_point(params: np.ndarray, theta_obs: np.ndarray,
                               z_lens: float, z_source: float,
                               H0: float) -> tuple[np.ndarray, np.ndarray]:
    """점질량 렌즈.  params = [log10_M]"""
    ph = _phys()
    G = ph["G"]
    M_sun = ph["M_sun"]
    Mpc_m = ph["Mpc_m"]
    arcsec = ph["arcsec_rad"]
    c_m = ph["c_m_s"]

    log10_M = float(params[0])
    M = 10 ** log10_M * M_sun

    # D_LS/D_S 근사
    Dls_Ds = (z_source - z_lens) / z_source
    D_L = (c_m / (H0 * 1e3 / Mpc_m)) * z_lens / (1 + z_lens)
    theta_E_rad = np.sqrt(4 * G * M / c_m ** 2 * Dls_Ds / D_L)
    theta_E = theta_E_rad / arcsec

    r = np.linalg.norm(theta_obs, axis=1)
    u = r / (theta_E + 1e-9)
    mu_pred = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4) + 1e-9)
    return theta_obs.copy(), mu_pred


_PREDICT_FN = {
    "SIS": _predict_observables_sis,
    "NFW": _predict_observables_nfw,
    "SIE": _predict_observables_sie,
    "POINT": _predict_observables_point,
}


def _loss(params: np.ndarray, theta_obs: np.ndarray, mu_obs: np.ndarray,
          z_lens: float, z_source: float, H0: float,
          lens_model: str, approx_level: int) -> float:
    """관측량 residual 합 (SIS 근사로 fallback — approx_level=2)."""
    if approx_level == 2:
        effective_model = "SIS"
        if lens_model == "SIS":
            effective_params = params
        else:
            # 첫 번째 파라미터를 σ_v에 해당하는 유효값으로 맵핑
            effective_params = np.array([200.0 + params[0] * 10])
    else:
        effective_model = lens_model
        effective_params = params

    fn = _PREDICT_FN[effective_model]
    _, mu_pred = fn(effective_params, theta_obs, z_lens, z_source, H0)

    residual_mu = (mu_pred - mu_obs) / (np.abs(mu_obs) + 1e-6)
    return float(np.sum(residual_mu ** 2))


def invert_dm(
    dt_obs: np.ndarray,
    theta_obs: np.ndarray,
    mu_obs: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
    lens_model: str = "SIS",
    approx_level: int = 1,
    n_bootstrap: int = 200,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """
    Mode 2 — 암흑물질 분포 파라미터 역산.

    Parameters
    ----------
    dt_obs : ndarray [n_pairs]
        관측 시간 지연 [days] (현재 residual 계산에 보조 사용)
    theta_obs : ndarray [n_images, 2]
        상의 위치 [arcsec]
    mu_obs : ndarray [n_images]
        관측 배율
    H0 : float
        고정 허블 상수 [km/s/Mpc]
    z_lens, z_source : float
    lens_model : str
        "SIS" | "NFW" | "SIE" | "POINT"
    approx_level : int
        1=FAST, 2=TURBO
    n_bootstrap : int
    rng_seed : int

    Returns
    -------
    dict with keys:
        dm_params        : ndarray  [K]
        dm_uncertainty   : ndarray  [K]
        lens_model       : str
        approx_level     : int
        param_names      : list[str]
    """
    dt_obs = np.asarray(dt_obs, dtype=float)
    theta_obs = np.asarray(theta_obs, dtype=float)
    mu_obs = np.asarray(mu_obs, dtype=float)

    assert lens_model in LENS_PARAM_DIM, f"알 수 없는 lens_model: {lens_model}"
    assert z_source > z_lens + 0.05

    bounds_list = LENS_PARAM_BOUNDS[lens_model]
    lo = np.array([b[0] for b in bounds_list])
    hi = np.array([b[1] for b in bounds_list])

    # 멀티스타트 최적화 — 10개 초기점으로 전역 최솟값 탐색
    rng_init = np.random.default_rng(rng_seed)
    n_starts = 10
    starts   = [lo + (hi - lo) * rng_init.uniform(size=len(lo))
                for _ in range(n_starts)]
    starts.append((lo + hi) / 2.0)  # 중간값 포함

    best_fun   = np.inf
    params_best = (lo + hi) / 2.0
    for x0 in starts:
        try:
            res = minimize(
                _loss, x0=x0,
                args=(theta_obs, mu_obs, z_lens, z_source, H0, lens_model, approx_level),
                method="L-BFGS-B",
                bounds=bounds_list,
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
            )
            if res.fun < best_fun:
                best_fun    = res.fun
                params_best = res.x
        except Exception:
            continue

    # 부트스트랩 불확도
    rng = np.random.default_rng(rng_seed)
    n = len(mu_obs)
    K = LENS_PARAM_DIM[lens_model]
    samples = np.empty((n_bootstrap, K))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            res = minimize(
                _loss,
                x0=params_best,
                args=(theta_obs[idx], mu_obs[idx],
                      z_lens, z_source, H0, lens_model, approx_level),
                method="L-BFGS-B",
                bounds=bounds_list,
                options={"maxiter": 100},
            )
            samples[i] = res.x
        except Exception:
            samples[i] = params_best

    unc = np.std(samples, axis=0)

    param_names_map = {
        "SIS": ["sigma_v"],
        "NFW": ["log10_M200", "c_nfw"],
        "SIE": ["sigma_v", "q"],
        "POINT": ["log10_M"],
    }

    return {
        "dm_params": params_best,
        "dm_uncertainty": unc,
        "lens_model": lens_model,
        "approx_level": approx_level,
        "param_names": param_names_map[lens_model],
    }
