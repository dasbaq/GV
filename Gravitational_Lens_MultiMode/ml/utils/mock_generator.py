"""
Mock HDF5 생성기.

ARCHITECTURE.md HDF5 스키마와 동일한 구조의 toy 데이터(1024 샘플)를
지정 경로에 생성.  경로는 호출자가 제공 — 하드코딩 없음.

Mode 3용 이미지 그룹(images/)과 mode3_source_residual은 삭제됨
(DECISIONS.md [2026-05-25] 참조).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np


def create_mock_h5(
    out_path: Union[str, Path],
    n_systems: int = 1024,
    max_epochs: int = 300,
    mode2_max_dm_dim: int = 4,
    seed: int = 42,
) -> Path:
    """
    ARCHITECTURE.md 스키마와 동일한 구조의 mock HDF5 생성.

    Parameters
    ----------
    out_path      : 저장 경로 (호출자 제공, 하드코딩 금지)
    n_systems     : 샘플 수
    max_epochs    : 광도곡선 최대 길이
    mode2_max_dm_dim : DM 파라미터 패딩 차원
    seed          : 난수 시드

    Returns
    -------
    Path  저장된 파일 경로
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(out_path, "w") as f:
        # ---- metadata ----
        meta = f.create_group("metadata")
        meta.attrs["created_at"] = "mock"
        meta.attrs["n_systems"] = n_systems
        meta.attrs["approx_level"] = 1
        meta.attrs["git_commit"] = "mock"
        meta.attrs["random_seed"] = seed

        # ---- params ----
        H0_arr     = rng.uniform(60.0, 80.0, n_systems).astype(np.float32)
        z_lens_arr = rng.uniform(0.1, 1.0, n_systems).astype(np.float32)
        z_src_arr  = z_lens_arr + rng.uniform(0.3, 2.0, n_systems).astype(np.float32)
        z_src_arr  = np.clip(z_src_arr, 0.5, 3.5).astype(np.float32)
        sigma_v_arr = rng.uniform(150, 350, n_systems).astype(np.float32)
        M200_arr    = 10 ** rng.uniform(12, 14, n_systems)
        conc_arr    = rng.uniform(3, 15, n_systems).astype(np.float32)

        params_g = f.create_group("params")
        params_g.create_dataset("H0",           data=H0_arr)
        params_g.create_dataset("z_lens",       data=z_lens_arr)
        params_g.create_dataset("z_source",     data=z_src_arr)
        params_g.create_dataset("sigma_v",      data=sigma_v_arr)
        params_g.create_dataset("M200",         data=M200_arr.astype(np.float32))
        params_g.create_dataset("concentration",data=conc_arr)
        # lens_model: 모든 mock은 SIS
        lens_models = np.array([b"SIS"] * n_systems)
        params_g.create_dataset("lens_model",   data=lens_models)

        # ---- light_curves ----
        # 각 시스템 n_epochs는 100~max_epochs
        n_epochs_arr = rng.integers(100, max_epochs + 1, n_systems)
        F_joint    = np.zeros((n_systems, max_epochs), dtype=np.float32)
        sigma_noise= np.zeros((n_systems, max_epochs), dtype=np.float32)
        t_obs_arr  = np.zeros((n_systems, max_epochs), dtype=np.float32)

        for i in range(n_systems):
            ne = int(n_epochs_arr[i])
            t  = np.linspace(0, 1000, ne).astype(np.float32)
            # DRW 근사: AR(1) 과정
            flux = np.zeros(ne, dtype=np.float32)
            flux[0] = float(rng.normal(20.0, 0.5))
            tau, sf = float(rng.uniform(100, 500)), float(rng.uniform(0.1, 0.5))
            dt_step = 1000.0 / (ne - 1) if ne > 1 else 1.0
            for j in range(1, ne):
                phi = np.exp(-dt_step / tau)
                flux[j] = phi * flux[j-1] + np.sqrt(1 - phi**2) * float(rng.normal(0, sf))
            noise = float(rng.uniform(0.01, 0.05))
            F_joint[i, :ne] = flux + rng.normal(0, noise, ne).astype(np.float32)
            sigma_noise[i, :ne] = noise
            t_obs_arr[i, :ne] = t

        lc_g = f.create_group("light_curves")
        lc_g.create_dataset("F_joint",     data=F_joint)
        lc_g.create_dataset("sigma_noise", data=sigma_noise)
        lc_g.create_dataset("t_obs",       data=t_obs_arr)
        lc_g.create_dataset("n_epochs",    data=n_epochs_arr.astype(np.int32))

        # ---- true_values ----
        # Mode 1 라벨: H0_true = H0 + 작은 노이즈
        H0_true = H0_arr + rng.normal(0, 0.5, n_systems).astype(np.float32)
        dt_true = rng.uniform(5, 100, n_systems).astype(np.float32)
        mu_true = rng.uniform(0.5, 5.0, n_systems).astype(np.float32)
        theta_E_true = sigma_v_arr / 300.0 * 1.0
        D_delt = rng.uniform(500, 3000, n_systems).astype(np.float32)
        # Mode 2 라벨: [sigma_v] padded to mode2_max_dm_dim
        dm_params_true = np.zeros((n_systems, mode2_max_dm_dim), dtype=np.float32)
        dm_params_true[:, 0] = sigma_v_arr
        dm_dim_arr = np.ones(n_systems, dtype=np.int32)  # SIS → dim=1

        tv_g = f.create_group("true_values")
        tv_g.create_dataset("dt_true",       data=dt_true)
        tv_g.create_dataset("mu_true",       data=mu_true)
        tv_g.create_dataset("theta_E",       data=theta_E_true.astype(np.float32))
        tv_g.create_dataset("H0_true",       data=H0_true)
        tv_g.create_dataset("dm_params_true",data=dm_params_true)
        tv_g.create_dataset("dm_dim",        data=dm_dim_arr)
        tv_g.create_dataset("D_delta_t",     data=D_delt)

        # ---- ray_paths ----
        theta_1 = rng.uniform(-2, 2, (n_systems, 2)).astype(np.float32)
        theta_2 = rng.uniform(-2, 2, (n_systems, 2)).astype(np.float32)
        fermat  = rng.uniform(0.1, 5.0, n_systems).astype(np.float32)

        rp_g = f.create_group("ray_paths")
        rp_g.create_dataset("theta_1",         data=theta_1)
        rp_g.create_dataset("theta_2",         data=theta_2)
        rp_g.create_dataset("fermat_potential", data=fermat)

        # ---- simplification_errors ----
        m1_err = rng.normal(0, 1.0, n_systems).astype(np.float32)
        m2_err = np.zeros((n_systems, mode2_max_dm_dim), dtype=np.float32)
        m2_err[:, 0] = rng.normal(0, 5.0, n_systems).astype(np.float32)
        al_used = np.ones(n_systems, dtype=np.int32)

        se_g = f.create_group("simplification_errors")
        se_g.create_dataset("mode1_H0_error",        data=m1_err)
        se_g.create_dataset("mode2_dm_error",        data=m2_err)
        se_g.create_dataset("approx_level_used",     data=al_used)

    return out_path
