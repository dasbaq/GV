"""
COSMOGRAIL 어댑터 — Mode 1 학습/검증용.

COSMOGRAIL 웹사이트: https://www.cosmograil.org  (다운로드 코드 금지, URL만)
형식: CSV  (lens_name, dt_AB [days], dt_AB_err [days], H0_ref, ...)

has_ground_truth = True (공표 Δt + H0 기준값 있음)
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd


EXPECTED_COLS = ["lens_name", "dt_AB", "dt_AB_err"]


def load_cosmograil_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parameters
    ----------
    csv_path : COSMOGRAIL 포맷 CSV 경로

    Returns
    -------
    DataFrame with columns: lens_name, dt_AB, dt_AB_err, (optionally H0_ref)
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"COSMOGRAIL CSV에 필수 컬럼 없음: {missing}")
    return df


def cosmograil_to_h5(
    csv_path: Union[str, Path],
    out_path: Union[str, Path],
    image_size: int = 128,
    max_lc_len: int = 1024,
    mode2_max_dm_dim: int = 4,
) -> Path:
    """
    COSMOGRAIL CSV → ARCHITECTURE.md 스키마 HDF5 변환.

    has_ground_truth 메타데이터 기록.
    광도곡선 데이터가 CSV에 없는 경우 zeros로 채움 (shape 유지).
    """
    df = load_cosmograil_csv(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(df)

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["source"]           = "COSMOGRAIL"
        meta.attrs["n_systems"]        = n
        meta.attrs["has_ground_truth"] = True
        meta.attrs["approx_level"]     = -1   # 실측 = 근사 없음

        params_g = f.create_group("params")
        params_g.create_dataset("H0",           data=np.full(n, 70.0, np.float32))
        params_g.create_dataset("z_lens",       data=np.zeros(n, np.float32))
        params_g.create_dataset("z_source",     data=np.zeros(n, np.float32))
        params_g.create_dataset("sigma_v",      data=np.full(n, 250.0, np.float32))
        params_g.create_dataset("M200",         data=np.full(n, 1e13, np.float32))
        params_g.create_dataset("concentration",data=np.full(n, 7.0,  np.float32))
        lens_models = np.array([b"SIS"] * n)
        params_g.create_dataset("lens_model",   data=lens_models)

        lc_g = f.create_group("light_curves")
        lc_g.create_dataset("F_joint",    data=np.zeros((n, max_lc_len), np.float32))
        lc_g.create_dataset("sigma_noise",data=np.zeros((n, max_lc_len), np.float32))
        lc_g.create_dataset("t_obs",      data=np.zeros((n, max_lc_len), np.float32))
        lc_g.create_dataset("n_epochs",   data=np.zeros(n, np.int32))

        img_g = f.create_group("images")
        img_g.create_dataset("I_obs",      data=np.zeros((n, image_size, image_size), np.float32))
        img_g.create_dataset("S_true",     data=np.zeros((n, image_size, image_size), np.float32))
        img_g.create_dataset("psf",        data=np.zeros((n, 11, 11), np.float32))
        img_g.create_dataset("pixel_scale",data=np.full(n, 0.05, np.float32))

        tv_g = f.create_group("true_values")
        tv_g.create_dataset("dt_true",       data=df["dt_AB"].values.astype(np.float32))
        tv_g.create_dataset("mu_true",       data=np.ones(n, np.float32))
        tv_g.create_dataset("theta_E",       data=np.zeros(n, np.float32))
        h0_ref = df["H0_ref"].values.astype(np.float32) if "H0_ref" in df else np.full(n, 70.0, np.float32)
        tv_g.create_dataset("H0_true",       data=h0_ref)
        tv_g.create_dataset("dm_params_true",data=np.zeros((n, mode2_max_dm_dim), np.float32))
        tv_g.create_dataset("dm_dim",        data=np.ones(n, np.int32))
        tv_g.create_dataset("D_delta_t",     data=np.zeros(n, np.float32))

        rp_g = f.create_group("ray_paths")
        rp_g.create_dataset("theta_1",         data=np.zeros((n, 2), np.float32))
        rp_g.create_dataset("theta_2",         data=np.zeros((n, 2), np.float32))
        rp_g.create_dataset("fermat_potential", data=np.zeros(n, np.float32))

        se_g = f.create_group("simplification_errors")
        se_g.create_dataset("mode1_H0_error",       data=np.zeros(n, np.float32))
        se_g.create_dataset("mode2_dm_error",       data=np.zeros((n, mode2_max_dm_dim), np.float32))
        se_g.create_dataset("mode3_source_residual",data=np.zeros((n, image_size, image_size), np.float32))
        se_g.create_dataset("approx_level_used",    data=np.full(n, -1, np.int32))

    return out_path
