"""
Mode 3 — 기존 source 복원 솔버 호출 wrapper.

기존 구현: src_py/modes/smbh.py (SMBHMode — 수정 금지)
인터페이스 변환:
  - 외부 호출: reconstruct_source(image_obs, psf, pixel_scale, lens_params)
  - 내부 위임: SMBHMode.load_data() + preprocess_features() + sklearn predict

smbh.py의 실제 인터페이스:
  입력: DataFrame with columns [final_x, final_y, time_delay, ...]
  출력: (X, Y) where Y = [init_x, init_y]
  → wrapper가 이미지 → DataFrame 변환, 출력을 2D 이미지로 역변환.

주의: 기존 솔버는 sklearn 모델을 내장하지 않음 — preprocess_features()는
  (X, Y) 쌍을 반환하며 학습은 외부(ml_pipeline.py)에서 수행.
  따라서 이 wrapper는 "이미지 → 소스 위치 예측" 기능을 근사 재구성으로 제공.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# 기존 Mode 3 솔버 임포트 (수정 금지)
_SRC_PY = Path(__file__).parent.parent / "src_py"
sys.path.insert(0, str(_SRC_PY))

try:
    from modes.smbh import SMBHMode as _SMBHMode
    _HAS_SMBH = True
except ImportError as e:
    _HAS_SMBH = False
    _SMBH_IMPORT_ERR = str(e)


def _image_to_dataframe(image_obs: np.ndarray, pixel_scale: float,
                        lens_params: dict) -> "pd.DataFrame":
    """
    [H, W] 이미지를 smbh.py가 기대하는 DataFrame 형식으로 변환.
    각 픽셀 → (final_x, final_y, time_delay, ...) 행.
    """
    import pandas as pd

    H, W = image_obs.shape
    ys, xs = np.mgrid[0:H, 0:W]
    # 픽셀 좌표 → 물리 좌표 [arcsec]
    cx, cy = W / 2.0, H / 2.0
    x_arcsec = (xs - cx) * pixel_scale
    y_arcsec = (ys - cy) * pixel_scale

    flux = image_obs.ravel()
    # time_delay를 플럭스로 근사 (wrapper 단에서 의미있는 대응 없음 — 스케일만 맞춤)
    time_delay_proxy = np.abs(flux) / (np.abs(flux).max() + 1e-9) * lens_params.get("dt_scale", 10.0)

    df = pd.DataFrame({
        "final_x":    x_arcsec.ravel(),
        "final_y":    y_arcsec.ravel(),
        "time_delay": time_delay_proxy,
        "init_x":     np.zeros(H * W),   # 더미 라벨 (preprocess_features 내부 필요)
        "init_y":     np.zeros(H * W),
    })
    return df


def _direct_flux_backproject(
    image_obs: np.ndarray,
    src_x: np.ndarray,   # [H*W] 소스 평면 x 좌표 [arcsec]
    src_y: np.ndarray,   # [H*W] 소스 평면 y 좌표 [arcsec]
    H: int, W: int,
    pixel_scale: float,
) -> np.ndarray:
    """
    관측 픽셀 플럭스를 소스 평면에 직접 투영 (bilinear 누적).
    밀도 히스토그램 대신 flux를 보존하여 PSNR 향상.
    """
    cx, cy = W / 2.0, H / 2.0
    # 소스 평면 픽셀 좌표 (실수)
    sx_f = src_x / pixel_scale + cx   # [H*W]
    sy_f = src_y / pixel_scale + cy

    flux = image_obs.ravel()          # [H*W]
    out  = np.zeros((H, W), dtype=float)
    cnt  = np.zeros((H, W), dtype=float)

    # 벡터화 bilinear splat
    x0 = np.floor(sx_f).astype(int)
    y0 = np.floor(sy_f).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    dx = sx_f - x0
    dy = sy_f - y0

    for (xi, yi, wx, wy) in [
        (x0, y0, (1-dx)*(1-dy), None),
        (x1, y0, dx*(1-dy),     None),
        (x0, y1, (1-dx)*dy,     None),
        (x1, y1, dx*dy,         None),
    ]:
        mask = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.add.at(out, (yi[mask], xi[mask]), (flux * wx)[mask])
        np.add.at(cnt, (yi[mask], xi[mask]), wx[mask])

    # 커버리지가 있는 픽셀만 정규화
    valid = cnt > 1e-9
    out[valid] /= cnt[valid]

    return out.astype(np.float32)


def reconstruct_source(
    image_obs: np.ndarray,
    psf: np.ndarray,
    pixel_scale: float,
    lens_params: dict,
    approx_level: int = 1,
) -> dict[str, Any]:
    """
    Mode 3 — 기존 소스 복원 솔버(smbh.py) 호출.

    Parameters
    ----------
    image_obs : ndarray [H, W]
        관측(렌즈된) 이미지
    psf : ndarray [ph, pw]
        PSF 커널 (현재 버전에서 deconvolution 미적용 — 메타에 기록)
    pixel_scale : float
        [arcsec/pixel]
    lens_params : dict
        렌즈 파라미터 (dt_scale 등 wrapper 힌트 포함 가능)
    approx_level : int
        1=FAST, 2=TURBO (현재 wrapper에서는 동일 경로)

    Returns
    -------
    dict with keys:
        source       : ndarray [H, W]  복원된 소스 이미지
        approx_level : int
        solver_meta  : dict            solverr 메타 정보
    """
    if not _HAS_SMBH:
        raise ImportError(
            f"기존 Mode 3 솔버(smbh.py) 임포트 실패: {_SMBH_IMPORT_ERR}"
        )

    image_obs = np.asarray(image_obs, dtype=float)
    assert image_obs.ndim == 2, "image_obs must be 2D [H, W]"
    H, W = image_obs.shape

    # 1. 이미지 → DataFrame
    df = _image_to_dataframe(image_obs, pixel_scale, lens_params)

    # 2. 기존 솔버 호출 (수정 금지 — load_data + preprocess_features만)
    solver = _SMBHMode()
    solver.load_data(df)
    X, Y = solver.preprocess_features()   # X: features, Y: (init_x, init_y) 더미

    # 3. X에서 θ_E 추정 (smbh.py einstein_approx 컬럼 활용)
    if hasattr(X, "values"):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    final_x = df["final_x"].values
    final_y = df["final_y"].values
    r_vals  = np.sqrt(final_x ** 2 + final_y ** 2)

    # einstein_approx = sqrt(time_delay / r) → 이를 flux 가중 평균으로 θ_E 추정
    ea_col = X_arr[:, -1]   # einstein_approx 컬럼
    flux   = image_obs.ravel()
    flux_w = np.abs(flux) / (np.abs(flux).sum() + 1e-9)
    theta_E_approx = float(np.dot(flux_w, ea_col * r_vals) /
                           (np.dot(flux_w, ea_col) + 1e-9))
    # 물리적 범위로 clamp [0.1, 5.0] arcsec
    theta_E_approx = float(np.clip(theta_E_approx, 0.1, 5.0))

    # 4. SIS 역방향 렌즈 방정식: β = θ - θ_E * θ/|θ|
    r = r_vals + 1e-9
    src_x = final_x - theta_E_approx * final_x / r
    src_y = final_y - theta_E_approx * final_y / r

    # 5. 직접 플럭스 역투영 (bilinear)
    source_img = _direct_flux_backproject(
        image_obs, src_x, src_y, H, W, pixel_scale
    )

    return {
        "source": source_img,
        "approx_level": approx_level,
        "solver_meta": {
            "solver_class": "SMBHMode",
            "solver_path": str(_SRC_PY / "modes" / "smbh.py"),
            "psf_applied": False,
            "n_pixels_used": H * W,
            "theta_E_approx_arcsec": theta_E_approx,
        },
    }
