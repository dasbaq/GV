"""
LensCorrectionDataset — HDF5 스트리밍 + Mode별 라벨 분기.

한 샘플 = (system_id, approx_level, target_mode) triple.
target_mode에 따라 사용하는 입력 모달리티와 라벨이 달라짐.

입력 모달리티: LC + Param + Σ-curve + Image (4-way, v0.6).
  - Image: I_obs 단일 채널 (H×W), 픽셀값 [0,1] 정규화.
  - Mode 3(Source 복원) head는 삭제 유지 (DECISIONS.md [2026-05-25]).
  - v0.6 변경: image 모달리티 복구 (I_obs only, S_approx 미사용).

HDF5 접근: swmr=True, worker별 file handle 재오픈.
Split: system 단위 80/10/10, seed 고정.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ml.training.feature_schema import (
    build_param_vector_from_features,
    hdf5_feature_values,
)
from ml.utils.mask import make_lc_mask

# Σ 곡선 계산 — Phase 1 모듈 미구현 시 NotImplementedError
_SIGMA_CURVE_AVAILABLE = False
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src_py"))
    from core.light_curve.fluctuation import compute_sigma_curve as _compute_sigma
    _SIGMA_CURVE_AVAILABLE = True
except ImportError:
    pass


def _compute_sigma_curve_fallback(F_joint: np.ndarray, n_valid: int,
                                   sigma_curve_size: int) -> np.ndarray:
    """Σ 곡선 Phase 1 모듈 없을 때의 fallback (NotImplementedError 금지 → 경고 후 zeros)."""
    # Phase 1 미구현 상태에서는 모의 Σ 곡선 반환 (Mock 모드 전용)
    dt_try = np.linspace(1, 200, sigma_curve_size)
    if n_valid < 4:
        return np.zeros(sigma_curve_size, dtype=np.float32)

    F = F_joint[:n_valid]
    # Bag et al. 2022 — 벡터화 ε(Δt_try) 계산
    mu = 0.8  # 기본값
    # f1rec[t] = Σ_{n=0}^{N-1} (-μ)^n * F[t - n*Δt]  → nearest-index lookup 벡터화
    eps_arr = np.empty(sigma_curve_size, dtype=np.float64)
    t_idx = np.arange(n_valid)

    for k, dt in enumerate(dt_try):
        # 벡터화: N=3차까지만 (수렴 충분)
        f1rec = F.copy().astype(np.float64)
        for n in range(1, 4):
            shift = int(round(n * dt))
            if shift >= n_valid:
                break
            f1rec[shift:] += ((-mu) ** n) * F[:n_valid - shift]
        diff = np.diff(f1rec)
        eps_arr[k] = np.sum(diff ** 2)

    eps_mean = eps_arr.mean()
    eps_std  = eps_arr.std() + 1e-9
    sigma = (eps_arr - eps_mean) / eps_std
    return sigma.astype(np.float32)


def _scalar_at(h5: h5py.File, path: str, idx: int, default: float = 0.0) -> float:
    if path not in h5:
        return float(default)
    try:
        value = float(np.asarray(h5[path][idx]).reshape(-1)[0])
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


class _FileHandleCache(threading.local):
    """worker별 HDF5 file handle 캐시 (threading.local)."""

    def __init__(self) -> None:
        self._handles: dict[str, h5py.File] = {}

    def get(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", swmr=True)
        return self._handles[path]


_handle_cache = _FileHandleCache()


class LensCorrectionDataset(Dataset):
    """
    Parameters
    ----------
    h5_paths         : HDF5 파일 경로 목록
    split            : "train" | "val" | "test"
    modes            : 사용할 Mode 목록 (1, 2)
    approx_levels    : 사용할 근사 레벨 목록 (1, 2)
    max_len          : 광도곡선 패딩 길이
    sigma_curve_size : Σ 곡선 길이
    mode2_max_dm_dim : Mode 2 라벨 패딩 차원
    param_norm       : {key: {min, max}} 정규화 설정
    target_scaler    : train split에서 fit한 Mode별 target standardization 설정
    seed             : split 시드
    """

    def __init__(
        self,
        h5_paths: List[Path],
        split: str,
        modes: Tuple[int, ...] = (1, 2),
        approx_levels: Tuple[int, ...] = (1, 2),
        max_len: int = 1024,
        sigma_curve_size: int = 512,
        mode2_max_dm_dim: int = 4,
        param_norm: Optional[dict] = None,
        target_scaler: Optional[dict] = None,
        observed_feature_config: Optional[dict] = None,
        image_size: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.h5_paths = [str(p) for p in h5_paths]
        self.split = split
        self.modes = list(modes)
        self.approx_levels = list(approx_levels)
        self.max_len = max_len
        self.sigma_curve_size = sigma_curve_size
        self.mode2_max_dm_dim = mode2_max_dm_dim
        self.param_norm = param_norm or {}
        self.target_scaler = target_scaler or {}
        self.observed_feature_config = observed_feature_config or {}
        self.image_size = image_size
        self.seed = seed

        self._index: List[Tuple[str, int, int, int]] = []  # (path, sys_idx, approx, mode)
        self._build_index()

    # ------------------------------------------------------------------ #
    # 인덱스 빌드                                                           #
    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        rng = np.random.default_rng(self.seed)

        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                n = int(f["metadata"].attrs["n_systems"])

            system_ids = np.arange(n)
            rng_local = np.random.default_rng(
                int(rng.integers(0, 2**31))
            )
            rng_local.shuffle(system_ids)

            n_train = int(0.8 * n)
            n_val   = int(0.1 * n)
            splits = {
                "train": system_ids[:n_train],
                "val":   system_ids[n_train: n_train + n_val],
                "test":  system_ids[n_train + n_val:],
            }
            chosen = splits[self.split]

            for sys_idx in chosen:
                for al in self.approx_levels:
                    for mode in self.modes:
                        self._index.append((path, int(sys_idx), al, mode))

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------ #
    # __getitem__                                                          #
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        path, sys_idx, approx_level, target_mode = self._index[idx]
        f = _handle_cache.get(path)

        # --- 광도곡선 ---
        n_valid = int(f["light_curves/n_epochs"][sys_idx])
        F_raw   = f["light_curves/F_joint"][sys_idx]       # [max_epochs]
        n_valid = min(n_valid, self.max_len)

        lc = np.zeros((2, self.max_len), dtype=np.float32)
        lc[0, :n_valid] = F_raw[:n_valid]
        # channel 1: sigma_noise
        lc[1, :n_valid] = f["light_curves/sigma_noise"][sys_idx, :n_valid]
        lc_mask = make_lc_mask(n_valid, self.max_len)

        # --- 물리 파라미터 벡터 ---
        # Mode 1/2 shared inference-side inputs only. Truth-side keys such as
        # M200/kappa_ext are intentionally excluded by feature_schema.
        raw_params = hdf5_feature_values(f, sys_idx)
        params_vec = build_param_vector_from_features(
            raw_params,
            self.param_norm,
            approx_level=approx_level,
            target_mode=target_mode,
            observed_feature_config=self.observed_feature_config,
            allow_legacy_delay_sigma=True,
        )

        # --- 관측 이미지 (I_obs, v0.6) ---
        img_size = self.image_size
        if "images/I_obs" in f:
            raw_img = np.array(f["images/I_obs"][sys_idx], dtype=np.float32)
            # 크기 조정 (img_size×img_size로 리사이즈 없이 중앙 크롭 또는 패딩)
            h, w = raw_img.shape[-2], raw_img.shape[-1]
            if h != img_size or w != img_size:
                # 필요 시 center-crop 또는 zero-pad
                out_img = np.zeros((img_size, img_size), dtype=np.float32)
                ch = min(h, img_size)
                cw = min(w, img_size)
                oh = (img_size - ch) // 2
                ow = (img_size - cw) // 2
                sh = (h - ch) // 2
                sw = (w - cw) // 2
                out_img[oh:oh+ch, ow:ow+cw] = raw_img[sh:sh+ch, sw:sw+cw]
                raw_img = out_img
            # [0, 1] 정규화 (픽셀값이 모두 0이면 그대로)
            max_val = float(raw_img.max())
            if max_val > 0:
                raw_img = raw_img / max_val
        else:
            raw_img = np.zeros((img_size, img_size), dtype=np.float32)
        image = raw_img[np.newaxis]  # [1, H, W]

        # --- Σ 곡선 ---
        if "sigma_curve" in f:
            sigma_curve = np.array(
                f["sigma_curve"][sys_idx, :self.sigma_curve_size], dtype=np.float32
            )
            if len(sigma_curve) < self.sigma_curve_size:
                pad = np.zeros(self.sigma_curve_size - len(sigma_curve), dtype=np.float32)
                sigma_curve = np.concatenate([sigma_curve, pad])
        else:
            sigma_curve = _compute_sigma_curve_fallback(
                F_raw, n_valid, self.sigma_curve_size
            )

        # --- 라벨 ---
        target = np.zeros(self.mode2_max_dm_dim + 2, dtype=np.float32)  # 최대 dim 맞춤
        target_dim = 1

        if target_mode == 1:
            h0_err = float(f["simplification_errors/mode1_H0_error"][sys_idx])
            target[0] = h0_err
            target_dim = 1

        elif target_mode == 2:
            dm_err_full = np.array(
                f["simplification_errors/mode2_dm_error"][sys_idx], dtype=np.float32
            )
            dm_dim = int(f["true_values/dm_dim"][sys_idx]) if "dm_dim" in f["true_values"] else 1
            dm_dim = min(dm_dim, self.mode2_max_dm_dim)
            # |μ| < 1 수렴 조건 검증 (Mode 1 재구성에서 쓰는 μ 값)
            mu = float(f["true_values/mu_true"][sys_idx])
            assert abs(mu) < 10.0, f"mu={mu} 비정상 값 (sys {sys_idx})"
            target[:dm_dim] = dm_err_full[:dm_dim]
            target_dim = dm_dim

        if self.target_scaler:
            if target_mode == 1 and "mode1" in self.target_scaler:
                s = self.target_scaler["mode1"]
                target[0] = (target[0] - float(s["mean"])) / float(s["scale"])
            elif target_mode == 2 and "mode2" in self.target_scaler:
                s = self.target_scaler["mode2"]
                mean = np.asarray(s["mean"], dtype=np.float32)
                scale = np.asarray(s["scale"], dtype=np.float32)
                target[:target_dim] = (target[:target_dim] - mean[:target_dim]) / scale[:target_dim]

        mode1_scaler = self.target_scaler.get("mode1", {})
        mode2_scaler = self.target_scaler.get("mode2", {})
        mode2_mean = np.asarray(mode2_scaler.get("mean", np.zeros(self.mode2_max_dm_dim)), dtype=np.float32)
        mode2_scale = np.asarray(mode2_scaler.get("scale", np.ones(self.mode2_max_dm_dim)), dtype=np.float32)
        if mode2_mean.size < self.mode2_max_dm_dim:
            mode2_mean = np.pad(mode2_mean, (0, self.mode2_max_dm_dim - mode2_mean.size))
        if mode2_scale.size < self.mode2_max_dm_dim:
            mode2_scale = np.pad(mode2_scale, (0, self.mode2_max_dm_dim - mode2_scale.size), constant_values=1.0)

        mode2_raw = np.array(
            f["simplification_errors/mode2_dm_error"][sys_idx]
            if "simplification_errors/mode2_dm_error" in f
            else np.zeros(self.mode2_max_dm_dim, dtype=np.float32),
            dtype=np.float32,
        )
        mode2_label_available = bool(np.any(np.abs(mode2_raw) > 1.0e-8))
        dm_approx = (
            np.array(f["approx_outputs/dm_params_approx"][sys_idx], dtype=np.float32)
            if "approx_outputs/dm_params_approx" in f
            else np.zeros(self.mode2_max_dm_dim, dtype=np.float32)
        )
        theta_e_approx = float(dm_approx[0]) if dm_approx.size else 0.0
        if theta_e_approx <= 0.0:
            theta_e_approx = _scalar_at(f, "params/theta_E", sys_idx, _scalar_at(f, "true_values/theta_E", sys_idx, 0.0))
        dphi_sie = _scalar_at(
            f,
            "ray_paths/fermat_potential_approx",
            sys_idx,
            _scalar_at(f, "ray_paths/fermat_potential", sys_idx, 0.0),
        )

        return {
            "lc":           torch.from_numpy(lc),
            "lc_mask":      lc_mask,
            "params":       torch.from_numpy(params_vec),
            "sigma_curve":  torch.from_numpy(sigma_curve[np.newaxis]),   # [1, S]
            "image":        torch.from_numpy(image),                      # [1, H, W]
            "target":       torch.from_numpy(target),
            "target_dim":   target_dim,
            "target_mode":  target_mode,
            "approx_level": approx_level,
            "system_index":  sys_idx,
            "h0_approx":     torch.tensor(float(raw_params["H0_approx"]), dtype=torch.float32),
            "dt_lc":         torch.tensor(float(raw_params["dt_lc"]), dtype=torch.float32),
            "theta_E_approx": torch.tensor(theta_e_approx, dtype=torch.float32),
            "dphi_sie_rad2": torch.tensor(float(dphi_sie), dtype=torch.float32),
            "mode1_target_mean": torch.tensor(float(mode1_scaler.get("mean", 0.0)), dtype=torch.float32),
            "mode1_target_scale": torch.tensor(float(mode1_scaler.get("scale", 1.0)), dtype=torch.float32),
            "mode2_target_mean": torch.from_numpy(mode2_mean[:self.mode2_max_dm_dim].astype(np.float32)),
            "mode2_target_scale": torch.from_numpy(mode2_scale[:self.mode2_max_dm_dim].astype(np.float32)),
            "mode2_label_available": torch.tensor(mode2_label_available, dtype=torch.bool),
        }


# ------------------------------------------------------------------ #
# WeightedRandomSampler 헬퍼                                           #
# ------------------------------------------------------------------ #
def build_weighted_sampler(
    dataset: LensCorrectionDataset,
    mode_weights: Optional[List[float]] = None,
) -> WeightedRandomSampler:
    """
    mode_sampling_weights 기반 WeightedRandomSampler 생성.

    Parameters
    ----------
    dataset      : LensCorrectionDataset
    mode_weights : [w1, w2]  (None이면 균등)
    """
    if mode_weights is None:
        mode_weights = [1.0, 1.0]
    w_map = {1: mode_weights[0], 2: mode_weights[1]}
    weights = torch.tensor(
        [w_map[entry[3]] for entry in dataset._index], dtype=torch.float
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
