"""Min-max 정규화 유틸리티.  config/ml.yaml param_normalization 기반."""

from __future__ import annotations
import numpy as np
import torch


def minmax_normalize(value: float | np.ndarray | torch.Tensor,
                     lo: float, hi: float) -> float | np.ndarray | torch.Tensor:
    """[lo, hi] → [0, 1].  경계에서 clamp."""
    return (value - lo) / (hi - lo + 1e-9)


def build_param_vector(params_dict: dict, norm_cfg: dict) -> np.ndarray:
    """
    params_dict의 값들을 norm_cfg 기준으로 정규화한 벡터 반환.
    알 수 없는 키는 0으로 채움.

    Parameters
    ----------
    params_dict : dict  {key: float}
    norm_cfg    : dict  {key: {min: float, max: float}}

    Returns
    -------
    ndarray [len(norm_cfg)]
    """
    out = []
    for key, bounds in norm_cfg.items():
        val = params_dict.get(key, 0.0)
        out.append(float(minmax_normalize(val, bounds["min"], bounds["max"])))
    return np.array(out, dtype=np.float32)
