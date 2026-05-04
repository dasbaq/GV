"""Mode 3 wrapper — 출력 shape, NaN 없음 테스트."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from inversion.mode3_wrapper import reconstruct_source


def _dummy_psf(size: int = 11) -> np.ndarray:
    yp, xp = np.mgrid[-size//2:size//2+1, -size//2:size//2+1]
    k = np.exp(-(xp**2 + yp**2) / (2 * 1.5**2)).astype(np.float32)
    return k / k.sum()


def test_output_shape():
    """출력 source 이미지 shape = 입력 이미지 shape."""
    H, W = 64, 64
    img = np.random.rand(H, W).astype(np.float32)
    psf = _dummy_psf()
    result = reconstruct_source(img, psf, pixel_scale=0.05,
                                 lens_params={"dt_scale": 10.0}, approx_level=1)
    assert result["source"].shape == (H, W), f"shape mismatch: {result['source'].shape}"


def test_no_nan():
    """출력에 NaN 없음."""
    img = np.random.rand(32, 32).astype(np.float32)
    psf = _dummy_psf()
    result = reconstruct_source(img, psf, pixel_scale=0.05, lens_params={})
    assert not np.any(np.isnan(result["source"])), "source에 NaN 존재"


def test_output_keys():
    """필수 키 존재."""
    img = np.ones((32, 32), dtype=np.float32)
    psf = _dummy_psf()
    result = reconstruct_source(img, psf, pixel_scale=0.05, lens_params={})
    for k in ["source", "approx_level", "solver_meta"]:
        assert k in result


def test_approx_level_stored():
    """approx_level이 반환값에 저장."""
    img = np.random.rand(32, 32).astype(np.float32)
    psf = _dummy_psf()
    for al in [1, 2]:
        result = reconstruct_source(img, psf, pixel_scale=0.05,
                                     lens_params={}, approx_level=al)
        assert result["approx_level"] == al
