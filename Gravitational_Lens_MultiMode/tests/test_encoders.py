"""
인코더 forward shape + mask 효과 검증.
ImageEncoder는 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import yaml
from ml.models.encoders import (
    LightCurveEncoder, ParamEncoder, SigmaCurveEncoder,
)
from ml.utils.mask import make_lc_mask


B, T, S, D = 4, 256, 128, 128


@pytest.fixture
def lc():
    return torch.randn(B, 2, T)

@pytest.fixture
def mask():
    m = torch.ones(B, T, dtype=torch.bool)
    m[0, 200:] = False   # sample 0: 200개만 유효
    return m

@pytest.fixture
def params():
    with open(Path(__file__).parent.parent / "config" / "ml.yaml") as f:
        param_dim = len(yaml.safe_load(f)["data"]["param_normalization"]) + 5
    return torch.randn(B, param_dim)

@pytest.fixture
def sigma():
    return torch.randn(B, 1, S)


# ---- LightCurveEncoder ----
def test_lc_encoder_shape(lc, mask):
    enc  = LightCurveEncoder(D)
    out  = enc(lc, mask)
    assert out.shape == (B, D)

def test_lc_encoder_mask_effect(lc):
    enc = LightCurveEncoder(D)
    enc.eval()
    m_full  = torch.ones(B, T, dtype=torch.bool)
    m_half  = torch.ones(B, T, dtype=torch.bool)
    m_half[:, T//2:] = False
    with torch.no_grad():
        out_full = enc(lc, m_full)
        out_half = enc(lc, m_half)
    # mask 차이 → 출력 달라야 함
    assert not torch.allclose(out_full, out_half)


# ---- ParamEncoder ----
def test_param_encoder_shape(params):
    enc = ParamEncoder(in_dim=params.shape[1], d_model=D)
    out = enc(params)
    assert out.shape == (B, D)

def test_param_encoder_grad():
    enc = ParamEncoder(in_dim=8, d_model=D)
    x   = torch.randn(2, 8, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None


# ---- SigmaCurveEncoder ----
def test_sigma_encoder_shape(sigma):
    enc = SigmaCurveEncoder(D)
    out = enc(sigma)
    assert out.shape == (B, D)
