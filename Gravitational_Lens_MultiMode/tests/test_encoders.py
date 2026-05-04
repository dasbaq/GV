"""
인코더 forward shape + mask 효과 검증.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from ml.models.encoders import (
    LightCurveEncoder, ParamEncoder, SigmaCurveEncoder, ImageEncoder
)
from ml.utils.mask import make_lc_mask


B, T, S, H, D = 4, 256, 128, 64, 128


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
    return torch.randn(B, 11)

@pytest.fixture
def sigma():
    return torch.randn(B, 1, S)

@pytest.fixture
def image():
    return torch.randn(B, 1, H, H)


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
    enc = ParamEncoder(in_dim=11, d_model=D)
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


# ---- ImageEncoder ----
def test_image_encoder_shape(image):
    enc = ImageEncoder(D)
    global_feat, skips = enc(image)
    assert global_feat.shape == (B, D)
    assert len(skips) == 4

def test_image_encoder_skip_shapes(image):
    enc = ImageEncoder(D)
    _, skips = enc(image)
    s1, s2, s3, s4 = skips
    assert s1.shape == (B, 32,  H,   H)
    assert s2.shape == (B, 64,  H//2, H//2)
    assert s3.shape == (B, 128, H//4, H//4)
    assert s4.shape == (B, D,   H//8, H//8)
