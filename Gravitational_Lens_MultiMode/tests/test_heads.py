"""
Mode 1/2/3 head forward shape + log_sigma clamp 검증.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from ml.models.heads import Mode1Head, Mode2Head, Mode3Head

B, D, MAX_DM, IMG = 4, 128, 4, 32


@pytest.fixture
def fused():
    return torch.randn(B, D)

@pytest.fixture
def skips():
    return [
        torch.randn(B, 32,  IMG,    IMG),
        torch.randn(B, 64,  IMG//2, IMG//2),
        torch.randn(B, 128, IMG//4, IMG//4),
        torch.randn(B, D,   IMG//8, IMG//8),
    ]


# ---- Mode1Head ----
def test_mode1_shape(fused):
    h   = Mode1Head(D)
    out = h(fused)
    assert out["h0_correction"].shape == (B,)
    assert out["log_sigma"].shape     == (B,)

def test_mode1_log_sigma_clamp(fused):
    h   = Mode1Head(D)
    out = h(fused)
    assert out["log_sigma"].min() >= -5.0 - 1e-5
    assert out["log_sigma"].max() <=  2.0 + 1e-5

def test_mode1_grad():
    h = Mode1Head(D)
    x = torch.randn(2, D, requires_grad=True)
    out = h(x)
    out["h0_correction"].sum().backward()
    assert x.grad is not None


# ---- Mode2Head ----
def test_mode2_shape(fused):
    h   = Mode2Head(D, MAX_DM)
    out = h(fused)
    assert out["dm_correction"].shape == (B, MAX_DM)
    assert out["log_sigma"].shape     == (B, MAX_DM)

def test_mode2_log_sigma_clamp(fused):
    h   = Mode2Head(D, MAX_DM)
    out = h(fused)
    assert out["log_sigma"].min() >= -5.0 - 1e-5
    assert out["log_sigma"].max() <=  2.0 + 1e-5


# ---- Mode3Head ----
def test_mode3_shape(fused, skips):
    h   = Mode3Head(D, IMG)
    out = h(fused, skips)
    assert out["source_residual"].shape == (B, 1, IMG, IMG)

def test_mode3_finite(fused, skips):
    h   = Mode3Head(D, IMG)
    out = h(fused, skips)
    assert torch.isfinite(out["source_residual"]).all()

def test_mode3_grad(fused, skips):
    h = Mode3Head(D, IMG)
    x = fused.clone().requires_grad_(True)
    out = h(x, skips)
    out["source_residual"].sum().backward()
    assert x.grad is not None
