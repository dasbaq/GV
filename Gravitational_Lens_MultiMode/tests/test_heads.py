"""
Mode 1/2 head forward shape + log_sigma clamp 검증.
Mode 3(Source 복원) 헤드는 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from ml.models.heads import Mode1Head, Mode2Head

B, D, MAX_DM = 4, 128, 4


@pytest.fixture
def fused():
    return torch.randn(B, D)


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
