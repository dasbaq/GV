"""
MultiModalErrorCorrector — mixed-mode 배치 forward + backward 1-step.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import yaml
from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.losses import composite_loss


IMG, D = 32, 128


@pytest.fixture(scope="module")
def cfg():
    with open(Path(__file__).parent.parent / "config" / "ml.yaml") as f:
        c = yaml.safe_load(f)
    return c


@pytest.fixture(scope="module")
def model(cfg):
    model_cfg = dict(cfg["model"])
    model_cfg["param_in_dim"] = len(cfg["data"]["param_normalization"]) + 5
    model_cfg["image_size"]   = IMG
    return MultiModalErrorCorrector(model_cfg)


def _make_batch(B=6, T=64, S=32, IMG=32, max_dm=4):
    """mode 1·2·3 혼합 배치 생성."""
    modes = torch.tensor([1, 1, 2, 2, 3, 3])
    use_image = (modes == 3)
    lc          = torch.randn(B, 2, T)
    lc_mask     = torch.ones(B, T, dtype=torch.bool)
    params      = torch.randn(B, 11)
    sigma_curve = torch.randn(B, 1, S)
    image       = torch.randn(B, 1, IMG, IMG)
    image[~use_image] = 0.0

    target      = torch.randn(B, max_dm + 2)
    target_dim  = torch.tensor([1, 1, 1, 1, 0, 0])
    target_image= torch.randn(B, IMG, IMG)
    target_image[modes != 3] = 0.0

    return {
        "lc": lc, "lc_mask": lc_mask, "params": params,
        "sigma_curve": sigma_curve, "image": image,
        "use_image": use_image, "target_mode": modes,
        "target": target, "target_dim": target_dim,
        "target_image": target_image,
    }


def test_forward_all_modes(model, cfg):
    batch = _make_batch()
    out   = model(**{k: batch[k] for k in
                     ["lc","lc_mask","params","sigma_curve","image","use_image","target_mode"]})
    assert out["mode1"] is not None
    assert out["mode2"] is not None
    assert out["mode3"] is not None


def test_mode1_output_shape(model, cfg):
    batch = _make_batch()
    out   = model(**{k: batch[k] for k in
                     ["lc","lc_mask","params","sigma_curve","image","use_image","target_mode"]})
    n1 = (batch["target_mode"] == 1).sum().item()
    assert out["mode1"]["h0_correction"].shape == (n1,)


def test_mode3_output_shape(model, cfg):
    batch = _make_batch()
    out   = model(**{k: batch[k] for k in
                     ["lc","lc_mask","params","sigma_curve","image","use_image","target_mode"]})
    n3 = (batch["target_mode"] == 3).sum().item()
    assert out["mode3"]["source_residual"].shape == (n3, 1, IMG, IMG)


def test_backward_one_step(model, cfg):
    """손실 backward → gradient 흐름 확인."""
    batch   = _make_batch()
    weights = cfg["training"]["loss_weights"]
    opt     = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.zero_grad()

    out  = model(**{k: batch[k] for k in
                    ["lc","lc_mask","params","sigma_curve","image","use_image","target_mode"]})
    ls   = composite_loss(out, batch, weights)
    ls["total"].backward()
    opt.step()

    # 임의 파라미터 grad 확인
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)


def test_mode_only_1_batch(model, cfg):
    """Mode 1만 있는 배치 — mode2·mode3 None."""
    B = 4
    batch = {
        "lc":           torch.randn(B, 2, 64),
        "lc_mask":      torch.ones(B, 64, dtype=torch.bool),
        "params":       torch.randn(B, 11),
        "sigma_curve":  torch.randn(B, 1, 32),
        "image":        torch.zeros(B, 1, IMG, IMG),
        "use_image":    torch.zeros(B, dtype=torch.bool),
        "target_mode":  torch.ones(B, dtype=torch.long),
    }
    out = model(**{k: batch[k] for k in
                   ["lc","lc_mask","params","sigma_curve","image","use_image","target_mode"]})
    assert out["mode1"] is not None
    assert out["mode2"] is None
    assert out["mode3"] is None


def test_log_sigma_clamp(model, cfg):
    """log_sigma clamp [-5, 2] 준수."""
    batch = _make_batch()
    out   = model(**{k: batch[k] for k in
                     ["lc","lc_mask","params","sigma_curve","image","use_image","target_mode"]})
    for mode_key in ["mode1", "mode2"]:
        if out[mode_key]:
            ls = out[mode_key]["log_sigma"]
            assert ls.min() >= -5.0 - 1e-5
            assert ls.max() <=  2.0 + 1e-5
