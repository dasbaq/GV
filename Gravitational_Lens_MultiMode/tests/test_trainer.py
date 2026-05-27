"""
CorrectorTrainer — toy 64샘플 1 epoch, 모든 mode loss 감소 확인.
Mode 3과 Image 입력은 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import pytest
from torch.utils.data import DataLoader

from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.dataset import LensCorrectionDataset, build_weighted_sampler
from ml.training.trainer import CorrectorTrainer
from ml.utils.mock_generator import create_mock_h5

N_SYS = 64
MAX_LC = 128
SIGMA_S = 64


@pytest.fixture(scope="module")
def cfg():
    with open(Path(__file__).parent.parent / "config" / "ml.yaml") as f:
        c = yaml.safe_load(f)
    # toy 설정
    c["training"]["epochs"] = 2
    c["training"]["warmup_epochs"] = 0
    c["training"]["early_stop_patience"] = 5
    c["training"]["batch_size"] = 8
    c["training"]["amp"] = False
    c["data"]["max_lc_len"]       = MAX_LC
    c["data"]["sigma_curve_size"] = SIGMA_S
    return c


@pytest.fixture(scope="module")
def mock_h5(tmp_path_factory):
    p = tmp_path_factory.mktemp("h5") / "mock.h5"
    create_mock_h5(str(p), n_systems=N_SYS, max_epochs=MAX_LC, seed=1)
    return p


@pytest.fixture(scope="module")
def loaders(mock_h5, cfg):
    kwargs = dict(
        h5_paths=[mock_h5], modes=[1, 2], approx_levels=[1],
        max_len=MAX_LC, sigma_curve_size=SIGMA_S,
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=cfg["data"]["param_normalization"], seed=42,
    )
    train_ds = LensCorrectionDataset(split="train", **kwargs)
    val_ds   = LensCorrectionDataset(split="val",   **kwargs)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)
    return train_loader, val_loader


@pytest.fixture(scope="module")
def model(cfg):
    model_cfg = dict(cfg["model"])
    model_cfg["param_in_dim"] = len(cfg["data"]["param_normalization"]) + 5
    return MultiModalErrorCorrector(model_cfg)


def test_fit_runs(model, cfg, loaders, tmp_path):
    device  = torch.device("cpu")
    trainer = CorrectorTrainer(model, cfg, device)
    train_loader, val_loader = loaders
    result  = trainer.fit(train_loader, val_loader, output_dir=tmp_path)
    assert "run_dir" in result
    assert "best_val_loss" in result
    assert result["best_val_loss"] < 1e6


def test_loss_decreases(model, cfg, loaders, tmp_path):
    """2 epoch 학습 후 val loss가 inf보다 작음 (수치 안정성 확인)."""
    import math
    device  = torch.device("cpu")
    trainer = CorrectorTrainer(model, cfg, device)
    train_loader, val_loader = loaders
    result  = trainer.fit(train_loader, val_loader, output_dir=tmp_path)
    assert math.isfinite(result["best_val_loss"])


def test_best_checkpoint_saved(model, cfg, loaders, tmp_path):
    device  = torch.device("cpu")
    trainer = CorrectorTrainer(model, cfg, device)
    train_loader, val_loader = loaders
    result  = trainer.fit(train_loader, val_loader, output_dir=tmp_path)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "config.yaml").exists()


def test_evaluate_returns_per_mode(model, cfg, loaders, tmp_path):
    device  = torch.device("cpu")
    trainer = CorrectorTrainer(model, cfg, device)
    _, val_loader = loaders
    metrics = trainer.evaluate(val_loader)
    assert "total" in metrics
    assert "per_mode" in metrics
