"""
LensCorrectionDataset 단위 테스트.
- shape / mask 정합성
- mode별 라벨 정확도
- split 누수 없음
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from ml.utils.mock_generator import create_mock_h5
from ml.training.dataset import LensCorrectionDataset, build_weighted_sampler

N_SYS = 64
IMG   = 32
MAX_LC = 128
SIGMA_S = 64


@pytest.fixture(scope="module")
def mock_h5(tmp_path_factory):
    p = tmp_path_factory.mktemp("h5") / "mock.h5"
    create_mock_h5(str(p), n_systems=N_SYS, image_size=IMG,
                   max_epochs=MAX_LC, seed=0)
    return p


@pytest.fixture(scope="module")
def norm_cfg():
    import yaml
    with open(Path(__file__).parent.parent / "config" / "ml.yaml") as f:
        return yaml.safe_load(f)["data"]["param_normalization"]


def _make_ds(mock_h5, norm_cfg, split, modes=(1, 2, 3)):
    return LensCorrectionDataset(
        h5_paths=[mock_h5], split=split,
        modes=modes, approx_levels=[1, 2],
        max_len=MAX_LC, sigma_curve_size=SIGMA_S,
        image_size=IMG, mode2_max_dm_dim=4,
        param_norm=norm_cfg, seed=42,
    )


# ---- shape 검증 ----
def test_item_shapes(mock_h5, norm_cfg):
    ds   = _make_ds(mock_h5, norm_cfg, "train")
    item = ds[0]

    assert item["lc"].shape        == (2, MAX_LC)
    assert item["lc_mask"].shape   == (MAX_LC,)
    assert item["sigma_curve"].shape == (1, SIGMA_S)
    assert item["image"].shape     == (1, IMG, IMG)
    assert item["target_image"].shape == (IMG, IMG)
    assert item["params"].ndim     == 1


def test_lc_mask_valid_range(mock_h5, norm_cfg):
    ds = _make_ds(mock_h5, norm_cfg, "train")
    item = ds[0]
    n_valid = item["lc_mask"].sum().item()
    assert 1 <= n_valid <= MAX_LC, f"유효 길이 {n_valid} 범위 초과"


def test_params_normalized(mock_h5, norm_cfg):
    """정규화된 파라미터 값은 대략 [0, 1] 범위."""
    ds = _make_ds(mock_h5, norm_cfg, "train")
    item = ds[0]
    n_norm = len(norm_cfg)
    normed = item["params"][:n_norm]
    # clamp 없으므로 약간 벗어날 수 있음 → [-0.5, 1.5]
    assert normed.min() >= -0.5 and normed.max() <= 1.5


# ---- mode별 라벨 ----
def test_mode1_label_is_scalar(mock_h5, norm_cfg):
    ds = _make_ds(mock_h5, norm_cfg, "train", modes=[1])
    for item in [ds[0], ds[1]]:
        assert item["target_mode"] == 1
        assert item["target_dim"]  == 1
        assert torch.isfinite(item["target"][0])


def test_mode2_label_padding(mock_h5, norm_cfg):
    ds = _make_ds(mock_h5, norm_cfg, "train", modes=[2])
    item = ds[0]
    assert item["target_mode"] == 2
    dm_dim = item["target_dim"]
    assert 1 <= dm_dim <= 4
    # 패딩 영역은 0
    assert (item["target"][dm_dim:] == 0).all()


def test_mode3_label_is_image(mock_h5, norm_cfg):
    ds = _make_ds(mock_h5, norm_cfg, "train", modes=[3])
    item = ds[0]
    assert item["target_mode"]   == 3
    assert item["target_dim"]    == 0
    assert item["use_image"]     == True
    assert item["target_image"].shape == (IMG, IMG)


def test_mode3_image_nonzero(mock_h5, norm_cfg):
    """Mode 3에서 image 입력이 실제 값을 가짐."""
    ds = _make_ds(mock_h5, norm_cfg, "train", modes=[3])
    item = ds[0]
    assert item["image"].abs().sum().item() > 0


# ---- split 누수 없음 ----
def test_no_split_leak(mock_h5, norm_cfg):
    train_ds = _make_ds(mock_h5, norm_cfg, "train", modes=[1])
    val_ds   = _make_ds(mock_h5, norm_cfg, "val",   modes=[1])
    test_ds  = _make_ds(mock_h5, norm_cfg, "test",  modes=[1])

    train_idx = {e[1] for e in train_ds._index}
    val_idx   = {e[1] for e in val_ds._index}
    test_idx  = {e[1] for e in test_ds._index}

    assert len(train_idx & val_idx)   == 0, "train-val 누수"
    assert len(train_idx & test_idx)  == 0, "train-test 누수"
    assert len(val_idx   & test_idx)  == 0, "val-test 누수"


def test_split_size(mock_h5, norm_cfg):
    train_sys = len({e[1] for e in _make_ds(mock_h5, norm_cfg, "train", modes=[1])._index})
    val_sys   = len({e[1] for e in _make_ds(mock_h5, norm_cfg, "val",   modes=[1])._index})
    test_sys  = len({e[1] for e in _make_ds(mock_h5, norm_cfg, "test",  modes=[1])._index})
    total     = train_sys + val_sys + test_sys
    assert total == N_SYS


# ---- WeightedRandomSampler ----
def test_weighted_sampler(mock_h5, norm_cfg):
    ds      = _make_ds(mock_h5, norm_cfg, "train")
    sampler = build_weighted_sampler(ds, mode_weights=[2.0, 1.0, 1.0])
    assert len(sampler) == len(ds)
