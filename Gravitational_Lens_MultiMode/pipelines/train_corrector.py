"""
train_corrector.py — Phase 5 CLI 엔트리포인트.

Usage:
  # 사전학습 (multi-task)
  python -m pipelines.train_corrector --config config/ml.yaml --stage pretrain \
      --synthetic "data/simulation_*.h5" --modes 1,2,3 --output runs/

  # Mode별 fine-tune
  python -m pipelines.train_corrector --config config/ml.yaml --stage finetune \
      --ckpt runs/<id>/best.pt --real "data/real/*.h5" --modes 1 --output runs/

  # 추론 + 보정
  python -m pipelines.train_corrector --config config/ml.yaml --stage infer \
      --ckpt runs/<id>/best.pt --input data/observed.h5 --target-mode 1 --out results/

  # 검증
  python -m pipelines.train_corrector --config config/ml.yaml --stage evaluate \
      --ckpt runs/<id>/best.pt --benchmarks all

  --dry-run : 1 step만 실행 (CI용)
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.dataset import LensCorrectionDataset, build_weighted_sampler
from ml.training.trainer import CorrectorTrainer
from ml.utils.mock_generator import create_mock_h5
from ml.utils.seed import set_seed


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
    return [p for p in paths if p.suffix == ".h5"]


def _build_model(cfg: dict) -> MultiModalErrorCorrector:
    param_norm = cfg["data"]["param_normalization"]
    param_in_dim = len(param_norm) + 5  # scalar feature schema + 2 approx + 3 mode
    model_cfg = dict(cfg["model"])
    model_cfg["param_in_dim"] = param_in_dim
    return MultiModalErrorCorrector(model_cfg)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader_kwargs(device: torch.device) -> dict:
    return {
        "num_workers": 4,
        "persistent_workers": True,
        "pin_memory": device.type == "cuda",
    }


def _build_loaders(h5_paths: list[Path], cfg: dict,
                   dry_run: bool, modes: list[int],
                   target_scaler: dict | None = None,
                   device: torch.device | None = None) -> tuple:
    data_cfg = cfg["data"]
    device = device or select_device()
    loader_kwargs = _loader_kwargs(device)
    train_ds = LensCorrectionDataset(
        h5_paths=h5_paths, split="train",
        modes=modes,
        approx_levels=data_cfg["approx_levels"],
        max_len=data_cfg["max_lc_len"],
        sigma_curve_size=data_cfg["sigma_curve_size"],
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=data_cfg["param_normalization"],
        target_scaler=target_scaler,
        observed_feature_config=data_cfg.get("observed_features", {}),
        seed=cfg["seed"],
    )
    val_ds = LensCorrectionDataset(
        h5_paths=h5_paths, split="val",
        modes=modes,
        approx_levels=data_cfg["approx_levels"],
        max_len=data_cfg["max_lc_len"],
        sigma_curve_size=data_cfg["sigma_curve_size"],
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=data_cfg["param_normalization"],
        target_scaler=target_scaler,
        observed_feature_config=data_cfg.get("observed_features", {}),
        seed=cfg["seed"],
    )

    batch_size = 4 if dry_run else cfg["training"]["batch_size"]
    sampler = build_weighted_sampler(train_ds, data_cfg["mode_sampling_weights"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader


def _ensure_mock(cfg: dict) -> list[Path]:
    """simulation_*.h5 없으면 mock 생성 후 경로 반환."""
    mock_dir  = ROOT / "data" / "mock"
    mock_path = mock_dir / "simulation_mock.h5"
    if not mock_path.exists():
        print("[MOCK] simulation_*.h5 미존재 → mock 데이터 생성 중...")
        create_mock_h5(
            out_path=mock_path,
            n_systems=1024,
            max_epochs=cfg["data"]["max_lc_len"],
            mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
            seed=cfg["seed"],
        )
        print(f"[MOCK] 생성 완료: {mock_path}")
    return [mock_path]


def _load_target_scaler(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _apply_mode1_only(cfg: dict) -> None:
    weights = cfg["training"].setdefault("loss_weights", {})
    weights["mode2"] = 0.0
    weights["physics"] = 0.0


# ======================================================================= #
# Stage: pretrain                                                           #
# ======================================================================= #
def stage_pretrain(args: argparse.Namespace, cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = select_device()
    print(f"Device: {device}")

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.mode1_only:
        _apply_mode1_only(cfg)

    h5_paths = _resolve_paths(args.synthetic) if args.synthetic else []
    if not h5_paths:
        h5_paths = _ensure_mock(cfg)

    modes = [int(m) for m in args.modes.split(",")] if args.modes else [1, 2]

    target_scaler = _load_target_scaler(args.target_scaler)
    train_loader, val_loader = _build_loaders(
        h5_paths, cfg, args.dry_run, modes, target_scaler=target_scaler, device=device
    )
    model = _build_model(cfg)

    if args.dry_run:
        cfg["training"]["epochs"] = 1

    trainer = CorrectorTrainer(model, cfg, device)
    result  = trainer.fit(train_loader, val_loader,
                          output_dir=Path(args.output) if args.output else None)
    if args.checkpoint_out:
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(result["run_dir"]) / "best.pt", args.checkpoint_out)
    if args.history_out:
        history_path = Path(args.history_out)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = [
            {"epoch": i + 1, "train": tr, "val": vl}
            for i, (tr, vl) in enumerate(zip(result["history"]["train"], result["history"]["val"]))
        ]
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    print(f"Pretrain done. Run dir: {result['run_dir']}")
    print(f"Best val loss: {result['best_val_loss']:.6f}")


# ======================================================================= #
# Stage: finetune                                                           #
# ======================================================================= #
def stage_finetune(args: argparse.Namespace, cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = select_device()

    if not args.ckpt:
        raise ValueError("--ckpt 필수 (finetune stage)")

    h5_paths = _resolve_paths(args.real) if args.real else []
    if not h5_paths:
        h5_paths = _ensure_mock(cfg)

    modes = [int(m) for m in args.modes.split(",")] if args.modes else [1]

    train_loader, val_loader = _build_loaders(h5_paths, cfg, args.dry_run, modes, device=device)
    model = _build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    if args.dry_run:
        cfg["training"]["epochs"] = 1

    trainer = CorrectorTrainer(model, cfg, device)
    result  = trainer.fine_tune(
        train_loader, val_loader,
        freeze_encoders=True,
        output_dir=Path(args.output) if args.output else None,
    )
    print(f"Finetune done. Run dir: {result['run_dir']}")


# ======================================================================= #
# Stage: infer                                                              #
# ======================================================================= #
def stage_infer(args: argparse.Namespace, cfg: dict) -> None:
    device = select_device()

    if not args.ckpt:
        raise ValueError("--ckpt 필수 (infer stage)")
    if not args.input:
        raise ValueError("--input 필수 (infer stage)")

    model = _build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval().to(device)

    target_mode = int(args.target_mode) if args.target_mode else 1

    ds = LensCorrectionDataset(
        h5_paths=[Path(args.input)], split="test",
        modes=[target_mode],
        approx_levels=cfg["data"]["approx_levels"],
        max_len=cfg["data"]["max_lc_len"],
        sigma_curve_size=cfg["data"]["sigma_curve_size"],
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=cfg["data"]["param_normalization"],
        observed_feature_config=cfg["data"].get("observed_features", {}),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, **_loader_kwargs(device)
    )

    out_dir = Path(args.out) if args.out else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

            pred = model(
                lc=batch_d["lc"], lc_mask=batch_d["lc_mask"],
                params=batch_d["params"], sigma_curve=batch_d["sigma_curve"],
                target_mode=batch_d["target_mode"].to(device),
            )
            if target_mode == 1 and pred["mode1"]:
                results.append({
                    "idx": i,
                    "h0_correction": pred["mode1"]["h0_correction"].cpu().numpy().tolist(),
                })
            elif target_mode == 2 and pred["mode2"]:
                results.append({
                    "idx": i,
                    "dm_correction": pred["mode2"]["dm_correction"].cpu().numpy().tolist(),
                })

            if args.dry_run:
                break

    import json
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"추론 완료: {out_dir}/predictions.json")


# ======================================================================= #
# Stage: evaluate                                                           #
# ======================================================================= #
def stage_evaluate(args: argparse.Namespace, cfg: dict) -> None:
    import json
    device = select_device()

    if not args.ckpt:
        raise ValueError("--ckpt 필수 (evaluate stage)")

    model = _build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    h5_paths = _ensure_mock(cfg)

    test_ds = LensCorrectionDataset(
        h5_paths=h5_paths, split="test",
        modes=[1, 2],
        approx_levels=cfg["data"]["approx_levels"],
        max_len=cfg["data"]["max_lc_len"],
        sigma_curve_size=cfg["data"]["sigma_curve_size"],
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=cfg["data"]["param_normalization"],
        observed_feature_config=cfg["data"].get("observed_features", {}),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, **_loader_kwargs(device)
    )

    trainer = CorrectorTrainer(model, cfg, device)
    metrics = trainer.evaluate(test_loader)
    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        if k != "per_mode":
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    if "per_mode" in metrics:
        for k, v in metrics["per_mode"].items():
            print(f"  {k}: {v:.6f}")


# ======================================================================= #
# main                                                                      #
# ======================================================================= #
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 — ML 보정 시스템 CLI")
    parser.add_argument("--config",      default="config/ml.yaml")
    parser.add_argument("--stage",       choices=["pretrain", "finetune", "infer", "evaluate"],
                        required=True)
    parser.add_argument("--synthetic",   help="합성 HDF5 glob 패턴")
    parser.add_argument("--real",        help="실측 HDF5 glob 패턴")
    parser.add_argument("--ckpt",        help="체크포인트 경로")
    parser.add_argument("--input",       help="추론 입력 HDF5")
    parser.add_argument("--target-mode", help="추론 대상 Mode (1/2)", dest="target_mode")
    parser.add_argument("--modes",       help="학습 Mode 목록 (예: 1,2)")
    parser.add_argument("--output",      help="runs/ 출력 디렉토리")
    parser.add_argument("--out",         help="추론 결과 출력 디렉토리")
    parser.add_argument("--benchmarks",  help="벤치마크 대상 (all 또는 쉼표구분)")
    parser.add_argument("--target-scaler", help="Mode별 target scaler pickle 경로")
    parser.add_argument("--epochs", type=int, help="학습 epoch override")
    parser.add_argument("--checkpoint-out", help="best.pt 복사 저장 경로")
    parser.add_argument("--history-out", help="epoch history JSON 저장 경로")
    parser.add_argument("--mode1-only", action="store_true",
                        help="Mode 1 loss만 활성화(mode2/physics weight=0)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="데이터 로드 + 모델 빌드 + 1 step (CI용)")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    if args.stage == "pretrain":
        stage_pretrain(args, cfg)
    elif args.stage == "finetune":
        stage_finetune(args, cfg)
    elif args.stage == "infer":
        stage_infer(args, cfg)
    elif args.stage == "evaluate":
        stage_evaluate(args, cfg)


if __name__ == "__main__":
    main()
