"""Shared runtime helpers for ML round entry scripts.

The helpers here only cover environment parsing, device selection, and output
path construction. Training loops, model construction, losses, and optimizer
settings stay in each round script so historical runs remain reproducible.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class RoundPaths:
    """Resolved paths for a round script.

    ``data`` is the filtered training/evaluation HDF5. ``unfiltered`` is only
    populated for rounds with a selection-bias evaluation catalog.
    """

    data: Path
    work_root: Path
    scaler: Path
    checkpoint: Path
    history: Path
    eval: Path
    infra: Path
    runs: Path
    scaler_from_env: bool = False
    unfiltered: Path | None = None
    eval_unfiltered: Path | None = None


def env_path(name: str) -> Path | None:
    """Return an environment path if ``name`` is set and non-empty."""

    value = os.environ.get(name)
    return Path(value) if value else None


def resolve_data_path(root: Path, data_root: Path, data_name: str) -> Path:
    """Resolve ``LENS_DATA_PATH`` with ``LENS_DATA_ROOT`` fallback."""

    exact = env_path("LENS_DATA_PATH")
    if exact is not None:
        return exact
    mock_path = data_root / "mock" / data_name
    if mock_path.exists():
        return mock_path
    return data_root / data_name


def resolve_unfiltered_path(
    data_root: Path,
    data_name: str,
    cli_path: Path | None = None,
) -> Path:
    """Resolve selection-bias eval data with CLI/env/root fallback."""

    if cli_path is not None:
        return cli_path
    exact = env_path("LENS_DATA_PATH_UNFILTERED")
    if exact is not None:
        return exact
    mock_path = data_root / "mock" / data_name
    if mock_path.exists():
        return mock_path
    return data_root / data_name


def resolve_work_root(root: Path) -> Path:
    """Resolve ``LENS_WORK_ROOT`` or default to repo ``data/``."""

    return env_path("LENS_WORK_ROOT") or root / "data"


def resolve_scaler_path(work_root: Path, default_name: str) -> Path:
    """Resolve ``LENS_SCALER_PATH`` or default to ``work_root/default_name``."""

    return env_path("LENS_SCALER_PATH") or work_root / default_name


def build_round_paths(
    *,
    root: Path,
    data_name: str,
    round_name: str,
    checkpoint_name: str,
    history_name: str,
    eval_name: str,
    infra_name: str,
    scaler_name: str,
    unfiltered_name: str | None = None,
    eval_unfiltered_name: str | None = None,
    cli_unfiltered: Path | None = None,
) -> RoundPaths:
    """Build standard round paths from the shared environment contract."""

    data_root = env_path("LENS_DATA_ROOT") or root / "data"
    work_root = resolve_work_root(root)
    unfiltered = None
    eval_unfiltered = None
    if unfiltered_name is not None:
        unfiltered = resolve_unfiltered_path(data_root, unfiltered_name, cli_unfiltered)
    if eval_unfiltered_name is not None:
        eval_unfiltered = work_root / "logs" / eval_unfiltered_name
    scaler_env = env_path("LENS_SCALER_PATH")
    return RoundPaths(
        data=resolve_data_path(root, data_root, data_name),
        work_root=work_root,
        scaler=scaler_env or work_root / scaler_name,
        checkpoint=work_root / "checkpoints" / checkpoint_name,
        history=work_root / "logs" / history_name,
        eval=work_root / "logs" / eval_name,
        infra=work_root / "logs" / infra_name,
        runs=work_root / "runs" / round_name,
        scaler_from_env=scaler_env is not None,
        unfiltered=unfiltered,
        eval_unfiltered=eval_unfiltered,
    )


def display_path(path: Path, root: Path) -> str:
    """Display repo-relative paths when possible."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def select_device(name: str) -> torch.device:
    """Select CPU, CUDA, or MPS by the standard round-script policy."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this process.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process.")
    return torch.device(name)


def default_worker_candidate(device: torch.device) -> int:
    """Return the standard worker candidate for accelerator checks."""

    return 4 if device.type in {"cuda", "mps"} else 0


def dataloader_kwargs(
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    train: bool,
) -> dict[str, Any]:
    """Common DataLoader kwargs without changing sampler/shuffle policy."""

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    if not train:
        kwargs["shuffle"] = False
    return kwargs


def add_phase_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard phase dispatcher CLI surface to a round script."""

    parser.add_argument(
        "--phase",
        choices=["equivalence", "train", "all"],
        default="train",
        help="equivalence=M2 checks only, train=CUDA training/eval, all=legacy combined flow",
    )
    parser.add_argument(
        "--equivalence-from",
        type=Path,
        default=None,
        help="JSON produced by --phase equivalence; included in train acceptance reports.",
    )
    parser.add_argument(
        "--min-epochs-for-acceptance",
        type=int,
        default=10,
        help="Skip acceptance/leak checks for smoke runs below this epoch count.",
    )


def load_equivalence_payload(path: Path | None) -> dict[str, Any]:
    """Load a handoff equivalence JSON or return a not-provided marker."""

    if path is None:
        return {"status": "not_provided"}
    import json

    with path.open() as f:
        payload = json.load(f)
    payload.setdefault("status", "provided")
    payload["path"] = str(path)
    return payload


def should_run_acceptance(
    *,
    phase: str,
    epochs_requested: int,
    min_epochs: int,
    ended_epoch: int,
    early_stopped: bool,
) -> bool:
    """Gate acceptance so short sanity runs cannot trip leak triggers."""

    if phase not in {"train", "all"}:
        return False
    if epochs_requested < min_epochs:
        return False
    return bool(early_stopped or ended_epoch >= epochs_requested)


def skipped_acceptance(reason: str = "skipped_smoke") -> dict[str, Any]:
    """Standard JSON payload for skipped smoke-run acceptance."""

    return {
        "acceptance": reason,
        "leak_triggered": None,
        "pass_rows": [],
    }
