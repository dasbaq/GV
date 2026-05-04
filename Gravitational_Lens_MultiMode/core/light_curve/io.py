"""HDF5 IO for time-delay extraction results."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _read_dataset(ds: h5py.Dataset, system_idx: int | None = None) -> np.ndarray:
    arr = ds[...]
    if system_idx is not None and arr.ndim > 1:
        arr = arr[system_idx]
    return np.asarray(arr)


def load_light_curve(h5_path: Path, system_idx: int | None = None) -> dict[str, Any]:
    """Load a light curve from the ARCHITECTURE.md ``light_curves`` group."""
    with h5py.File(h5_path, "r", swmr=True) as h5:
        group = h5["light_curves"]
        f_name = "F_joint" if "F_joint" in group else "F"
        F = _read_dataset(group[f_name], system_idx).astype(float)
        sigma = _read_dataset(group["sigma_noise"], system_idx).astype(float)
        t_obs = _read_dataset(group["t_obs"], system_idx).astype(float)
        if "n_epochs" in group:
            n_epochs_arr = _read_dataset(group["n_epochs"], system_idx)
            n_epochs = int(np.asarray(n_epochs_arr).reshape(-1)[0])
        else:
            n_epochs = int(F.size)
        F, sigma, t_obs = F[:n_epochs], sigma[:n_epochs], t_obs[:n_epochs]
        metadata: dict[str, Any] = {}
        if "metadata" in h5:
            metadata.update(dict(h5["metadata"].attrs))
            for key, value in h5["metadata"].items():
                metadata[key] = value[()].decode() if value.dtype.kind == "S" else value[()]
    return {
        "F": F,
        "sigma_noise": sigma,
        "t_obs": t_obs,
        "n_epochs": n_epochs,
        "metadata": metadata,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    data = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def save_extraction_results(
    h5_path: Path,
    results: list[dict[str, Any]],
    config: dict[str, Any],
    save_sigma_map: bool = True,
) -> None:
    """Save time-delay extraction results to HDF5."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    compression = config.get("io", {}).get("compression", "gzip")
    string_dtype = h5py.string_dtype("utf-8")

    with h5py.File(h5_path, "w", libver="latest") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["git_commit"] = _git_commit()
        meta.attrs["config_hash"] = _config_hash(config)
        meta.attrs["n_systems"] = len(results)

        grid = h5.create_group("grid")
        first_grid = results[0].get("grid", {}) if results else {}
        if isinstance(first_grid, dict) and "dt_grid" in first_grid:
            grid.create_dataset("dt_try", data=np.asarray(first_grid["dt_grid"]))
            grid.create_dataset("mu_try", data=np.asarray(first_grid["mu_grid"]))

        res = h5.create_group("results")
        res.create_dataset("dt_best", data=[r.get("dt", np.nan) for r in results])
        res.create_dataset("dt_uncertainty", data=[r.get("dt_uncertainty", np.nan) for r in results])
        res.create_dataset("mu_best", data=[r.get("mu", np.nan) for r in results])
        res.create_dataset("mu_uncertainty", data=[r.get("mu_uncertainty", np.nan) for r in results])
        res.create_dataset("sigma_min", data=[r.get("sigma_min", np.nan) for r in results])
        grades = [r.get("confidence_grade", "rejected") for r in results]
        res.create_dataset("confidence_grade", data=np.asarray(grades, dtype=object), dtype=string_dtype)

        if save_sigma_map and results and isinstance(first_grid, dict) and "sigma_map" in first_grid:
            maps = np.stack([r["grid"]["sigma_map"] for r in results])
            h5.create_dataset(
                "sigma_map",
                data=maps,
                compression=compression,
                compression_opts=4 if compression == "gzip" else None,
            )
        h5.swmr_mode = True
