"""CLI for Phase 1 time-delay extraction."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from core.light_curve.io import load_light_curve, save_extraction_results
from core.light_curve.time_delay import extract_time_delay


def _load_cfg(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _plot(result: dict[str, Any], plot_dir: Path, stem: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    grid = result.get("grid", {})
    sigma = grid["sigma_map"]
    dt_grid = grid["dt_grid"]
    mu_grid = grid["mu_grid"]
    plt.figure(figsize=(8, 5))
    plt.imshow(
        sigma.T,
        origin="lower",
        aspect="auto",
        extent=[dt_grid[0], dt_grid[-1], mu_grid[0], mu_grid[-1]],
        cmap="viridis",
    )
    plt.colorbar(label="Sigma")
    plt.xlabel("Delta t [days]")
    plt.ylabel("mu")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_sigma_map.png", dpi=150)
    plt.close()

    mu_idx = int(abs(mu_grid - result.get("mu", 0.0)).argmin())
    plt.figure(figsize=(8, 4))
    plt.plot(dt_grid, sigma[:, mu_idx])
    plt.xlabel("Delta t [days]")
    plt.ylabel("Sigma")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_sigma_dt_slice.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--system-idx", type=int)
    parser.add_argument("--config", default="config/time_delay.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--save-sigma-map", action="store_true")
    parser.add_argument("--plot-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    inputs = [Path(p) for p in glob.glob(args.input)] or [Path(args.input)]
    results = []
    for path in tqdm(inputs, desc="extract_td"):
        indices = [args.system_idx]
        if args.system_idx is None:
            indices = [0]
        for idx in indices:
            lc = load_light_curve(path, idx)
            res = extract_time_delay(
                lc["F"],
                lc["t_obs"],
                lc["sigma_noise"],
                cfg,
                return_grid=args.save_sigma_map or bool(args.plot_dir),
            )
            results.append(res)
            if args.plot_dir:
                _plot(res, Path(args.plot_dir), f"{path.stem}_system{idx}")
            if args.dry_run:
                break
        if args.dry_run:
            break

    output = Path(args.output)
    if output.suffix != ".h5":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "td_extraction.h5"
    save_extraction_results(output, results, cfg, save_sigma_map=args.save_sigma_map)


if __name__ == "__main__":
    main()
