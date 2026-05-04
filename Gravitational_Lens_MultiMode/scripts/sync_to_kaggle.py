"""Stage round data and print/run Kaggle Dataset upload commands.

Default mode is dry-run: it prints the files and Kaggle CLI command without
uploading. Use ``--execute`` to copy files into the staging directory and invoke
the Kaggle CLI. Kaggle authentication is the standard ``~/.kaggle/kaggle.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_slug(round_name: str) -> str:
    return f"lens-{round_name.replace('_', '-')}"


def default_owner() -> str:
    return os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME")


def round_files(round_name: str) -> list[Path]:
    files = [
        ROOT / "data" / "mock" / f"{round_name}.h5",
        ROOT / "data" / f"target_scaler_{round_name}.pkl",
    ]
    unfiltered = ROOT / "data" / "mock" / f"{round_name}_eval_unfiltered.h5"
    if unfiltered.exists():
        files.append(unfiltered)
    equivalence = ROOT / "data" / "logs" / f"{round_name}_equivalence.json"
    if equivalence.exists():
        files.append(equivalence)
    return files


def metadata(owner: str, slug: str, round_name: str) -> dict:
    return {
        "title": f"Gravitational Lens {round_name}",
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }


def command_for(args: argparse.Namespace, staging_dir: Path) -> list[str]:
    if args.init_dataset:
        return ["kaggle", "datasets", "create", "-p", str(staging_dir)]
    cmd = ["kaggle", "datasets", "version", "-p", str(staging_dir)]
    if args.version_notes:
        cmd.extend(["-m", args.version_notes])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="Round name, e.g. phase4_v0_1")
    parser.add_argument("--slug", default=None, help="Dataset slug, default lens-<round>")
    parser.add_argument("--owner", default=None, help="Kaggle owner; defaults to KAGGLE_USERNAME")
    parser.add_argument("--init-dataset", action="store_true", help="Use kaggle datasets create")
    parser.add_argument("--version-notes", default="", help="Version notes for kaggle datasets version")
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually copy files and run Kaggle CLI")
    args = parser.parse_args()

    slug = args.slug or default_slug(args.round)
    owner = args.owner or default_owner()
    staging_dir = args.staging_dir or Path(tempfile.gettempdir()) / f"kaggle_{slug}"
    files = round_files(args.round)
    missing = [p for p in files if not p.exists()]
    meta = metadata(owner, slug, args.round)
    cmd = command_for(args, staging_dir)

    plan = {
        "dry_run": not args.execute,
        "round": args.round,
        "dataset": meta["id"],
        "staging_dir": str(staging_dir),
        "files": [str(p) for p in files],
        "missing": [str(p) for p in missing],
        "metadata": meta,
        "command": cmd,
        "auth_note": "Kaggle CLI expects ~/.kaggle/kaggle.json; never commit this file.",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit("Missing required round files; aborting.")
    if not args.execute:
        return

    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, staging_dir / path.name)
    with (staging_dir / "dataset-metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
