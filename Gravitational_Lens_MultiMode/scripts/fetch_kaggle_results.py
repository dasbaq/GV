"""Fetch or unpack Kaggle round outputs into local artifact directories.

Default mode is dry-run. It prints the Kaggle CLI command or zip routing plan
without writing files. Use ``--execute`` to download/extract and copy artifacts.
Existing local files are never overwritten; a timestamp suffix is inserted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dest.with_name(f"{dest.stem}_{stamp}{dest.suffix}")


def target_for(path: Path) -> Path | None:
    suffix = path.suffix.lower()
    name = path.name
    if suffix == ".pt":
        return ROOT / "data" / "checkpoints" / name
    if suffix == ".json":
        return ROOT / "data" / "logs" / name
    return None


def route_files(source_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        target = target_for(path)
        if target is None:
            continue
        rows.append({"source": str(path), "dest": str(unique_dest(target))})
    return rows


def extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--notebook", help="Kaggle kernel slug, e.g. owner/notebook")
    src.add_argument("--from-zip", type=Path, help="Downloaded Kaggle output zip")
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually download/extract and copy outputs")
    args = parser.parse_args()

    download_dir = args.download_dir or Path(tempfile.gettempdir()) / "kaggle_round_outputs"
    command = None
    if args.notebook:
        command = ["kaggle", "kernels", "output", args.notebook, "-p", str(download_dir)]

    plan = {
        "dry_run": not args.execute,
        "notebook": args.notebook,
        "from_zip": str(args.from_zip) if args.from_zip else None,
        "download_dir": str(download_dir),
        "command": command,
        "routes": [],
        "overwrite_policy": "never overwrite; append timestamp suffix on collision",
    }

    if not args.execute:
        if args.from_zip and args.from_zip.exists():
            tmp = Path(tempfile.mkdtemp(prefix="kaggle_zip_plan_"))
            extract_zip(args.from_zip, tmp)
            plan["routes"] = route_files(tmp)
            shutil.rmtree(tmp)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    download_dir.mkdir(parents=True, exist_ok=True)
    if command is not None:
        subprocess.run(command, check=True)
        source_dir = download_dir
    else:
        source_dir = Path(tempfile.mkdtemp(prefix="kaggle_zip_extract_"))
        extract_zip(args.from_zip, source_dir)

    routes = route_files(source_dir)
    for row in routes:
        dest = Path(row["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], dest)
    plan["routes"] = routes
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
