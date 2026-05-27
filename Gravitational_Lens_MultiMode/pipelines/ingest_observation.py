"""CLI for raw observed Mode 1 ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.data_adapters.observed_mode1 import (
    ingest_observed_mode1,
    validation_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _default_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_ingest_report.json")


def ingest_observation(
    *,
    light_curves: str | Path,
    manifest: str | Path,
    output: str | Path,
    sidecar: str | Path | None = None,
    report_output: str | Path | None = None,
) -> dict[str, Any]:
    """Ingest raw observed light curves into Mode 1 observation HDF5.

    Units and SIE 표준 근사 assumptions follow
    :func:`ml.data_adapters.observed_mode1.ingest_observed_mode1`; sidecar
    reference values are written only to the report JSON, never the HDF5.
    """

    output_path = ingest_observed_mode1(light_curves, manifest, output)
    report = validation_report(
        output_hdf5=output_path,
        light_curve_path=light_curves,
        manifest_path=manifest,
        sidecar_path=sidecar,
    )
    report_path = Path(report_output) if report_output is not None else _default_report_path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_output"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Ingest raw observed Mode 1 data into HDF5")
    parser.add_argument("--light-curves", required=True, help="CSV/TSV/RDB light-curve table")
    parser.add_argument("--manifest", required=True, help="YAML manifest with columns, positions, and redshifts")
    parser.add_argument("--sidecar", help="Optional YAML/JSON validation-only reference sidecar")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "observations" / "observed_mode1.h5"),
        help="Output observation HDF5 path",
    )
    parser.add_argument("--report-output", help="Optional ingestion report JSON path")
    args = parser.parse_args(argv)

    report = ingest_observation(
        light_curves=args.light_curves,
        manifest=args.manifest,
        output=args.output,
        sidecar=args.sidecar,
        report_output=args.report_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
