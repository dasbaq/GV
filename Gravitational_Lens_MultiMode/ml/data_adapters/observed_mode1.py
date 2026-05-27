"""Raw observed Mode 1 light-curve ingestion.

This module converts CSV/TSV/RDB light curves plus a YAML manifest into the
project's observation-side HDF5 input for Mode 1. Units: manifest image
positions are [arcsec], light-curve times are [days], magnitudes are converted
to linear relative flux, and flux uncertainties are stored in the same units as
``F``. SIE 표준 근사 가정: the generated HDF5 contains observation-side inputs
only; reference Δt/H0 values are intentionally kept in external sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class SidecarReferences:
    """Reference values used only for validation reports, never HDF5 inputs."""

    dt_ref_days: float | None = None
    dt_ref_sigma_days: float | None = None
    H0_ref: float | None = None
    H0_ref_sigma: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in {
                "dt_ref_days": self.dt_ref_days,
                "dt_ref_sigma_days": self.dt_ref_sigma_days,
                "H0_ref": self.H0_ref,
                "H0_ref_sigma": self.H0_ref_sigma,
            }.items()
            if value is not None
        }


def _load_yaml_or_json(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def load_sidecar(path: str | Path) -> SidecarReferences:
    """Load validation-only reference values from YAML or JSON.

    Units: ``dt_ref_days`` and ``dt_ref_sigma_days`` are [days], H0 fields are
    [km/s/Mpc]. SIE 표준 근사 가정: sidecar values are external benchmark
    references and must not be copied into observation HDF5 inputs.
    """

    data = _load_yaml_or_json(path)
    return SidecarReferences(
        dt_ref_days=_optional_float(data, "dt_ref_days"),
        dt_ref_sigma_days=_optional_float(data, "dt_ref_sigma_days"),
        H0_ref=_optional_float(data, "H0_ref"),
        H0_ref_sigma=_optional_float(data, "H0_ref_sigma"),
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a Mode 1 observation manifest.

    Required fields: ``z_lens``, ``z_source``, ``image_positions``, and a column
    mapping under ``columns`` or ``column_mapping``. Units: image positions are
    [arcsec]. SIE 표준 근사 가정: manifest lens metadata feeds the fixed SIE
    inversion path and contains no truth-only full_numerical parameters.
    """

    manifest = _load_yaml_or_json(path)
    _require_keys(manifest, ("z_lens", "z_source", "image_positions"))
    if "columns" not in manifest and "column_mapping" not in manifest:
        raise ValueError("manifest must contain 'columns' or 'column_mapping'")
    return manifest


def read_light_curve_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, TSV, whitespace, or COSMOGRAIL-style RDB light curves.

    Units are not interpreted here. SIE 표준 근사 가정: this is raw observation
    parsing only and does not use lens-model or benchmark truth information.
    """

    table_path = Path(path)
    text = table_path.read_text(encoding="utf-8")
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if tokens and all(set(token) <= {"-"} for token in tokens):
            continue
        lines.append(raw)
    if not lines:
        raise ValueError(f"{table_path} contains no data rows")

    cleaned = "\n".join(lines)
    suffix = table_path.suffix.lower()
    if suffix == ".tsv":
        sep: str | None = "\t"
    elif suffix == ".rdb":
        sep = r"\s+"
    else:
        sep = None
    try:
        from io import StringIO

        return pd.read_csv(StringIO(cleaned), sep=sep, engine="python")
    except pd.errors.ParserError:
        from io import StringIO

        return pd.read_csv(StringIO(cleaned), sep=r"\s+", engine="python")


def magnitude_to_flux(mag: np.ndarray, mag_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert magnitudes to linear relative flux and propagated uncertainty.

    Units: ``mag`` and ``mag_err`` are astronomical magnitudes; returned flux is
    relative linear flux. SIE 표준 근사 가정: photometric conversion is an
    observation-side preprocessing step independent of the lens model.
    """

    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)
    flux = np.power(10.0, -0.4 * mag)
    sigma = 0.4 * np.log(10.0) * flux * mag_err
    return flux, sigma


def ingest_observed_mode1(
    light_curve_path: str | Path,
    manifest_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Convert raw observed Mode 1 inputs into observation HDF5.

    Units: output ``light_curves/t_obs`` are [days], ``ray_paths/theta_1`` and
    ``theta_2`` are [arcsec], and redshifts are dimensionless. SIE 표준 근사
    가정: output HDF5 contains only public observation-side inputs required by
    Mode 1; sidecar references are excluded to prevent truth leakage.
    """

    manifest = load_manifest(manifest_path)
    df = read_light_curve_table(light_curve_path)
    t, flux, sigma = build_light_curve_arrays(df, manifest)
    image_positions = _image_positions(manifest)
    z_lens = _finite_float(manifest["z_lens"], "z_lens")
    z_source = _finite_float(manifest["z_source"], "z_source")
    if not z_source > z_lens + 0.05:
        raise ValueError(f"z_source ({z_source}) must be greater than z_lens ({z_lens}) + 0.05")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = str(manifest.get("name", out.stem))
    source = str(manifest.get("source", manifest.get("survey", "observed_mode1")))
    with h5py.File(out, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["source"] = source
        meta.attrs["name"] = name
        meta.attrs["n_systems"] = 1
        meta.attrs["has_ground_truth"] = False
        meta.attrs["mock"] = False
        meta.attrs["synthetic"] = False
        meta.attrs["input_light_curves"] = str(Path(light_curve_path))
        meta.attrs["input_manifest"] = str(Path(manifest_path))

        params = h5.create_group("params")
        params.create_dataset("z_lens", data=np.array([z_lens], dtype=np.float32))
        params.create_dataset("z_source", data=np.array([z_source], dtype=np.float32))

        lc = h5.create_group("light_curves")
        lc.create_dataset("F", data=flux[None, :, :].astype(np.float32))
        lc.create_dataset("sigma_noise", data=sigma[None, :, :].astype(np.float32))
        lc.create_dataset("t_obs", data=t[None, :, :].astype(np.float32))
        lc.create_dataset("n_epochs", data=np.array([t.shape[1]], dtype=np.int32))

        rays = h5.create_group("ray_paths")
        rays.create_dataset("theta_1", data=image_positions[0:1].astype(np.float32))
        rays.create_dataset("theta_2", data=image_positions[1:2].astype(np.float32))
        rays.create_dataset("image_positions", data=image_positions[None, :, :].astype(np.float32))
        if manifest.get("image_fluxes") is not None:
            rays.create_dataset("image_fluxes", data=_image_fluxes(manifest)[None, :].astype(np.float32))

    return out


def build_light_curve_arrays(
    df: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(t_obs, F, sigma_noise)`` arrays from a manifest column mapping.

    Units: returned time is [days] after optional zero-point normalization;
    flux is linear relative flux. SIE 표준 근사 가정: the A/B series are
    observation inputs for downstream fixed-SIE inversion and include no truth
    labels.
    """

    columns = manifest.get("columns", manifest.get("column_mapping"))
    if not isinstance(columns, Mapping):
        raise ValueError("manifest columns must be a mapping")
    time_col = str(columns.get("time", columns.get("t_obs", "")))
    if not time_col:
        raise ValueError("manifest columns must define 'time' or 't_obs'")
    _require_dataframe_columns(df, [time_col])

    series_cfg = columns.get("series", columns.get("images"))
    if series_cfg is None:
        series_cfg = _series_from_legacy_columns(columns)
    if not isinstance(series_cfg, list) or len(series_cfg) < 2:
        raise ValueError("manifest must define at least two image series")

    t_base = np.asarray(df[time_col].to_numpy(), dtype=float)
    if bool(manifest.get("time_zero_at_first", True)):
        t_base = t_base - float(np.nanmin(t_base))
    if not np.isfinite(t_base).all():
        raise ValueError("time column contains NaN or non-finite values")

    order = np.argsort(t_base)
    t_sorted = t_base[order]
    if np.unique(t_sorted).size != t_sorted.size:
        raise ValueError("time grid contains duplicate epochs")

    flux_series: list[np.ndarray] = []
    sigma_series: list[np.ndarray] = []
    for item in series_cfg:
        if not isinstance(item, Mapping):
            raise ValueError("each series mapping must be an object")
        item_time_col = str(item.get("time", time_col))
        _require_dataframe_columns(df, [item_time_col])
        item_time = np.asarray(df[item_time_col].to_numpy(), dtype=float)
        if bool(manifest.get("time_zero_at_first", True)):
            item_time = item_time - float(np.nanmin(item_time))
        if not np.allclose(item_time, t_base, rtol=0.0, atol=float(manifest.get("time_grid_atol_days", 1.0e-8))):
            raise ValueError("multi-image light curves must share the same t_obs grid")
        flux_unit = str(item.get("unit", columns.get("flux_unit", "magnitude"))).lower()
        if flux_unit in {"mag", "magnitude", "magnitudes"}:
            mag_col = str(item.get("mag", item.get("magnitude", "")))
            err_col = str(item.get("mag_err", item.get("magerr", item.get("magnitude_err", ""))))
            if not mag_col or not err_col:
                raise ValueError("magnitude series must define 'mag' and 'mag_err' columns")
            _require_dataframe_columns(df, [mag_col, err_col])
            flux, sigma = magnitude_to_flux(df[mag_col].to_numpy(), df[err_col].to_numpy())
        elif flux_unit in {"flux", "linear_flux", "relative_flux"}:
            flux_col = str(item.get("flux", item.get("F", "")))
            err_col = str(item.get("flux_err", item.get("sigma", item.get("sigma_noise", ""))))
            if not flux_col or not err_col:
                raise ValueError("flux series must define 'flux' and 'flux_err' columns")
            _require_dataframe_columns(df, [flux_col, err_col])
            flux = np.asarray(df[flux_col].to_numpy(), dtype=float)
            sigma = np.asarray(df[err_col].to_numpy(), dtype=float)
        else:
            raise ValueError(f"unsupported flux unit: {flux_unit}")
        if not np.isfinite(flux).all() or not np.isfinite(sigma).all():
            raise ValueError("flux columns contain NaN or non-finite values")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma_noise must be positive")
        flux_series.append(np.asarray(flux[order], dtype=float))
        sigma_series.append(np.asarray(sigma[order], dtype=float))

    t = np.repeat(t_sorted[None, :], len(flux_series), axis=0)
    return t, np.stack(flux_series, axis=0), np.stack(sigma_series, axis=0)


def validation_report(
    *,
    output_hdf5: str | Path,
    light_curve_path: str | Path,
    manifest_path: str | Path,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a leak-free ingestion report.

    Units: sidecar reference fields retain their documented units. SIE 표준
    근사 가정: the report may include benchmark references, but those values
    are not written into the observation HDF5 consumed by Mode 1.
    """

    report: dict[str, Any] = {
        "output_hdf5": str(output_hdf5),
        "light_curves": str(light_curve_path),
        "manifest": str(manifest_path),
        "sidecar_used": sidecar_path is not None,
    }
    if sidecar_path is not None:
        report["references"] = load_sidecar(sidecar_path).to_dict()
        report["sidecar"] = str(sidecar_path)
    return report


def _series_from_legacy_columns(columns: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "A_mag" in columns and "B_mag" in columns:
        return [
            {"name": "A", "unit": "magnitude", "mag": columns["A_mag"], "mag_err": columns["A_mag_err"]},
            {"name": "B", "unit": "magnitude", "mag": columns["B_mag"], "mag_err": columns["B_mag_err"]},
        ]
    if "A_flux" in columns and "B_flux" in columns:
        return [
            {"name": "A", "unit": "flux", "flux": columns["A_flux"], "flux_err": columns["A_flux_err"]},
            {"name": "B", "unit": "flux", "flux": columns["B_flux"], "flux_err": columns["B_flux_err"]},
        ]
    raise ValueError("manifest columns must define 'series' or legacy A/B mappings")


def _image_positions(manifest: Mapping[str, Any]) -> np.ndarray:
    positions = np.asarray(manifest["image_positions"], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError("image_positions must have shape (n_img>=2, 2) in arcsec")
    if not np.isfinite(positions).all():
        raise ValueError("image_positions contains NaN or non-finite values")
    return positions


def _image_fluxes(manifest: Mapping[str, Any]) -> np.ndarray:
    fluxes = np.asarray(manifest["image_fluxes"], dtype=float)
    positions = _image_positions(manifest)
    if fluxes.ndim != 1 or fluxes.shape[0] != positions.shape[0]:
        raise ValueError("image_fluxes must have shape (n_img,)")
    if not np.isfinite(fluxes).all():
        raise ValueError("image_fluxes contains NaN or non-finite values")
    return fluxes


def _optional_float(data: Mapping[str, Any], key: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    return _finite_float(data[key], key)


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")


def _require_dataframe_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"light curve table missing columns: {missing}")
