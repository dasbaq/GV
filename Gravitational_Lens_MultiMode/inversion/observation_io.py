"""Observed lens-system input adapters for inversion modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from inversion.real_catalog import RealCatalogEntry, load_yaml_catalog

@dataclass(frozen=True)
class ObservedLightCurves:
    """Observed image light curves.

    Units: ``F`` is relative flux, ``t_obs`` is [days], and ``sigma_noise`` is
    flux uncertainty in the same units as ``F``. SIE 표준 근사 가정: this is an
    observation-side container only and contains no full_numerical truth
    information. Arrays are stored as shape ``(n_series, n_epoch)``; existing
    simulation HDF5 files with only ``F_joint`` are represented as one series.
    """

    F: np.ndarray
    t_obs: np.ndarray
    sigma_noise: np.ndarray


@dataclass(frozen=True)
class ObservedLensSystem:
    """Minimal observed lens-system input for Mode 1/2 inversion.

    Units: ``image_positions`` are [arcsec] with shape ``(n_img, 2)``;
    ``light_curves`` contains image or joint flux time series with ``t_obs``
    [days] and ``sigma_noise`` in flux units; redshifts are dimensionless;
    optional ``image_fluxes`` are relative image brightnesses. SIE 표준 근사
    가정: this container holds public observation-side inputs only and never
    stores truth-only full_numerical keys.
    """

    image_positions: np.ndarray
    light_curves: ObservedLightCurves
    z_lens: float
    z_source: float
    image_fluxes: np.ndarray | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable observation mapping.

        Units and SIE 표준 근사 assumptions are identical to
        :class:`ObservedLensSystem`.
        """

        data: dict[str, Any] = {
            "image_positions": self.image_positions.copy(),
            "light_curves": {
                "F": self.light_curves.F.copy(),
                "t_obs": self.light_curves.t_obs.copy(),
                "sigma_noise": self.light_curves.sigma_noise.copy(),
            },
            "z_lens": self.z_lens,
            "z_source": self.z_source,
        }
        if self.image_fluxes is not None:
            data["image_fluxes"] = self.image_fluxes.copy()
        if self.name is not None:
            data["name"] = self.name
        return data


def _as_float_array(value: Any, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{field_name} contains NaN or non-finite values")
    return arr


def _normalize_series(value: Any, field_name: str) -> np.ndarray:
    arr = _as_float_array(value, field_name)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{field_name} must have shape (n_series, n_epoch)")
    return arr


def _validate_system(system: ObservedLensSystem) -> ObservedLensSystem:
    positions = _as_float_array(system.image_positions, "image_positions")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("image_positions must have shape (n_img, 2) in arcsec")
    if positions.shape[0] < 2:
        raise ValueError("at least two image positions are required")

    z_lens = float(system.z_lens)
    z_source = float(system.z_source)
    if not np.isfinite([z_lens, z_source]).all():
        raise ValueError("z_lens and z_source must be finite")
    if not z_source > z_lens + 0.05:
        raise ValueError(
            f"z_source ({z_source}) must be greater than z_lens ({z_lens}) + 0.05"
        )

    F = _normalize_series(system.light_curves.F, "light_curves.F")
    t_obs = _normalize_series(system.light_curves.t_obs, "light_curves.t_obs")
    sigma = _normalize_series(
        system.light_curves.sigma_noise,
        "light_curves.sigma_noise",
    )
    if t_obs.shape[0] == 1 and F.shape[0] > 1:
        t_obs = np.repeat(t_obs, F.shape[0], axis=0)
    if sigma.shape[0] == 1 and F.shape[0] > 1:
        sigma = np.repeat(sigma, F.shape[0], axis=0)
    if F.shape != t_obs.shape or F.shape != sigma.shape:
        raise ValueError(
            "light_curves.F, light_curves.t_obs, and light_curves.sigma_noise "
            "must have matching shapes after single-series broadcasting"
        )
    if np.any(sigma < 0):
        raise ValueError("light_curves.sigma_noise must be non-negative")

    fluxes = None
    if system.image_fluxes is not None:
        fluxes = _as_float_array(system.image_fluxes, "image_fluxes")
        if fluxes.ndim != 1 or fluxes.shape[0] != positions.shape[0]:
            raise ValueError("image_fluxes must have shape (n_img,)")

    return ObservedLensSystem(
        image_positions=positions.astype(np.float32, copy=False),
        light_curves=ObservedLightCurves(
            F=F.astype(np.float32, copy=False),
            t_obs=t_obs.astype(np.float32, copy=False),
            sigma_noise=sigma.astype(np.float32, copy=False),
        ),
        z_lens=z_lens,
        z_source=z_source,
        image_fluxes=None if fluxes is None else fluxes.astype(np.float32, copy=False),
        name=None if system.name is None else str(system.name),
    )


def _light_curves_from_mapping(mapping: Mapping[str, Any]) -> ObservedLightCurves:
    group = mapping.get("light_curves")
    if not isinstance(group, Mapping):
        raise KeyError("light_curves mapping is required")
    f_value = group.get("F", group.get("F_joint"))
    if f_value is None:
        raise KeyError("light_curves must contain F or F_joint")
    return ObservedLightCurves(
        F=np.asarray(f_value, dtype=float),
        t_obs=np.asarray(group["t_obs"], dtype=float),
        sigma_noise=np.asarray(group["sigma_noise"], dtype=float),
    )


def from_dict(mapping: Mapping[str, Any]) -> ObservedLensSystem:
    """Build an observed system from a mapping.

    Units: ``image_positions`` [arcsec], ``light_curves.t_obs`` [days], fluxes
    and noise in relative flux units, redshifts dimensionless. SIE 표준 근사
    가정: only observation-side keys are consumed; truth-only keys
    ``M200``, ``concentration``, ``kappa_ext``, and ``nfw_offset`` are ignored.
    """

    system = ObservedLensSystem(
        image_positions=np.asarray(mapping["image_positions"], dtype=float),
        light_curves=_light_curves_from_mapping(mapping),
        z_lens=float(mapping["z_lens"]),
        z_source=float(mapping["z_source"]),
        image_fluxes=(
            None
            if mapping.get("image_fluxes") is None
            else np.asarray(mapping["image_fluxes"], dtype=float)
        ),
        name=None if mapping.get("name") is None else str(mapping["name"]),
    )
    return _validate_system(system)


def _read_indexed(group: h5py.Group, key: str, system_index: int) -> np.ndarray:
    if key not in group:
        raise KeyError(f"missing HDF5 dataset {group.name}/{key}")
    arr = np.asarray(group[key][...])
    if arr.ndim == 0:
        return arr
    return np.asarray(arr[system_index])


def _read_optional_indexed(
    group: h5py.Group,
    keys: tuple[str, ...],
    system_index: int,
) -> np.ndarray | None:
    for key in keys:
        if key in group:
            return _read_indexed(group, key, system_index)
    return None


def from_hdf5(path: str | Path, system_index: int = 0) -> ObservedLensSystem:
    """Load an observed system from a simulation-schema HDF5 file.

    Units: image positions are read from ``ray_paths`` in [arcsec], light-curve
    times from ``light_curves/t_obs`` in [days], and redshifts from ``params``.
    SIE 표준 근사 가정: this adapter reads only observation-side datasets
    required by inversion and deliberately does not read truth-only keys
    ``M200``, ``concentration``, ``kappa_ext``, or ``nfw_offset``.
    """

    h5_path = Path(path)
    with h5py.File(h5_path, "r") as h5:
        params = h5["params"]
        light_curves = h5["light_curves"]
        ray_paths = h5["ray_paths"]

        theta_1 = _read_indexed(ray_paths, "theta_1", system_index)
        theta_2 = _read_indexed(ray_paths, "theta_2", system_index)
        extra_positions = _read_optional_indexed(
            ray_paths,
            ("theta_extra", "theta_all", "image_positions"),
            system_index,
        )
        if extra_positions is None:
            image_positions = np.stack([theta_1, theta_2], axis=0)
        else:
            image_positions = np.asarray(extra_positions, dtype=float)

        f_key = "F_joint" if "F_joint" in light_curves else "F"
        F = _read_indexed(light_curves, f_key, system_index)
        sigma = _read_indexed(light_curves, "sigma_noise", system_index)
        t_obs = _read_indexed(light_curves, "t_obs", system_index)
        if "n_epochs" in light_curves:
            n_epochs = int(
                np.asarray(
                    _read_indexed(light_curves, "n_epochs", system_index)
                ).reshape(-1)[0]
            )
            F = np.asarray(F)[..., :n_epochs]
            sigma = np.asarray(sigma)[..., :n_epochs]
            t_obs = np.asarray(t_obs)[..., :n_epochs]

        image_fluxes = _read_optional_indexed(
            ray_paths,
            ("image_fluxes", "fluxes", "mu_images"),
            system_index,
        )
        if image_fluxes is not None:
            image_fluxes = np.asarray(image_fluxes, dtype=float)

        name = None
        if "metadata" in h5 and "name" in h5["metadata"].attrs:
            name = str(h5["metadata"].attrs["name"])
        elif "name" in params:
            raw_name = _read_indexed(params, "name", system_index)
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

        mapping = {
            "image_positions": image_positions,
            "light_curves": {
                "F": F,
                "t_obs": t_obs,
                "sigma_noise": sigma,
            },
            "z_lens": _read_indexed(params, "z_lens", system_index),
            "z_source": _read_indexed(params, "z_source", system_index),
            "image_fluxes": image_fluxes,
            "name": name,
        }
    return from_dict(mapping)
