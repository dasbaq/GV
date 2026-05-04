"""Flat Lambda-CDM distance helpers for analytic sanity checks."""

from __future__ import annotations

import numpy as np

from core.physics.config import constants, default_cosmology, numerics


def _cosmo(cosmology: dict[str, float] | None = None) -> dict[str, float]:
    merged = default_cosmology().copy()
    if cosmology:
        merged.update(cosmology)
    return merged


def E_z(z: float | np.ndarray, cosmology: dict[str, float] | None = None) -> float | np.ndarray:
    """Dimensionless expansion rate for flat Lambda-CDM.

    Args:
        z: redshift, dimensionless.
        cosmology: optional ``H0`` [km s^-1 Mpc^-1], ``Omega_m``,
            ``Omega_lambda``.

    Returns:
        ``sqrt(Omega_m (1+z)^3 + Omega_lambda)``, dimensionless.

    Calculation assumption:
        Auxiliary cosmology helper for observation conversion and thin-lens
        sanity checks, not the primary n_eff ray-tracing calculation.
    """
    c = _cosmo(cosmology)
    out = np.sqrt(c["Omega_m"] * (1.0 + np.asarray(z)) ** 3 + c["Omega_lambda"])
    return float(out) if np.ndim(z) == 0 else out


def _integral_0_z(z: float, cosmology: dict[str, float]) -> float:
    if z < 0:
        raise ValueError("redshift z must be non-negative")
    try:
        from scipy.integrate import quad

        return float(quad(lambda zz: 1.0 / E_z(zz, cosmology), 0.0, z, epsrel=1e-8)[0])
    except Exception:
        n = int(numerics().get("integration_n", 4096))
        grid = np.linspace(0.0, z, max(n, 2))
        return float(np.trapz(1.0 / E_z(grid, cosmology), grid))


def comoving_distance(z: float, cosmology: dict[str, float] | None = None) -> float:
    """Comoving distance from redshift zero to ``z``.

    Args:
        z: redshift, dimensionless.
        cosmology: optional flat Lambda-CDM parameters.

    Returns:
        Comoving distance in Mpc.

    Calculation assumption:
        Auxiliary flat Lambda-CDM helper for analytic comparisons.
    """
    cosmo = _cosmo(cosmology)
    return constants()["c_km_s"] / cosmo["H0"] * _integral_0_z(float(z), cosmo)


def angular_diameter_distance(z: float, cosmology: dict[str, float] | None = None) -> float:
    """Angular-diameter distance from observer to redshift ``z`` in Mpc."""
    return comoving_distance(z, cosmology) / (1.0 + float(z))


def angular_diameter_distance_between(
    z_lens: float,
    z_source: float,
    cosmology: dict[str, float] | None = None,
) -> float:
    """Angular-diameter distance between lens and source in Mpc.

    Raises:
        ValueError: if ``z_source <= z_lens``.
    """
    if z_source <= z_lens:
        raise ValueError("z_source must be greater than z_lens")
    dc_l = comoving_distance(z_lens, cosmology)
    dc_s = comoving_distance(z_source, cosmology)
    return (dc_s - dc_l) / (1.0 + z_source)


def time_delay_distance(
    z_lens: float,
    z_source: float,
    cosmology: dict[str, float] | None = None,
) -> float:
    """Time-delay distance ``D_dt`` in Mpc.

    Definition:
        ``D_dt = (1 + z_lens) * D_lens * D_source / D_lens_source``.
        The ``(1 + z_lens)`` factor is included here; callers must not multiply
        it a second time.

    Calculation assumption:
        Auxiliary thin-lens sanity-check distance, not the primary n_eff path
        integration equation.
    """
    d_l = angular_diameter_distance(z_lens, cosmology)
    d_s = angular_diameter_distance(z_source, cosmology)
    d_ls = angular_diameter_distance_between(z_lens, z_source, cosmology)
    return (1.0 + z_lens) * d_l * d_s / d_ls
