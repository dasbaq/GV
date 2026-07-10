"""Lens mass models that provide Phi and n_eff fields for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.physics.config import constants, default_cosmology, numerics
from core.physics.distances import (
    angular_diameter_distance,
    angular_diameter_distance_between,
    time_delay_distance,
)
from core.physics.refractive_index import (
    effective_refractive_index as phi_to_neff,
    grad_refractive_index_from_grad_phi,
)


def _as_position(position: np.ndarray) -> np.ndarray:
    arr = np.asarray(position, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("position must have shape [..., 3] in meters")
    return arr


def _theta_rad(theta_arcsec: np.ndarray) -> np.ndarray:
    return np.asarray(theta_arcsec, dtype=float) * constants()["arcsec_to_rad"]


def _theta_arcsec(theta_rad: np.ndarray) -> np.ndarray:
    return np.asarray(theta_rad, dtype=float) * constants()["rad_to_arcsec"]


def _rotate_xy(xy: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s], [-s, c]])
    return xy @ rot.T


@dataclass
class BaseLens:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    softening_radius_m: float | None = None
    z_lens: float | None = None
    z_source: float | None = None
    cosmology: dict[str, float] | None = None
    _eps: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.center_arr = np.asarray(self.center, dtype=float)
        self._eps = float(numerics().get("eps", 1.0e-12))
        if self.softening_radius_m is None:
            self.softening_radius_m = 1.0e16

    def _relative(self, position: np.ndarray) -> np.ndarray:
        return _as_position(position) - self.center_arr

    def effective_refractive_index(self, position: np.ndarray) -> np.ndarray:
        """Evaluate n_eff at positions [m], returning dimensionless values."""
        return phi_to_neff(self.potential_3d(position))

    def grad_refractive_index(self, position: np.ndarray) -> np.ndarray:
        """Evaluate ``nabla n_eff`` [m^-1] at positions [m]."""
        return grad_refractive_index_from_grad_phi(self.grad_potential_3d(position))

    def _require_redshifts(self) -> tuple[float, float]:
        if self.z_lens is None or self.z_source is None:
            raise ValueError("z_lens and z_source are required for analytic sanity checks")
        return float(self.z_lens), float(self.z_source)

    def potential_3d(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_potential_3d(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SISLens(BaseLens):
    """Singular isothermal sphere provider for Phi and n_eff fields.

    The 3D potential is a computational effective-potential approximation for
    n_eff ray tracing. Inputs are positions in meters and outputs are Phi
    [m^2 s^-2] or gradients [m s^-2]. Thin-lens helpers are analytic sanity
    checks only.
    """

    sigma_v: float = 220.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sigma_v <= 0:
            raise ValueError("sigma_v must be positive [km/s]")
        self.sigma_m_s = self.sigma_v * 1000.0

    def potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return SIS effective Phi [m^2 s^-2] for positions [m]."""
        rel = self._relative(position)
        r = np.sqrt(np.sum(rel * rel, axis=-1) + float(self.softening_radius_m) ** 2)
        return 2.0 * self.sigma_m_s**2 * np.log(r / float(self.softening_radius_m))

    def grad_potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return SIS ``nabla Phi`` [m s^-2] for positions [m]."""
        rel = self._relative(position)
        r2 = np.sum(rel * rel, axis=-1, keepdims=True) + float(self.softening_radius_m) ** 2
        return 2.0 * self.sigma_m_s**2 * rel / np.maximum(r2, self._eps)

    def einstein_radius(self) -> float:
        """Analytic sanity-check Einstein radius in arcsec."""
        z_l, z_s = self._require_redshifts()
        d_s = angular_diameter_distance(z_s, self.cosmology)
        d_ls = angular_diameter_distance_between(z_l, z_s, self.cosmology)
        theta = 4.0 * np.pi * (self.sigma_m_s / constants()["c_m_s"]) ** 2 * d_ls / d_s
        return float(_theta_arcsec(theta))

    def deflection(self, theta: np.ndarray) -> np.ndarray:
        """Thin-lens analytic sanity-check deflection [arcsec]."""
        th = np.asarray(theta, dtype=float)
        r = np.linalg.norm(th, axis=-1, keepdims=True)
        return self.einstein_radius() * th / np.maximum(r, self._eps)

    def fermat_potential(self, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Thin-lens sanity-check Fermat potential [radian^2]."""
        th = _theta_rad(theta)
        be = _theta_rad(beta)
        theta_e = self.einstein_radius() * constants()["arcsec_to_rad"]
        return 0.5 * np.sum((th - be) ** 2, axis=-1) - theta_e * np.linalg.norm(th, axis=-1)

    def analytic_time_delay(
        self,
        theta_a: np.ndarray,
        theta_b: np.ndarray,
        beta: np.ndarray,
        z_lens: float,
        z_source: float,
        cosmology: dict[str, float] | None = None,
    ) -> float:
        """Thin-lens analytic sanity-check time delay in days."""
        ddt_m = time_delay_distance(z_lens, z_source, cosmology or self.cosmology) * constants()["Mpc_m"]
        dphi = self.fermat_potential(theta_a, beta) - self.fermat_potential(theta_b, beta)
        return float(ddt_m / constants()["c_m_s"] * dphi / constants()["day_s"])


@dataclass
class SIELens(SISLens):
    """SIE standard-approximation lens field provider.

    SIE 표준 근사 가정: 단일 평면, κ_ext=0, smooth mass profile, isotropic
    velocity dispersion. Phase 2 uses an elliptical effective-potential
    approximation for Phi and n_eff ray tracing; the public API is intentionally
    replaceable by a more exact SIE closed form later. Inputs are positions [m],
    outputs are Phi [m^2 s^-2], gradients [m s^-2], n_eff dimensionless, and
    ``nabla n_eff`` [m^-1]. Thin-lens methods are sanity checks only.
    """

    q: float = 0.8
    position_angle: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.q <= 1.0):
            raise ValueError("SIELens q must satisfy 0 < q <= 1")

    def potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return elliptical effective Phi [m^2 s^-2] under SIE standard approximation."""
        rel = self._relative(position)
        xy = _rotate_xy(rel[..., :2], self.position_angle)
        ell2 = (self.q * xy[..., 0]) ** 2 + (xy[..., 1] / self.q) ** 2 + rel[..., 2] ** 2
        r = np.sqrt(ell2 + float(self.softening_radius_m) ** 2)
        return 2.0 * self.sigma_m_s**2 * np.log(r / float(self.softening_radius_m))

    def grad_potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return elliptical effective ``nabla Phi`` [m s^-2] for positions [m]."""
        rel = self._relative(position)
        xy = _rotate_xy(rel[..., :2], self.position_angle)
        ell2 = (self.q * xy[..., 0]) ** 2 + (xy[..., 1] / self.q) ** 2 + rel[..., 2] ** 2
        denom = ell2[..., None] + float(self.softening_radius_m) ** 2
        grad_rot = np.empty_like(rel)
        grad_rot[..., 0] = 2.0 * self.sigma_m_s**2 * self.q**2 * xy[..., 0] / np.maximum(denom[..., 0], self._eps)
        grad_rot[..., 1] = 2.0 * self.sigma_m_s**2 * xy[..., 1] / (self.q**2 * np.maximum(denom[..., 0], self._eps))
        grad_rot[..., 2] = 2.0 * self.sigma_m_s**2 * rel[..., 2] / np.maximum(denom[..., 0], self._eps)
        inv_xy = _rotate_xy(grad_rot[..., :2], -self.position_angle)
        return np.concatenate([inv_xy, grad_rot[..., 2:3]], axis=-1)

    def deflection(self, theta: np.ndarray) -> np.ndarray:
        """Thin-lens sanity-check SIE-like deflection [arcsec]."""
        th = np.asarray(theta, dtype=float)
        xy = _rotate_xy(th, self.position_angle)
        ell = np.sqrt(self.q * xy[..., 0:1] ** 2 + xy[..., 1:2] ** 2 / self.q)
        alpha_rot = self.einstein_radius() * np.concatenate(
            [self.q * xy[..., 0:1], xy[..., 1:2] / self.q], axis=-1
        ) / np.maximum(ell, self._eps)
        return _rotate_xy(alpha_rot, -self.position_angle)

    def fermat_potential(self, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Thin-lens SIE-like Fermat potential [radian^2].

        Units: ``theta`` and ``beta`` are [arcsec], return is [rad²]. SIE 표준
        근사 가정: this uses the same elliptical potential whose angular
        gradient gives ``SIELens.deflection``. For q=1 it reduces to the SIS
        expression.
        """
        th = _theta_rad(theta)
        be = _theta_rad(beta)
        xy = _rotate_xy(th, self.position_angle)
        theta_e = self.einstein_radius() * constants()["arcsec_to_rad"]
        psi = theta_e * np.sqrt(
            self.q * xy[..., 0] ** 2 + xy[..., 1] ** 2 / self.q
        )
        return 0.5 * np.sum((th - be) ** 2, axis=-1) - psi


@dataclass
class PointMassLens(BaseLens):
    """Point-mass lens provider for Phi and n_eff fields.

    Inputs are positions [m]. ``potential_3d`` returns Phi [m^2 s^-2] and
    ``grad_potential_3d`` returns ``nabla Phi`` [m s^-2]. Thin-lens helpers are
    analytic sanity checks only.
    """

    mass_msun: float = 1.0e11

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mass_msun <= 0:
            raise ValueError("mass_msun must be positive")
        self.mass_kg = self.mass_msun * constants()["M_sun_kg"]

    def potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return softened point-mass Phi [m^2 s^-2] for positions [m]."""
        rel = self._relative(position)
        r = np.sqrt(np.sum(rel * rel, axis=-1) + float(self.softening_radius_m) ** 2)
        return -constants()["G_si"] * self.mass_kg / np.maximum(r, self._eps)

    def grad_potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return softened point-mass ``nabla Phi`` [m s^-2]."""
        rel = self._relative(position)
        r2 = np.sum(rel * rel, axis=-1, keepdims=True) + float(self.softening_radius_m) ** 2
        return constants()["G_si"] * self.mass_kg * rel / np.maximum(r2, self._eps) ** 1.5

    def einstein_radius(self) -> float:
        """Analytic sanity-check Einstein radius in arcsec."""
        z_l, z_s = self._require_redshifts()
        d_l = angular_diameter_distance(z_l, self.cosmology) * constants()["Mpc_m"]
        d_s = angular_diameter_distance(z_s, self.cosmology) * constants()["Mpc_m"]
        d_ls = angular_diameter_distance_between(z_l, z_s, self.cosmology) * constants()["Mpc_m"]
        theta = np.sqrt(4.0 * constants()["G_si"] * self.mass_kg / constants()["c_m_s"] ** 2 * d_ls / (d_l * d_s))
        return float(_theta_arcsec(theta))

    def deflection(self, theta: np.ndarray) -> np.ndarray:
        """Thin-lens analytic sanity-check deflection [arcsec]."""
        th = np.asarray(theta, dtype=float)
        r2 = np.sum(th * th, axis=-1, keepdims=True)
        return self.einstein_radius() ** 2 * th / np.maximum(r2, self._eps)

    def fermat_potential(self, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Thin-lens point-mass sanity-check Fermat potential [radian^2]."""
        th = _theta_rad(theta)
        be = _theta_rad(beta)
        theta_e = self.einstein_radius() * constants()["arcsec_to_rad"]
        r = np.linalg.norm(th, axis=-1)
        return 0.5 * np.sum((th - be) ** 2, axis=-1) - theta_e**2 * np.log(np.maximum(r, self._eps))

    def analytic_time_delay(self, theta_a: np.ndarray, theta_b: np.ndarray, beta: np.ndarray, z_lens: float, z_source: float, cosmology: dict[str, float] | None = None) -> float:
        """Thin-lens analytic sanity-check time delay in days."""
        ddt_m = time_delay_distance(z_lens, z_source, cosmology or self.cosmology) * constants()["Mpc_m"]
        dphi = self.fermat_potential(theta_a, beta) - self.fermat_potential(theta_b, beta)
        return float(ddt_m / constants()["c_m_s"] * dphi / constants()["day_s"])


@dataclass
class NFWLens(BaseLens):
    """NFW halo model for full numerical truth or comparison halo model use.

    Inputs are positions [m]. When initialized, ``r200``, ``rs`` and ``rho_s``
    are computed for flat Lambda-CDM. This model can support full numerical
    truth generation or comparison halo model workflows.
    """

    M200: float = 1.0e12
    concentration: float = 8.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.M200 <= 0:
            raise ValueError("M200 must be positive [M_sun]")
        if self.concentration <= 0:
            raise ValueError("concentration must be positive")
        cosmo = default_cosmology()
        if self.cosmology:
            cosmo.update(self.cosmology)
        z = float(self.z_lens or 0.0)
        hz = cosmo["H0"] * 1000.0 / constants()["Mpc_m"] * np.sqrt(cosmo["Omega_m"] * (1 + z) ** 3 + cosmo["Omega_lambda"])
        rho_crit = 3.0 * hz * hz / (8.0 * np.pi * constants()["G_si"])
        self.mass_kg = self.M200 * constants()["M_sun_kg"]
        self.r200 = (3.0 * self.mass_kg / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
        self.rs = self.r200 / self.concentration
        f_c = np.log(1.0 + self.concentration) - self.concentration / (1.0 + self.concentration)
        self.rho_s = self.mass_kg / (4.0 * np.pi * self.rs**3 * f_c)

    def potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return NFW Phi [m^2 s^-2] for positions [m]."""
        rel = self._relative(position)
        r = np.sqrt(np.sum(rel * rel, axis=-1) + float(self.softening_radius_m) ** 2)
        x = r / self.rs
        return -4.0 * np.pi * constants()["G_si"] * self.rho_s * self.rs**3 * np.log1p(x) / np.maximum(r, self._eps)

    def grad_potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Return NFW ``nabla Phi`` [m s^-2] for positions [m]."""
        rel = self._relative(position)
        r = np.sqrt(np.sum(rel * rel, axis=-1, keepdims=True) + float(self.softening_radius_m) ** 2)
        x = r / self.rs
        m_enc = 4.0 * np.pi * self.rho_s * self.rs**3 * (np.log1p(x) - x / (1.0 + x))
        return constants()["G_si"] * m_enc * rel / np.maximum(r, self._eps) ** 3


class IrregularGridLens(BaseLens):
    """Interface for pixel-grid full numerical truth 생성용 lens fields.

    The class stores grid metadata and exposes the Phase 2 Phi/n_eff provider
    API. Numerical integration from Sigma(x,y) or Phi grids is intentionally a
    skeleton here and raises clear NotImplementedError until Phase 3 truth
    generation needs it.
    """

    def __init__(self, grid: np.ndarray, coordinates: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        arr = np.asarray(grid, dtype=float)
        if arr.ndim not in {2, 3}:
            raise ValueError("IrregularGridLens grid must be 2D Sigma(x,y) or 3D Phi(x,y,z)")
        if min(arr.shape) < 2:
            raise ValueError("IrregularGridLens grid dimensions must all be >= 2")
        self.grid = arr
        self.coordinates = coordinates or {}

    def potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Full numerical truth 생성용 Phi interpolation skeleton for positions [m]."""
        raise NotImplementedError("IrregularGridLens potential interpolation is reserved for full numerical truth generation")

    def grad_potential_3d(self, position: np.ndarray) -> np.ndarray:
        """Full numerical truth 생성용 gradient interpolation skeleton for positions [m]."""
        raise NotImplementedError("IrregularGridLens gradient interpolation is reserved for full numerical truth generation")
