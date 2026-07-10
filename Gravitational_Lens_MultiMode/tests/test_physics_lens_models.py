import numpy as np
import pytest

from core.physics.lens_models import IrregularGridLens, NFWLens, PointMassLens, SIELens, SISLens


def test_sis_lens_fields_are_finite():
    lens = SISLens(sigma_v=220.0, z_lens=0.3, z_source=1.5)
    pos = np.array([[1.0e20, 0.0, 0.0], [0.0, 2.0e20, 0.0]])
    phi = lens.potential_3d(pos)
    grad = lens.grad_potential_3d(pos)
    neff = lens.effective_refractive_index(pos)
    assert phi.shape == (2,)
    assert grad.shape == pos.shape
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(neff))
    assert lens.einstein_radius() > 0


def test_sie_q_validation_and_docstring():
    with pytest.raises(ValueError):
        SIELens(sigma_v=220.0, q=0.0)
    lens = SIELens(sigma_v=220.0, q=0.75, z_lens=0.3, z_source=1.5)
    pos = np.array([[1.0e20, 2.0e20, 0.0]])
    assert np.all(np.isfinite(lens.potential_3d(pos)))
    assert np.all(np.isfinite(lens.effective_refractive_index(pos)))
    assert "SIE 표준 근사 가정" in (SIELens.__doc__ or "")


def test_sie_fermat_potential_matches_sis_when_q_is_one():
    sie = SIELens(sigma_v=220.0, q=1.0, z_lens=0.3, z_source=1.5)
    sis = SISLens(sigma_v=220.0, z_lens=0.3, z_source=1.5)
    theta = np.array([[0.7, 0.2], [-0.4, 0.1]], dtype=float)
    beta = np.array([0.05, -0.02], dtype=float)
    assert np.allclose(sie.fermat_potential(theta, beta), sis.fermat_potential(theta, beta))


def test_sie_fermat_potential_gradient_matches_deflection():
    lens = SIELens(
        sigma_v=230.0,
        q=0.7,
        position_angle=0.4,
        z_lens=0.35,
        z_source=1.7,
    )
    theta = np.array([0.8, 0.35], dtype=float)
    beta = np.zeros(2, dtype=float)
    eps = 1.0e-5

    def psi(theta_arcsec: np.ndarray) -> float:
        from core.physics.config import constants

        th = np.asarray(theta_arcsec, dtype=float)
        th_rad = th * constants()["arcsec_to_rad"]
        be_rad = beta * constants()["arcsec_to_rad"]
        return 0.5 * np.sum((th_rad - be_rad) ** 2) - float(lens.fermat_potential(th, beta))

    grad = np.array(
        [
            (psi(theta + np.array([eps, 0.0])) - psi(theta - np.array([eps, 0.0]))) / (2.0 * eps),
            (psi(theta + np.array([0.0, eps])) - psi(theta - np.array([0.0, eps]))) / (2.0 * eps),
        ]
    )
    from core.physics.config import constants

    alpha_from_potential = grad / constants()["arcsec_to_rad"] ** 2
    assert np.allclose(alpha_from_potential, lens.deflection(theta), rtol=1.0e-5, atol=1.0e-6)


def test_point_mass_lens_fields_and_einstein_radius():
    lens = PointMassLens(mass_msun=1.0e11, z_lens=0.3, z_source=1.5)
    pos = np.array([[1.0e20, 0.0, 0.0]])
    assert lens.einstein_radius() > 0
    assert np.all(np.isfinite(lens.potential_3d(pos)))
    assert np.all(np.isfinite(lens.grad_potential_3d(pos)))


def test_nfw_lens_initialization_and_validation():
    with pytest.raises(ValueError):
        NFWLens(M200=-1.0, concentration=5.0)
    with pytest.raises(ValueError):
        NFWLens(M200=1.0e12, concentration=0.0)
    lens = NFWLens(M200=1.0e12, concentration=8.0, z_lens=0.3)
    assert lens.r200 > 0
    assert lens.rs > 0
    assert lens.rho_s > 0


def test_irregular_grid_shape_validation():
    with pytest.raises(ValueError):
        IrregularGridLens(np.ones((1, 4)))
    lens = IrregularGridLens(np.ones((8, 8)), coordinates={"pixel_size_m": 1.0e19})
    assert lens.grid.shape == (8, 8)
    with pytest.raises(NotImplementedError):
        lens.potential_3d(np.zeros((1, 3)))
