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
