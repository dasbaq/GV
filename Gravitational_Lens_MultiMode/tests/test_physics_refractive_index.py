import numpy as np

from core.physics.refractive_index import (
    effective_refractive_index,
    grad_refractive_index_from_grad_phi,
    refractive_index_from_potential,
)


def test_phi_zero_gives_neff_one():
    assert effective_refractive_index(0.0) == 1.0


def test_bound_potential_gives_neff_greater_than_one():
    assert effective_refractive_index(-1.0e12) > 1.0


def test_scalar_and_array_inputs_work():
    phi = np.array([0.0, -1.0e12])
    out = refractive_index_from_potential(phi)
    assert out.shape == phi.shape
    assert out[0] == 1.0
    assert out[1] > 1.0


def test_grad_refractive_index_shape():
    grad_phi = np.ones((4, 3))
    grad_n = grad_refractive_index_from_grad_phi(grad_phi)
    assert grad_n.shape == grad_phi.shape
    assert np.all(np.isfinite(grad_n))
