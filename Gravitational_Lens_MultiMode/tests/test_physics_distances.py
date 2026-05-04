import pytest

from core.physics.distances import (
    angular_diameter_distance,
    angular_diameter_distance_between,
    comoving_distance,
    time_delay_distance,
)


def test_distances_positive():
    assert comoving_distance(0.5) > 0
    assert angular_diameter_distance(0.5) > 0
    assert angular_diameter_distance_between(0.3, 1.5) > 0
    assert time_delay_distance(0.3, 1.5) > 0


def test_source_redshift_must_exceed_lens():
    with pytest.raises(ValueError):
        angular_diameter_distance_between(0.5, 0.5)
    with pytest.raises(ValueError):
        time_delay_distance(0.5, 0.4)
