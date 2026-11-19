"""Analytic trajectories verify acceleration signs, units and missing-data handling."""

import numpy as np
import pytest

from gas_spring.lap_profiles import estimate_lap_acceleration, STANDARD_GRAVITY_M_S2


def irregular_times():
    rng = np.random.default_rng(42)
    return np.cumsum(rng.uniform(0.12, 0.25, 100)) - 2.0


def test_accelerating_straight_recovers_acceleration_in_si_and_g():
    times = irregular_times()
    acceleration = 2.3
    speed = 20.0 + acceleration * times
    x = 20.0 * times + 0.5 * acceleration * times**2
    query = np.linspace(0, 12, 90)
    result = estimate_lap_acceleration(times, speed * 3.6, times,
                                       np.column_stack((x, np.zeros(len(x)))), query)
    np.testing.assert_allclose(result.estimated_longitudinal_acceleration_m_s2, acceleration, atol=1e-10)
    np.testing.assert_allclose(result.estimated_longitudinal_acceleration_g,
                               acceleration / STANDARD_GRAVITY_M_S2, atol=1e-10)
    np.testing.assert_allclose(result.estimated_lateral_acceleration_m_s2, 0, atol=1e-10)
    assert result.longitudinal_estimate_valid.all() and result.lateral_estimate_valid.all()


@pytest.mark.parametrize("direction", [-1, 1])
def test_constant_speed_circle_recovers_signed_centripetal_acceleration(direction):
    times = irregular_times()
    speed, radius = 30.0, 100.0
    theta = direction * speed / radius * times
    xy = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    query = np.linspace(0, 12, 80)
    result = estimate_lap_acceleration(times, np.full(len(times), speed * 3.6), times, xy, query)
    np.testing.assert_allclose(result.estimated_lateral_acceleration_m_s2,
                               direction * speed**2 / radius, rtol=0.01)
    np.testing.assert_allclose(result.estimated_longitudinal_acceleration_m_s2, 0, atol=1e-10)
    # Arbitrary coordinate translation, rotation and uniform length scale cancel.
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = (xy @ rotation.T) * 10.0 + [100000.0, -250000.0]
    other = estimate_lap_acceleration(times, np.full(len(times), speed * 3.6), times, transformed, query)
    np.testing.assert_allclose(other.estimated_lateral_acceleration_m_s2,
                               result.estimated_lateral_acceleration_m_s2, rtol=1e-8)


def test_position_gap_marks_only_lateral_estimate_unavailable():
    times = np.arange(-2.0, 12.01, 0.2)
    available = (times < 4) | (times > 6)
    position_times = times[available]
    theta = 0.3 * position_times
    xy = 100 * np.column_stack((np.cos(theta), np.sin(theta)))
    result = estimate_lap_acceleration(times, np.full(len(times), 108.0), position_times, xy,
                                       np.array([1.0, 5.0, 9.0]))
    assert result.lateral_estimate_valid.tolist() == [True, False, True]
    assert np.isnan(result.estimated_lateral_acceleration_g.iloc[1])
    assert result.longitudinal_estimate_valid.all()
    np.testing.assert_allclose(result.estimated_longitudinal_acceleration_m_s2, 0, atol=1e-12)


def test_nonincreasing_source_times_are_rejected():
    times = np.arange(10.0)
    times[4] = times[3]
    with pytest.raises(ValueError, match="increasing timestamps"):
        estimate_lap_acceleration(times, np.full(10, 108.0), np.arange(10.0),
                                  np.column_stack((np.arange(10.0), np.arange(10.0))), np.array([4.0]))
