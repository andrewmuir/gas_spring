import copy
import math

import numpy as np
import pytest

from gas_spring.config import load_yaml
from gas_spring.circuit import circuit_nodes_from_initial_state
from gas_spring.coupling import IncompressibleSystem, PISTONS
from gas_spring.initialization import initialize_system
from gas_spring.hydraulics import (
    DigressiveCurve,
    piston_area,
    pipe_power_derivatives,
    pipe_pressure_drop,
    restriction_power_slope,
    regularized_restriction_pressure_drop,
)
from gas_spring.thermal_coupling import thermal_exchange, liquid_heat_rate_derivative
from gas_spring.solver import _Layout
from gas_spring.force_input import InputInterpolator, interpolate_inputs
from gas_spring.circuit import nitrogen_rates
from gas_spring.config import load_inputs


@pytest.mark.parametrize("reynolds", [100.0, 3000.0, 10000.0])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_darcy_power_derivatives_match_the_loss_law(reynolds, direction):
    rho, mu, length, diameter = 1062.0, 0.01, 1.0, 0.004
    q = direction * reynolds * math.pi * diameter * mu / (4 * rho)
    dq, dmu = pipe_power_derivatives(q, rho, mu, length, diameter)

    def power(flow, viscosity):
        return flow * pipe_pressure_drop(flow, rho, viscosity, length, diameter)

    hq, hm = abs(q) * 1e-5, mu * 1e-5
    assert dq == pytest.approx(
        (power(q + hq, mu) - power(q - hq, mu)) / (2 * hq), rel=1e-7
    )
    assert dmu == pytest.approx(
        (power(q, mu + hm) - power(q, mu - hm)) / (2 * hm), rel=1e-7
    )


@pytest.mark.parametrize("q", [-0.01, -1e-6, -1e-11, 0.0, 1e-11, 1e-6, 0.01])
def test_restriction_power_derivative_matches_tail_and_reversal(q):
    area = piston_area(0.038, 0.014)
    c = DigressiveCurve.from_points(area, 0.15, 4000.0, 0.8, 6000.0)
    e = DigressiveCurve.from_points(area, 0.15, 5000.0, 0.8, 7000.0)

    def power(flow):
        return flow * regularized_restriction_pressure_drop(flow, c, e, 0.95, 1e-11)

    h = max(abs(q) * 1e-5, 1e-17)
    assert restriction_power_slope(q, c, e, 0.95, 1e-11) == pytest.approx(
        (power(q + h) - power(q - h)) / (2 * h), rel=1e-6, abs=1e-8
    )


def state_and_rate():
    physical, run = load_yaml("config/physical.yaml"), load_yaml("config/run.yaml")
    initial = initialize_system(
        physical, run, 1729.0, 1729.0, pipe_cells=2, actuator_wall_cells=3
    )
    liquid, gas = circuit_nodes_from_initial_state(initial)
    state = dict(
        position=initial["position_m"],
        velocity=initial["velocity_m_s"],
        liquid=liquid,
        gas=gas,
        solid=initial["solid_temperatures_c"],
    )
    state["solid"]["actuator_a_wall"][:] = [45.0, 50.0, 65.0]
    state["velocity"].update(actuator_a=0.01, actuator_b=-0.003, blowoff=0.002)
    system = IncompressibleSystem(physical, run, 2, liquid)
    u = dict(
        actuator_a_force_n=2000.0,
        actuator_b_force_n=1800.0,
        hot_ambient_c=120.0,
        duct_air_speed_m_s=2.0,
    )
    rate = system.evaluate(state, u)
    return system, state, u, rate


def advance(state, rate, dt, acceleration):
    shifted = copy.deepcopy(state)
    for name in PISTONS:
        shifted["position"][name] += dt * state["velocity"][name]
        shifted["velocity"][name] += dt * acceleration[name]
    for group in ("liquid", "gas"):
        for name, node in shifted[group].items():
            for field, key in (
                ("mass_kg", "mass_rate_kg_s"),
                ("internal_energy_j", "energy_rate_w"),
                ("boundary_energy_j", "boundary_energy_rate_w"),
            ):
                if field in node:
                    node[field] += dt * rate[group]["node_rates"][name][key]
    for name, value in rate["thermal"]["solid_temperature_rate_c_s"].items():
        shifted["solid"][name] += dt * value
    return shifted


def test_liquid_heat_derivative_includes_wetted_area_and_boundary_temperature():
    system, state, u, rate = state_and_rate()
    predicted, _ = liquid_heat_rate_derivative(
        system.physical,
        state["liquid"],
        state["solid"],
        state["position"],
        state["velocity"],
        rate["liquid"]["node_rates"],
        rate["thermal"]["solid_temperature_rate_c_s"],
    )
    powers = []
    for dt in (-1e-6, 1e-6):
        shifted = advance(state, rate, dt, rate["velocity_rate_m_s2"])
        system.volumes(shifted)
        powers.append(
            thermal_exchange(
                system.physical,
                system.run,
                shifted["liquid"],
                shifted["gas"],
                shifted["solid"],
                shifted["position"],
                u["hot_ambient_c"],
                u["duct_air_speed_m_s"],
            )["liquid_heat_w"]
        )
    for name in predicted:
        assert predicted[name] == pytest.approx(
            (powers[1][name] - powers[0][name]) / 2e-6, rel=2e-6, abs=2e-6
        )


def test_expansion_acceleration_matches_directional_thermal_and_loss_change():
    system, state, u, rate = state_and_rate()
    acceleration = dict(zip(PISTONS, [2.0, -1.0, 3.0, -0.5]))
    predicted = system.source_rate_affine(state, rate)(
        np.array(list(acceleration.values()))
    )
    sources = []
    for dt in (-1e-6, 1e-6):
        shifted = advance(state, rate, dt, acceleration)
        sources.append(system.thermal_state(shifted, u)["sources"])
    for name in predicted:
        assert predicted[name] == pytest.approx(
            (sources[1][name] - sources[0][name]) / 2e-6, rel=3e-6, abs=1e-13
        )


def test_prepared_input_history_keeps_validation_and_owns_its_arrays():
    run = load_yaml("config/run_5s.yaml")
    inputs = load_inputs(run, "config/run_5s.yaml")
    interpolate = InputInterpolator(inputs)
    expected = interpolate_inputs(inputs, 4.137)
    assert interpolate(4.137) == expected
    inputs.loc[:, "actuator_a_force_n"] = 0.0
    assert interpolate(4.137) == expected
    with pytest.raises(ValueError, match="outside"):
        interpolate(5.001)
    inputs.loc[7, "duct_air_speed_m_s"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        InputInterpolator(inputs)


def test_layout_evaluations_do_not_mutate_each_other_or_the_template():
    system, state, _, _ = state_and_rate()
    layout = _Layout(state, system, "main_accumulator")
    original = layout.unpack(layout.y0)
    changed = layout.unpack(layout.y0)
    changed["liquid"]["main_fluid"]["mass_kg"] *= 2
    changed["gas"]["main_gas"]["internal_energy_j"] *= 2
    changed["solid"]["actuator_a_wall"][0] = 999.0
    changed["position"]["actuator_a"] = 1.0
    again = layout.unpack(layout.y0)
    for group in ("liquid", "gas", "position", "velocity"):
        assert again[group] == original[group]
    for name in original["solid"]:
        np.testing.assert_array_equal(again["solid"][name], original["solid"][name])


def test_connected_gas_has_one_energy_state_and_exact_total_mass():
    system, state, _, _ = state_and_rate()
    layout = _Layout(state, system, "main_accumulator")
    fields = [
        f
        for f in layout.fields
        if f[0] == "gas"
        and f[2] == "internal_energy_j"
        and f[1] in system.gas_network.names
    ]
    assert fields == [("gas", "low_pressure_reservoir_gas", "internal_energy_j")]
    changed = layout.y0.copy()
    mass_index = layout.indices[("gas", "actuator_a_gas", "mass_kg")]
    changed[mass_index] *= 1.01
    reconstructed = layout.unpack(changed)
    assert sum(
        reconstructed["gas"][n]["mass_kg"] for n in system.gas_network.names
    ) == pytest.approx(layout.network_mass, abs=1e-17)
    assert all(
        ("gas", n, "mass_kg") not in layout.indices
        for n in ("main_gas", "blowoff_gas", "low_pressure_reservoir_gas")
    )


@pytest.mark.parametrize(
    "velocities", [(0.0, 0.0, 0.0), (0.01, -0.003, 0.002), (0.05, 0.03, 0.004)]
)
def test_frozen_thermal_coefficients_give_a_consistent_newton_linearization(velocities):
    system, state, u, _ = state_and_rate()
    state["velocity"].update(zip(("actuator_a", "actuator_b", "blowoff"), velocities))
    layout = _Layout(state, system, "main_accumulator")
    base = layout.unpack(layout.y0)
    rate = system.evaluate(base, u)
    frozen = rate["source_rate_function"]
    recomputed = system.evaluate(
        layout.unpack(layout.y0), u, frozen_source_rate_function=frozen
    )
    np.testing.assert_allclose(
        layout.derivative(recomputed, base),
        layout.derivative(rate, base),
        rtol=1e-10,
        atol=1e-10,
    )
    rng = np.random.default_rng(1234)
    direction = []
    for value, (group, name, field) in zip(layout.y0, layout.fields):
        if group == "position":
            scale = 0.001
        elif group == "velocity":
            scale = 0.02
        elif group == "solid":
            scale = 1.0
        elif field == "mass_kg":
            scale = abs(value) * 0.01
        else:
            scale = max(abs(value), 1.0) * 0.01
        direction.append(scale * rng.normal())
    direction = np.array(direction)
    full, approximate = [], []
    for h in (-1e-5, 1e-5):
        shifted = layout.unpack(layout.y0 + h * direction)
        full.append(layout.derivative(system.evaluate(shifted, u), shifted))
        shifted = layout.unpack(layout.y0 + h * direction)
        approximate.append(
            layout.derivative(
                system.evaluate(shifted, u, frozen_source_rate_function=frozen), shifted
            )
        )
    true_change = (full[1] - full[0]) / 2e-5
    estimated = (approximate[1] - approximate[0]) / 2e-5
    # Only the weak thermal-volume coefficient variation is lagged. The
    # physical RHS and the dominant mechanical/thermal derivatives agree.
    np.testing.assert_allclose(estimated, true_change, rtol=0.01, atol=1e-5)


def test_reused_gas_thermal_jacobian_block_matches_complete_frozen_evaluations():
    system, state, u, _ = state_and_rate()
    layout = _Layout(state, system, "main_accumulator")
    base_state = layout.unpack(layout.y0)
    base = system.evaluate(base_state, u)
    checked = 0
    for i, field in enumerate(layout.fields):
        if not layout.gas_thermal_field(field):
            continue
        shifted = layout.y0.copy()
        shifted[i] += max(abs(shifted[i]) * 0.001, 1e-7)
        fast_state = layout.unpack(shifted)
        fast = system.gas_thermal_jacobian_rate(fast_state, u, base, "main_accumulator")
        full_state = layout.unpack(shifted)
        full = system.evaluate(
            full_state, u, frozen_source_rate_function=base["source_rate_function"]
        )
        np.testing.assert_allclose(
            layout.derivative(fast, fast_state),
            layout.derivative(full, full_state),
            rtol=1e-11,
            atol=1e-10,
        )
        checked += 1
    assert checked >= 15
