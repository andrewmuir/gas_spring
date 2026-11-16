import copy
import numpy as np
import pytest

from gas_spring.config import load_yaml
from gas_spring.incompressible import (
    accumulator_position_m,
    accumulator_velocity_m_s,
    constrained_accelerations,
    hydraulic_flows_m3_s,
    liquid_piston_areas,
)
from gas_spring.hydraulics import piston_area
from gas_spring.materials import ATMOSPHERIC_PRESSURE_PA


def test_accumulator_position_obeys_volume_constraint_with_blowoff_and_expansion():
    geometry = load_yaml("config/physical.yaml")["geometry"]
    af, am, ab = liquid_piston_areas(geometry)
    xa, xb, blowoff, expansion = 0.002, -0.001, 0.003, 2e-6
    main = accumulator_position_m(geometry, xa, xb, blowoff, expansion)
    assert -af * (xa + xb) - am * main + ab * blowoff == pytest.approx(
        expansion, abs=1e-15
    )
    assert accumulator_position_m(geometry, xa, xb, 0, 0) == pytest.approx(
        -af * (xa + xb) / am
    )
    assert accumulator_position_m(geometry, 0, 0, 0, 0) == 0


def test_thermal_expansion_and_blowoff_enter_flow_continuity():
    geometry = load_yaml("config/physical.yaml")["geometry"]
    sources = {
        "actuator_a_fluid": 1e-8,
        "actuator_b_fluid": 2e-8,
        "main_fluid": 3e-8,
        "blowoff_fluid": 4e-8,
        "hydraulic_inlet_junction": 5e-9,
        "hydraulic_outlet_junction": 6e-9,
        "hydraulic_pipe_0": 7e-9,
        "hydraulic_pipe_1": 8e-9,
    }
    va, vb, vblowoff = 0.01, -0.004, 0.003
    vmain = accumulator_velocity_m_s(geometry, va, vb, vblowoff, sum(sources.values()))
    velocities = {
        "actuator_a": va,
        "actuator_b": vb,
        "main_accumulator": vmain,
        "blowoff": vblowoff,
    }
    flows = hydraulic_flows_m3_s(geometry, velocities, sources, pipe_cells=2)
    assert flows["continuity_residual_m3_s"] == pytest.approx(0, abs=1e-18)
    assert len(flows["pipe"]) == 3
    assert np.isfinite(list(flows.values())[:-2]).all()


def test_incompressible_pressure_solution_constrains_acceleration():
    physical = load_yaml("config/physical.yaml")
    run = load_yaml("config/run.yaml")
    geometry = physical["geometry"]
    af, am, ab = liquid_piston_areas(geometry)
    ag = piston_area(geometry["actuator_piston_diameter_m"])
    gauge = run["initial_gas_gauge_pressure_bar"]
    gas = {
        "actuator_a": ATMOSPHERIC_PRESSURE_PA + gauge["low"] * 1e5,
        "actuator_b": ATMOSPHERIC_PRESSURE_PA + gauge["low"] * 1e5,
        "main_accumulator": ATMOSPHERIC_PRESSURE_PA + gauge["main"] * 1e5,
        "blowoff": ATMOSPHERIC_PRESSURE_PA + gauge["blowoff"] * 1e5,
    }
    balance = 1e5 * (gauge["main"] * af - gauge["low"] * ag)
    velocities = {
        "actuator_a": 0.0,
        "actuator_b": 0.0,
        "main_accumulator": 0.0,
        "blowoff": 0.0,
    }
    state = constrained_accelerations(
        physical,
        velocities,
        gas,
        {"actuator_a": balance + 500, "actuator_b": balance + 500},
        {},
        {},
        pipe_cells=2,
        stopped=("blowoff",),
    )
    a = state["acceleration_m_s2"]
    assert -af * (a["actuator_a"] + a["actuator_b"]) - am * a[
        "main_accumulator"
    ] + ab * a["blowoff"] == pytest.approx(0, abs=1e-12)
    assert a["actuator_a"] > 0
    assert a["main_accumulator"] < 0
    assert a["blowoff"] == pytest.approx(0, abs=1e-12)
    assert state["pressures_pa"]["actuator_a"] > gas["main_accumulator"]


def test_independent_main_accumulator_velocity_is_rejected():
    physical = load_yaml("config/physical.yaml")
    gas = {
        "actuator_a": 1601325.0,
        "actuator_b": 1601325.0,
        "main_accumulator": 3601325.0,
        "blowoff": 8101325.0,
    }
    velocity = {
        "actuator_a": 0.0,
        "actuator_b": 0.0,
        "main_accumulator": 0.001,
        "blowoff": 0.0,
    }
    with pytest.raises(ValueError, match="volume continuity"):
        constrained_accelerations(
            physical,
            velocity,
            gas,
            {"actuator_a": 1729.0, "actuator_b": 1729.0},
            {},
            {},
            1,
            stopped=("blowoff",),
        )


def test_stopped_main_accumulator_shares_volume_motion_between_actuators():
    physical = load_yaml("config/physical.yaml")
    geometry = physical["geometry"]
    af, am, ab = liquid_piston_areas(geometry)
    gas = {
        "actuator_a": 1601325.0,
        "actuator_b": 1601325.0,
        "main_accumulator": 3601325.0,
        "blowoff": 8101325.0,
    }
    velocities = {name: 0.0 for name in gas}
    balance = 35e5 * af - 15e5 * piston_area(geometry["actuator_piston_diameter_m"])
    state = constrained_accelerations(
        physical,
        velocities,
        gas,
        {"actuator_a": balance + 300, "actuator_b": balance - 300},
        {},
        {},
        pipe_cells=1,
        stopped=("main_accumulator", "blowoff"),
    )
    a = state["acceleration_m_s2"]
    assert a["main_accumulator"] == pytest.approx(0, abs=1e-12)
    assert a["blowoff"] == pytest.approx(0, abs=1e-12)
    assert a["actuator_a"] == pytest.approx(-a["actuator_b"], rel=1e-8)
    assert -af * (a["actuator_a"] + a["actuator_b"]) - am * a[
        "main_accumulator"
    ] + ab * a["blowoff"] == pytest.approx(0, abs=1e-12)


def test_thermal_expansion_rate_and_acceleration_enter_the_constraint():
    physical = load_yaml("config/physical.yaml")
    geometry = physical["geometry"]
    sources = {"actuator_a_fluid": 1e-8, "main_fluid": 2e-8, "hydraulic_pipe_0": 3e-8}
    source_rates = {"actuator_a_fluid": 1e-8, "main_fluid": 2e-8}
    velocities = {"actuator_a": 0.0, "actuator_b": 0.0, "blowoff": 0.0}
    velocities["main_accumulator"] = accumulator_velocity_m_s(
        geometry, 0, 0, 0, sum(sources.values())
    )
    gas = {
        "actuator_a": 1601325.0,
        "actuator_b": 1601325.0,
        "main_accumulator": 3601325.0,
        "blowoff": 8101325.0,
    }
    af, _, _ = liquid_piston_areas(geometry)
    balance = 35e5 * af - 15e5 * piston_area(geometry["actuator_piston_diameter_m"])
    state = constrained_accelerations(
        physical,
        velocities,
        gas,
        {"actuator_a": balance, "actuator_b": balance},
        sources,
        source_rates,
        pipe_cells=1,
        stopped=("blowoff",),
    )
    assert state["flows_m3_s"]["continuity_residual_m3_s"] == pytest.approx(
        0, abs=1e-18
    )
    assert state["volume_acceleration_residual_m3_s2"] == pytest.approx(0, abs=1e-12)
    assert all(
        state["friction_force_n"][name] * velocities[name] >= 0 for name in velocities
    )
    assert all(power >= 0 for power in state["branch_dissipation_w"].values())
    assert all(power >= 0 for power in state["pipe_dissipation_w"])


def test_pipe_inertia_remains_in_the_constrained_dynamics():
    physical = load_yaml("config/physical.yaml")
    geometry = physical["geometry"]
    af, _, _ = liquid_piston_areas(geometry)
    balance = 35e5 * af - 15e5 * piston_area(geometry["actuator_piston_diameter_m"])
    gas = {
        "actuator_a": 1601325.0,
        "actuator_b": 1601325.0,
        "main_accumulator": 3601325.0,
        "blowoff": 8101325.0,
    }
    velocities = {name: 0.0 for name in gas}
    forces = {"actuator_a": balance + 500, "actuator_b": balance + 500}
    normal = constrained_accelerations(
        physical, velocities, gas, forces, {}, {}, 2, stopped=("blowoff",)
    )
    low_inertia = copy.deepcopy(physical)
    low_inertia["fluid"]["density_kg_m3_at_20c"] = 100.0
    reduced = constrained_accelerations(
        low_inertia, velocities, gas, forces, {}, {}, 2, stopped=("blowoff",)
    )
    assert abs(normal["acceleration_m_s2"]["actuator_a"]) < abs(
        reduced["acceleration_m_s2"]["actuator_a"]
    )
    assert abs(normal["volume_acceleration_residual_m3_s2"]) < 1e-12


def test_blowoff_connection_fluid_inertia_loads_the_piston():
    physical = load_yaml("config/physical.yaml")
    geometry = physical["geometry"]
    af, am, ab = liquid_piston_areas(geometry)
    gas = {
        "actuator_a": 1601325.0,
        "actuator_b": 1601325.0,
        "main_accumulator": 3601325.0,
        "blowoff": 8101325.0,
    }
    velocity = {name: 0.0 for name in gas}
    balance = 35e5 * af - 15e5 * piston_area(0.038)
    preload = 80e5 * piston_area(0.045) - 35e5 * ab
    forces = {"actuator_a": balance, "actuator_b": balance}
    normal = constrained_accelerations(
        physical,
        velocity,
        gas,
        forces,
        {},
        {},
        1,
        external_piston_forces_n={"blowoff": preload + 500.0},
    )
    short = copy.deepcopy(physical)
    short["geometry"]["blowoff_connection_length_m"] *= 0.01
    reduced = constrained_accelerations(
        short,
        velocity,
        gas,
        forces,
        {},
        {},
        1,
        external_piston_forces_n={"blowoff": preload + 500.0},
    )
    assert (
        0
        < normal["acceleration_m_s2"]["blowoff"]
        < reduced["acceleration_m_s2"]["blowoff"]
    )
    assert abs(normal["volume_acceleration_residual_m3_s2"]) < 1e-12


def test_elastic_stops_react_load_with_the_specified_stiffness():
    from gas_spring.mechanics import end_stop_response

    k = 1e7
    assert end_stop_response(0.5, 0.0, 1.0, k)["force_n"] == 0
    for x, expected in ((-0.001, 10000.0), (1.001, -10000.0)):
        response = end_stop_response(x, 0.0, 1.0, k)
        assert response["force_n"] == pytest.approx(expected)
        assert response["energy_j"] == pytest.approx(5.0)
        h = 1e-7
        gradient = (
            end_stop_response(x + h, 0.0, 1.0, k)["energy_j"]
            - end_stop_response(x - h, 0.0, 1.0, k)["energy_j"]
        ) / (2 * h)
        assert -gradient == pytest.approx(expected, rel=1e-7)
    with pytest.raises(ValueError, match="stiffness"):
        end_stop_response(0.0, -1.0, 1.0, 0.0)


def test_loaded_blowoff_stop_recovers_preload_before_separating():
    from gas_spring.mechanics import end_stop_response

    k, preload = 1e7, 7500.0
    initial = end_stop_response(0.0, 0.0, 0.05, k, preload)
    assert initial["force_n"] == pytest.approx(preload)
    assert initial["lower_deflection_m"] == pytest.approx(0.00075)
    assert end_stop_response(0.0005, 0.0, 0.05, k, preload)["force_n"] == pytest.approx(
        2500.0
    )
    assert end_stop_response(0.001, 0.0, 0.05, k, preload)["force_n"] == 0.0


def test_main_gas_pressure_has_no_cap_as_compressed_volume_approaches_zero():
    from gas_spring.balances import ideal_gas_state
    from gas_spring.mechanics import chamber_volumes
    from gas_spring.initialization import initialize_system

    physical = load_yaml("config/physical.yaml")
    run = load_yaml("config/run.yaml")
    initial = initialize_system(physical, run, 1729.0, 1729.0)
    node = initial["gas_chambers"]["main_gas"]
    geometry = physical["geometry"]
    area = piston_area(geometry["main_accumulator_piston_diameter_m"])
    p0 = ideal_gas_state(
        node["mass_kg"], node["internal_energy_j"], node["volume_m3"], physical["gas"]
    )[0]
    for fraction in (1.0, 0.1, 0.01, 0.001, 0.000001):
        x = -node["volume_m3"] * (1 - fraction) / area
        volume = chamber_volumes(geometry, main_m=x)["main_gas"]
        pressure = ideal_gas_state(
            node["mass_kg"], node["internal_energy_j"], volume, physical["gas"]
        )[0]
        assert pressure == pytest.approx(p0 / fraction, rel=1e-8)
    # Positive travel is gas expansion, including beyond the elastic stop's
    # first-contact coordinate; there is no artificial pressure singularity.
    volume = chamber_volumes(geometry, main_m=0.008)["main_gas"]
    assert volume > node["volume_m3"]
    with pytest.raises(ValueError, match="volume"):
        chamber_volumes(geometry, main_m=-node["volume_m3"] / area - 1e-9)
