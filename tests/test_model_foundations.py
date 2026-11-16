import copy
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

from gas_spring.config import load_yaml
from gas_spring.hydraulics import (
    piston_area,
    DigressiveCurve,
    restriction_flow,
    pipe_inertance,
    regularized_restriction_pressure_drop,
    gas_pipe_mass_flow,
)
from gas_spring.balances import (
    advective_flux_rates,
    ideal_gas_state,
    ideal_gas_energy_rate,
)
from gas_spring.circuit import circuit_nodes_from_initial_state, nitrogen_rates
from gas_spring.coupling import IncompressibleSystem, LIQUID_PISTONS
from gas_spring.initialization import initialize_system
from gas_spring.force_input import interpolate_inputs, shaft_position_m
from gas_spring.diagnostics import isolated_actuator_warmup
from gas_spring.materials import (
    ATMOSPHERIC_PRESSURE_PA,
    liquid_density,
    liquid_temperatures,
    liquid_viscosity,
    liquid_internal_energy,
    nitrogen_mass,
    nitrogen_pressure,
)
from gas_spring.mechanics import (
    chamber_volumes,
    effective_piston_mass_kg,
    initial_static_equilibrium,
    piston_acceleration,
)
from gas_spring.sensors import CHANNELS, add_sensor_measurements
from gas_spring.thermal import actuator_wall_cell_areas, exchange_heat_rates
from gas_spring.thermal_coupling import thermal_exchange
from gas_spring.solver import simulate


@pytest.fixture
def config():
    return load_yaml("config/physical.yaml"), load_yaml("config/run.yaml")


def test_liquid_density_depends_only_on_temperature(config):
    fluid = config[0]["fluid"]
    assert "bulk_modulus_pa" not in fluid
    assert liquid_density(100, fluid) < liquid_density(20, fluid)
    np.testing.assert_allclose(liquid_viscosity([20, 100], fluid), [0.01, 0.0021])
    for t in (20.0, 50.0, 100.0):
        m = 0.1
        energy = m * liquid_internal_energy(t, fluid)
        assert liquid_temperatures(m, energy, fluid)[0] == pytest.approx(t)


def test_nitrogen_ideal_gas_uses_absolute_pressure(config):
    gas = config[0]["gas"]
    p = ATMOSPHERIC_PRESSURE_PA + 80e5
    m = nitrogen_mass(p, 0.001, 25, gas)
    assert nitrogen_pressure(m, 0.001, 25, gas) == pytest.approx(p)
    assert nitrogen_pressure(m, 0.0005, 25, gas) == pytest.approx(2 * p)


def test_chamber_volumes_and_differential_shaft_areas(config):
    g = config[0]["geometry"]
    zero = chamber_volumes(g)
    moved = chamber_volumes(
        g, actuator_a_m=0.05, actuator_b_m=-0.015, main_m=0.007, blowoff_m=0.05
    )
    fluid_area = piston_area(0.038, 0.014)
    gas_area = piston_area(0.038)
    assert moved["actuator_a_fluid"] == pytest.approx(fluid_area * 0.005)
    assert moved["actuator_b_gas"] == pytest.approx(gas_area * 0.005)
    assert moved["main_gas"] - zero["main_gas"] == pytest.approx(
        piston_area(0.045) * 0.007
    )
    assert moved["blowoff_gas"] - zero["blowoff_gas"] == pytest.approx(
        -piston_area(0.045) * 0.05
    )
    assert moved["blowoff_fluid"] - zero["blowoff_fluid"] == pytest.approx(
        piston_area(0.045, 0.012) * 0.05
    )
    with pytest.raises(ValueError, match="volume"):
        chamber_volumes(g, actuator_a_m=0.056)


def test_static_initialization_accepts_balance_and_rejects_conflict(config):
    physical, run = config
    state = initial_static_equilibrium(physical, run, 1729, 1729)
    assert state.hydraulic_gauge_pressure_pa == pytest.approx(35e5)
    for name in ("actuator_a", "actuator_b", "main_accumulator"):
        assert abs(state.residual_forces_n[name]) <= state.friction_limits_n[name]
    # The zero stop supports negative residual on the blow-off piston.
    assert state.residual_forces_n["blowoff"] < -state.friction_limits_n["blowoff"]
    with pytest.raises(ValueError, match="static equilibrium"):
        initial_static_equilibrium(physical, run, 0, 20000)


def make_truth():
    data = {
        "time_s": [0.0, 60.0, 120.0],
        "hot_ambient_c": [20.0, 30.0, 40.0],
        "cooler_region_temperature_c": [20.0, 22.0, 24.0],
    }
    for name, _, unit, _ in CHANNELS:
        data[f"truth_{name}_{unit}"] = [1.0, 1.0, 1.0]
    return pd.DataFrame(data)


def test_sensor_channels_apply_independent_ambient_and_calibration(config):
    physical, run = copy.deepcopy(config)
    for profile in physical["sensor_profiles"].values():
        profile["noise_std"] = profile["drift_std_per_sqrt_hour"] = 0
    run["sensor_calibration"]["actuator_a_displacement"] = {
        "gain": 2,
        "offset": 3,
        "temperature_coefficient": 0.1,
        "reference_temperature_c": 20,
    }
    run["sensor_calibration"]["blowoff_displacement"] = {
        "temperature_coefficient": 0.5,
        "reference_temperature_c": 20,
    }
    data = add_sensor_measurements(make_truth(), physical, run)
    np.testing.assert_allclose(data["sensor_actuator_a_displacement_mm"], [5, 6, 7])
    np.testing.assert_allclose(data["sensor_blowoff_displacement_mm"], [1, 2, 3])
    assert len([c for c in data if c.startswith("sensor_")]) == len(CHANNELS)


def test_sensor_channels_reproduce_and_do_not_share_noise(config):
    physical, run = config
    one = add_sensor_measurements(make_truth(), physical, run)
    two = add_sensor_measurements(make_truth(), physical, run)
    pd.testing.assert_frame_equal(one, two)
    assert not np.array_equal(
        one["sensor_actuator_a_force_n"], one["sensor_actuator_b_force_n"]
    )


def test_internal_mass_and_enthalpy_fluxes_conserve_with_flow_reversal():
    dm, du = advective_flux_rates([100, 200, 300], [(0, 1, 2), (1, 2, -3)])
    np.testing.assert_allclose(dm, [-2, 5, -3])
    np.testing.assert_allclose(du, [-200, 1100, -900])
    assert sum(dm) == pytest.approx(0)
    assert sum(du) == pytest.approx(0)


def test_gas_energy_and_pressure_work(config):
    gas = config[0]["gas"]
    mass, temp_k, volume = 0.02, 300.0, 0.001
    cv = gas["specific_heat_cp_j_kg_k"] - gas["specific_gas_constant_j_kg_k"]
    p, temp_c = ideal_gas_state(mass, mass * cv * temp_k, volume, gas)
    assert temp_c == pytest.approx(26.85)
    assert p == pytest.approx(
        mass * gas["specific_gas_constant_j_kg_k"] * temp_k / volume
    )
    assert ideal_gas_energy_rate(p, 1e-6, 0, 0) == pytest.approx(-p * 1e-6)


def test_wall_wetting_and_heat_exchange_conserve_energy():
    f, g = actuator_wall_cell_areas(0.038, 0.055, 0.020, 0.012, 7)
    assert sum(f) == pytest.approx(np.pi * 0.038 * 0.043)
    assert sum(g) == pytest.approx(np.pi * 0.038 * 0.032)
    heat = exchange_heat_rates([100, 50, 25], [(0, 1, 2), (1, 2, 3)])
    np.testing.assert_allclose(heat, [-100, 25, 75])
    assert sum(heat) == pytest.approx(0)


def test_initial_state_contains_balanced_masses_and_pipe_gradients(config):
    physical, run = config
    state = initialize_system(physical, run, 1729, 1729, pipe_cells=4)
    assert state["liquid_pressure_pa"] == pytest.approx(ATMOSPHERIC_PRESSURE_PA + 35e5)
    a = state["gas_chambers"]["actuator_a_gas"]
    assert nitrogen_pressure(
        a["mass_kg"], a["volume_m3"], a["temperature_c"], physical["gas"]
    ) == pytest.approx(ATMOSPHERIC_PRESSURE_PA + 15e5)
    np.testing.assert_allclose(
        state["pipes"]["gas"]["temperature_c"], [46.875, 40.625, 34.375, 28.125]
    )
    assert np.all(state["pipes"]["hydraulic"]["mass_kg"] > 0)
    fluid = state["liquid_chambers"]["actuator_a_fluid"]
    t, _, _ = liquid_temperatures(
        fluid["mass_kg"], fluid["internal_energy_j"], physical["fluid"]
    )
    assert t == pytest.approx(50)
    with pytest.raises(ValueError, match="static equilibrium"):
        initialize_system(physical, run, 0, 20000)


def test_liquid_volume_follows_sensible_energy_and_mass(config):
    fluid = config[0]["fluid"]
    for t in (20.0, 50.0, 100.0):
        mass = 0.1
        energy = mass * liquid_internal_energy(t, fluid)
        volume = (
            mass
            + fluid["volumetric_expansion_per_k"]
            * energy
            / fluid["specific_heat_j_kg_k"]
        ) / fluid["density_kg_m3_at_20c"]
        assert volume == pytest.approx(mass / liquid_density(t, fluid))


def test_restriction_inverse_and_extrapolation_match_both_fit_points():
    area = piston_area(0.038, 0.014)
    c = DigressiveCurve.from_points(area, 0.15, 4000, 0.8, 6000)
    e = DigressiveCurve.from_points(area, 0.15, 5000, 0.8, 7000)
    for flow in (
        area * 0.15,
        area * 0.8,
        -area * 0.15,
        -area * 0.8,
        area * 10,
        -area * 10,
    ):
        dp = regularized_restriction_pressure_drop(flow, c, e)
        assert dp * flow > 0
        assert restriction_flow(dp, c, e) == pytest.approx(flow)
    assert regularized_restriction_pressure_drop(
        area * 0.8, c, e
    ) * area == pytest.approx(6000)
    assert regularized_restriction_pressure_drop(
        -area * 0.8, c, e
    ) * area == pytest.approx(-7000)


def test_gas_pipe_flow_is_bidirectional_and_zero_when_balanced():
    args = (300.0, 296.8, 1.78e-5, 1.0, 0.004)
    forward = gas_pipe_mass_flow(1.7e6, 1.6e6, *args)
    reverse = gas_pipe_mass_flow(1.6e6, 1.7e6, *args)
    assert forward > 0
    assert reverse == pytest.approx(-forward)
    assert gas_pipe_mass_flow(1.6e6, 1.6e6, *args) == 0


def test_piston_stiction_sliding_and_stops():
    assert piston_acceleration(0, 0, 9, 10, 0.1, -1, 1) == 0
    assert piston_acceleration(0, 0, 12, 10, 0.1, -1, 1) == pytest.approx(20)
    assert piston_acceleration(
        0, 0, 12, 10, 0.1, -1, 1, kinetic_friction_limit_n=8
    ) == pytest.approx(40)
    assert piston_acceleration(0, 0.1, -10, 5, 0.1, -1, 1) == pytest.approx(-150)
    assert piston_acceleration(-1, 0, -100, 10, 0.1, -1, 1) == 0
    assert piston_acceleration(1, 0, 100, 10, 0.1, -1, 1) == 0
    with pytest.raises(ValueError, match="impact"):
        piston_acceleration(1, 0.1, 0, 10, 0.1, -1, 1)


def test_forcing_interpolation_and_shaft_deflection():
    data = pd.DataFrame(
        {
            "time_s": [0, 2],
            "actuator_a_force_n": [1000, 2000],
            "actuator_b_force_n": [-100, 100],
            "hot_ambient_c": [100, 120],
            "duct_air_speed_m_s": [1, 3],
        }
    )
    interpolated = interpolate_inputs(data, 1)
    assert interpolated == {
        "actuator_a_force_n": 1500,
        "actuator_b_force_n": 0,
        "hot_ambient_c": 110,
        "duct_air_speed_m_s": 2,
    }
    assert shaft_position_m(0.01, 2000, 20e6) == pytest.approx(0.0101)
    with pytest.raises(ValueError, match="outside"):
        interpolate_inputs(data, 3)


def instantaneous(config):
    physical, run = config
    initial = initialize_system(
        physical, run, 1729, 1729, pipe_cells=2, actuator_wall_cells=3
    )
    liquid, gas = circuit_nodes_from_initial_state(initial)
    state = dict(
        position=initial["position_m"],
        velocity=initial["velocity_m_s"],
        liquid=liquid,
        gas=gas,
        solid=initial["solid_temperatures_c"],
    )
    system = IncompressibleSystem(physical, run, 2, liquid)
    u = dict(
        actuator_a_force_n=1729.0,
        actuator_b_force_n=1729.0,
        hot_ambient_c=120.0,
        duct_air_speed_m_s=2.0,
    )
    return system, state, u


def test_complete_circuit_rate_assembly_conserves_internal_mass_and_volume(config):
    system, state, u = instantaneous(config)
    state["velocity"]["actuator_a"] = 0.01
    rate = system.evaluate(state, u, "main_accumulator")
    assert sum(
        n["mass_rate_kg_s"] for n in rate["liquid"]["node_rates"].values()
    ) == pytest.approx(0, abs=1e-14)
    assert sum(
        n["mass_rate_kg_s"] for n in rate["gas"]["node_rates"].values()
    ) == pytest.approx(0, abs=1e-14)
    assert rate["flows"]["continuity_residual_m3_s"] == pytest.approx(0, abs=1e-15)
    rho = system.fluid["density_kg_m3_at_20c"]
    for name, node_rate in rate["liquid"]["node_rates"].items():
        volume_rate = (
            node_rate["mass_rate_kg_s"] / rho + system.beta * node_rate["energy_rate_w"]
        )
        piston = next((p for p, n in LIQUID_PISTONS.items() if n == name), None)
        expected = system.j[piston] * state["velocity"][piston] if piston else 0.0
        assert volume_rate == pytest.approx(expected, abs=1e-14)


def test_reservoir_heat_drives_conservative_gas_transfer(config):
    physical, run = config
    state = initialize_system(physical, run, 1729, 1729, pipe_cells=2)
    _, gas = circuit_nodes_from_initial_state(state)
    reservoir = gas["low_pressure_reservoir_gas"]
    reservoir["heat_rate_w"] = 1.0
    rates = nitrogen_rates(gas, 2, physical)
    connection = [
        flow
        for a, b, flow in rates["mass_flows_kg_s"]
        if b == "low_pressure_reservoir_gas"
    ]
    assert len(connection) == 1 and connection[0] < 0
    assert sum(
        node["energy_rate_w"] for node in rates["node_rates"].values()
    ) == pytest.approx(1.0)


def test_coupled_rates_preserve_static_equilibrium(config):
    physical, run = copy.deepcopy(config)
    physical["fluid"]["volumetric_expansion_per_k"] = 0.0
    system, state, u = instantaneous((physical, run))
    area_gas = piston_area(physical["geometry"]["actuator_piston_diameter_m"])
    area_fluid = -system.j["actuator_a"]
    force = 35e5 * area_fluid - 15e5 * area_gas
    u["actuator_a_force_n"] = u["actuator_b_force_n"] = force
    rate = system.evaluate(state, u, "main_accumulator")
    assert all(abs(a) < 1e-8 for a in rate["velocity_rate_m_s2"].values())


def test_thermal_network_conserves_energy_and_retains_boundary_states(config):
    system, state, u = instantaneous(config)
    thermal = thermal_exchange(
        system.physical,
        system.run,
        state["liquid"],
        state["gas"],
        state["solid"],
        state["position"],
        120,
        2,
    )
    total = (
        sum(thermal["liquid_heat_w"].values())
        + sum(thermal["gas_heat_w"].values())
        + sum(thermal["solid_heat_w"].values())
    )
    assert total == pytest.approx(thermal["external_heat_w"], abs=1e-9)
    rate = system.evaluate(state, u, "main_accumulator")
    assert "boundary_energy_rate_w" in rate["liquid"]["node_rates"]["actuator_a_fluid"]
    assert "boundary_energy_rate_w" in rate["gas"]["node_rates"]["actuator_a_gas"]


def test_short_coupled_run_generates_truth_and_sensor_channels(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=0.001)
    inputs = pd.DataFrame(
        {
            "time_s": [0.0, 0.001],
            "actuator_a_force_n": [1729.0, 1729.0],
            "actuator_b_force_n": [1729.0, 1729.0],
            "hot_ambient_c": [120.0, 120.0],
            "duct_air_speed_m_s": [2.0, 2.0],
        }
    )
    result = simulate(physical, run, inputs)
    assert len(result) == 2
    assert result["truth_actuator_a_fluid_pressure_bar"].iloc[0] == pytest.approx(
        35, abs=0.01
    )
    assert result["truth_low_pressure_gas_temperature_c"].iloc[0] == pytest.approx(25)
    assert result["truth_actuator_a_gas_temperature_c"].iloc[0] == pytest.approx(50)
    assert result["truth_actuator_a_displacement_mm"].iloc[0] == pytest.approx(
        1729 / 20e6 * 1000
    )
    assert result["truth_actuator_a_wall_0_temperature_c"].iloc[-1] > 50
    assert len([c for c in result if c.startswith("sensor_")]) == len(CHANNELS)


def test_cli_writes_parquet_atomically_and_protects_input(
    config, tmp_path, monkeypatch
):
    from gas_spring.cli import main

    physical, run = copy.deepcopy(config)
    inputs = pd.DataFrame(
        {
            "time_s": [0.0, 0.001],
            "actuator_a_force_n": [1729.0, 1729.0],
            "actuator_b_force_n": [1729.0, 1729.0],
            "duct_air_speed_m_s": [2.0, 2.0],
        }
    )
    source = tmp_path / "input.parquet"
    inputs.to_parquet(source, index=False)
    physical_path, run_path = tmp_path / "physical.yaml", tmp_path / "run.yaml"
    physical_path.write_text(yaml.safe_dump(physical), encoding="utf-8")
    run.update(input_parquet="input.parquet", output_parquet="output.parquet")
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    monkeypatch.setattr(
        "gas_spring.solver.simulate",
        lambda p, r, data, **kwargs: pd.DataFrame(
            {"time_s": data.time_s, "truth_example": [3.0, 4.0]}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gas-spring",
            "--physical",
            str(physical_path),
            "--run",
            str(run_path),
            "--simulate",
        ],
    )
    main()
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "output.parquet"),
        pd.DataFrame({"time_s": [0.0, 0.001], "truth_example": [3.0, 4.0]}),
    )
    assert not list(tmp_path.glob(".gas_spring_*.parquet"))
    run["output_parquet"] = "input.parquet"
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite"):
        main()
    pd.testing.assert_frame_equal(pd.read_parquet(source), inputs)
    run["output_parquet"] = "physical.yaml"
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    physical_before = physical_path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        main()
    assert physical_path.read_bytes() == physical_before


def test_incompressible_run_preserves_volume_and_reports_progress(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, progress_interval_s=0.01)
    inputs = pd.DataFrame(
        dict(
            time_s=[0.0, 0.01, 0.02],
            actuator_a_force_n=[1729.0, 1800.0, 1900.0],
            actuator_b_force_n=[1729.0, 1700.0, 1600.0],
            hot_ambient_c=120.0,
            duct_air_speed_m_s=2.0,
        )
    )
    progress = []
    diagnostics = []
    output = simulate(
        physical,
        run,
        inputs,
        progress_callback=lambda *args: progress.append(args),
        diagnostics_callback=diagnostics.append,
    )
    assert max(abs(output.truth_liquid_volume_constraint_residual_m3)) < 1e-15
    assert max(abs(output.truth_liquid_continuity_residual_m3_s)) < 1e-14
    assert output.truth_liquid_mass_total_kg.iloc[-1] == pytest.approx(
        output.truth_liquid_mass_total_kg.iloc[0], rel=1e-10
    )
    assert progress[-1][0] == 0.02
    assert sum(d["accepted_steps"] for d in diagnostics) > 0
    assert diagnostics[-1]["cumulative_solve_wall_s"] == pytest.approx(
        sum(d["solve_wall_s"] for d in diagnostics)
    )
    assert all(
        d["simulation_elapsed_wall_s"] >= d["cumulative_solve_wall_s"]
        for d in diagnostics
    )
    assert all(
        b["simulation_elapsed_wall_s"] >= a["simulation_elapsed_wall_s"]
        for a, b in zip(diagnostics, diagnostics[1:])
    )


def test_cli_solver_failure_does_not_replace_output(
    config, tmp_path, monkeypatch, capsys
):
    from gas_spring.cli import main

    physical, run = copy.deepcopy(config)
    run.update(input_parquet="input.parquet", output_parquet="output.parquet")
    pd.DataFrame(
        {
            "time_s": [0.0, 0.001],
            "actuator_a_force_n": [1729.0, 1729.0],
            "actuator_b_force_n": [1729.0, 1729.0],
            "duct_air_speed_m_s": [2.0, 2.0],
        }
    ).to_parquet(tmp_path / "input.parquet", index=False)
    physical_path, run_path = tmp_path / "physical.yaml", tmp_path / "run.yaml"
    physical_path.write_text(yaml.safe_dump(physical), encoding="utf-8")
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    output = tmp_path / "output.parquet"
    output.write_bytes(b"existing output")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gas-spring",
            "--physical",
            str(physical_path),
            "--run",
            str(run_path),
            "--simulate",
        ],
    )

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate solver failure")

    monkeypatch.setattr("gas_spring.solver.simulate", fail)
    with pytest.raises(RuntimeError, match="deliberate solver failure"):
        main()
    assert output.read_bytes() == b"existing output"


def test_real_cli_simulation_writes_sensor_and_truth_parquet(
    config, tmp_path, monkeypatch
):
    from gas_spring.cli import main

    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=0.001)
    run.update(input_parquet="input.parquet", output_parquet="output.parquet")
    pd.DataFrame(
        {
            "time_s": [0.0, 0.001],
            "actuator_a_force_n": [1729.0, 1729.0],
            "actuator_b_force_n": [1729.0, 1729.0],
            "duct_air_speed_m_s": [2.0, 2.0],
        }
    ).to_parquet(tmp_path / "input.parquet", index=False)
    physical_path, run_path = tmp_path / "physical.yaml", tmp_path / "run.yaml"
    physical_path.write_text(yaml.safe_dump(physical), encoding="utf-8")
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gas-spring",
            "--physical",
            str(physical_path),
            "--run",
            str(run_path),
            "--simulate",
        ],
    )
    main()
    output = pd.read_parquet(tmp_path / "output.parquet")
    assert len(output) == 2
    assert "sensor_actuator_a_force_n" in output
    assert "truth_sampled_energy_balance_residual_j" in output
    assert "truth_mass_flow_hydraulic_outlet_junction_to_blowoff_fluid_kg_s" in output


def test_coupled_blowoff_opens_and_reaches_stop(config):
    physical, run = copy.deepcopy(config)
    # Bring the initially closed blow-off close to its opening threshold so a
    # short synthetic run exercises contact recovery and the opposite elastic stop.
    run["initial_gas_gauge_pressure_bar"]["blowoff"] = 36
    run["blowoff_max_travel_m"] = 0.0002
    run["solver"].update(
        pipe_cells=1, actuator_wall_cells=2, max_step_s=0.001, relative_tolerance=0.0002
    )
    inputs = pd.DataFrame(
        {
            "time_s": [0.0, 0.001, 0.002, 0.03],
            "actuator_a_force_n": [1729.0, 5000.0, 8000.0, 8000.0],
            "actuator_b_force_n": [1729.0] * 4,
            "hot_ambient_c": [120.0] * 4,
            "duct_air_speed_m_s": [2.0] * 4,
        }
    )
    result = simulate(physical, run, inputs)
    travel = result["truth_blowoff_piston_position_m"].to_numpy()
    assert travel[-1] > run["blowoff_max_travel_m"]
    assert result["truth_blowoff_upper_stop_deflection_m"].iloc[-1] > 0
    reaction = result["truth_blowoff_end_stop_force_n"].iloc[-1]
    assert reaction == pytest.approx(
        -physical["end_stops"]["stiffness_n_m"]
        * (travel[-1] - run["blowoff_max_travel_m"])
    )


def test_isolated_warmup_has_expected_equilibrium_and_thermal_lag(config):
    physical, run = copy.deepcopy(config)
    run["hot_ambient"]["temperature_c"] = 50
    equilibrium = isolated_actuator_warmup(physical, run)
    assert all(value == pytest.approx(50) for value in equilibrium.values())
    run["hot_ambient"]["temperature_c"] = 120
    warming = isolated_actuator_warmup(physical, run)
    assert 100 < warming["housing_c"] < 105
    assert (
        warming["housing_c"] > warming["fluid_c"] > warming["piston_fluid_face_c"] > 50
    )


def test_short_forced_run_is_consistent_with_tighter_steps(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(
        pipe_cells=1, actuator_wall_cells=2, max_step_s=0.001, relative_tolerance=0.0002
    )
    inputs = pd.DataFrame(
        {
            "time_s": [0.0, 0.001, 0.002],
            "actuator_a_force_n": [1729.0, 2000.0, 2200.0],
            "actuator_b_force_n": [1729.0] * 3,
            "hot_ambient_c": [120.0] * 3,
            "duct_air_speed_m_s": [2.0] * 3,
        }
    )
    coarse = simulate(physical, run, inputs)
    run["solver"].update(max_step_s=0.0005, relative_tolerance=0.00005)
    fine = simulate(physical, run, inputs)
    assert coarse["truth_actuator_a_piston_position_m"].iloc[-1] == pytest.approx(
        fine["truth_actuator_a_piston_position_m"].iloc[-1], abs=1e-6
    )
    assert coarse["truth_actuator_a_fluid_pressure_bar"].iloc[-1] == pytest.approx(
        fine["truth_actuator_a_fluid_pressure_bar"].iloc[-1], abs=0.2
    )


def test_coupled_three_minute_warmup_is_finite_and_heats_actuator(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(
        pipe_cells=1, actuator_wall_cells=2, max_step_s=0.5, relative_tolerance=0.0001
    )
    times = np.linspace(0, 180, 61)
    inputs = pd.DataFrame(
        {
            "time_s": times,
            "actuator_a_force_n": 1729 + times * 0,
            "actuator_b_force_n": 1729 + times * 0,
            "hot_ambient_c": 120 + times * 0,
            "duct_air_speed_m_s": 2 + times * 0,
        }
    )
    output = simulate(physical, run, inputs)
    assert np.isfinite(output.select_dtypes(include=[np.number]).to_numpy()).all()
    assert 95 < output["truth_actuator_a_wall_0_temperature_c"].iloc[-1] < 110
    assert output["truth_actuator_a_fluid_temperature_c"].iloc[-1] > 80
    for channel in ("truth_liquid_mass_total_kg", "truth_nitrogen_mass_total_kg"):
        assert output[channel].iloc[-1] == pytest.approx(
            output[channel].iloc[0], rel=1e-8
        )
    energy_change = (
        output["truth_system_energy_j"].iloc[-1]
        - output["truth_system_energy_j"].iloc[0]
    )
    # This audit integrates power at the 3 s output sampling interval, so it
    # includes quadrature error as well as numerical integration error.
    assert (
        abs(output["truth_sampled_energy_balance_residual_j"].iloc[-1])
        < abs(energy_change) * 0.002 + 1
    )


def test_main_accumulator_stop_releases_and_preserves_incompressible_motion(config):
    physical, run = copy.deepcopy(config)
    physical["geometry"]["main_accumulator_expansion_stop_m"] = 1e-6
    physical["fluid"]["volumetric_expansion_per_k"] = 0.0
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=0.002)
    inputs = pd.DataFrame(
        dict(
            time_s=[0.0, 0.01, 0.03, 0.04, 0.07],
            actuator_a_force_n=[1729.0, -1000.0, -1000.0, 4000.0, 4000.0],
            actuator_b_force_n=[1729.0, -1000.0, -1000.0, 4000.0, 4000.0],
            hot_ambient_c=120.0,
            duct_air_speed_m_s=2.0,
        )
    )
    events = []
    output = simulate(
        physical, run, inputs, event_callback=lambda *args: events.append(args)
    )
    main = output.truth_main_accumulator_piston_position_m.to_numpy()
    assert main.max() > 1e-6
    assert output.truth_main_accumulator_upper_stop_deflection_m.max() > 0
    np.testing.assert_allclose(
        output.truth_main_gas_volume_m3,
        physical["geometry"]["main_gas_volume_at_zero_m3"] + piston_area(0.045) * main,
        atol=1e-15,
    )
    assert main[-1] < 1e-6 - 1e-7
    assert any(
        name == "main_accumulator" and kind == "contact_upper"
        for t, name, kind in events
    )
    assert any(
        name == "main_accumulator" and kind == "release_upper"
        for t, name, kind in events
    )
    area_fluid = piston_area(0.038, 0.014)
    area_main = piston_area(0.045)
    expected = (
        -area_fluid
        * (
            output.truth_actuator_a_piston_position_m
            + output.truth_actuator_b_piston_position_m
        )
        / area_main
        + piston_area(0.045, 0.012) * output.truth_blowoff_piston_position_m / area_main
    )
    np.testing.assert_allclose(main, expected, atol=1e-12)


def test_positive_load_extends_actuators_and_compresses_main_accumulator(config):
    from examples.make_input import zero_position_static_load_n

    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=0.005)
    balance = zero_position_static_load_n(physical, run)
    assert balance == pytest.approx(1729.446755958)
    times = np.linspace(0.0, 0.1, 11)
    forces = balance + (3000.0 - balance) * times / 0.1
    inputs = pd.DataFrame(
        dict(
            time_s=times,
            actuator_a_force_n=forces,
            actuator_b_force_n=forces,
            hot_ambient_c=120.0,
            duct_air_speed_m_s=2.0,
        )
    )
    output = simulate(physical, run, inputs)
    for name in ("actuator_a", "actuator_b"):
        assert output[f"truth_{name}_piston_position_m"].iloc[-1] > 0
        assert output[f"truth_{name}_fluid_pressure_bar"].iloc[-1] > 35.0
        channel = f"truth_mass_flow_hydraulic_inlet_junction_to_{name}_fluid_kg_s"
        assert output[channel].iloc[-1] < 0
    assert output.truth_main_accumulator_piston_position_m.iloc[-1] < 0
    assert (
        output.truth_main_gas_volume_m3.iloc[-1]
        < output.truth_main_gas_volume_m3.iloc[0]
    )
    assert output.truth_main_gas_pressure_bar.iloc[-1] > 35.0
    assert (
        output.truth_mass_flow_hydraulic_outlet_junction_to_main_fluid_kg_s.iloc[-1] > 0
    )
    assert output.truth_hydraulic_pipe_flow_0_m3_s.iloc[-1] > 0
    assert output.truth_minimum_liquid_absolute_pressure_pa.min() > 0


def test_positive_load_power_and_internal_energy_rates_balance(config):
    from gas_spring.coupling import PISTONS
    from gas_spring.solver import _record

    physical, run = copy.deepcopy(config)
    physical["fluid"]["volumetric_expansion_per_k"] = 0.0
    system, state, u = instantaneous((physical, run))
    state["velocity"].update(actuator_a=0.003, actuator_b=0.002, blowoff=0.001)
    rate = system.evaluate(state, u)
    velocity = state["velocity"]
    acceleration = rate["velocity_rate_m_s2"]
    total = sum(
        node["energy_rate_w"]
        for group in ("liquid", "gas")
        for node in rate[group]["node_rates"].values()
    )
    total += sum(rate["thermal"]["solid_heat_w"].values())
    total += sum(
        effective_piston_mass_kg(physical, n) * velocity[n] * acceleration[n]
        for n in PISTONS
    )
    length = system.g["hydraulic_pipe_length_m"] / system.cells
    rho = physical["fluid"]["density_kg_m3_at_20c"]
    area_fluid = -system.j["actuator_a"]
    for i, q in enumerate(rate["flows"]["pipe"]):
        inertia = pipe_inertance(
            rho,
            length * (0.5 if i in (0, system.cells) else 1),
            system.g["hydraulic_pipe_diameter_m"],
        )
        total += (
            inertia
            * q
            * area_fluid
            * (acceleration["actuator_a"] + acceleration["actuator_b"])
        )
    inertia = pipe_inertance(
        rho,
        system.g["blowoff_connection_length_m"],
        system.g["blowoff_connection_diameter_m"],
    )
    total += (
        inertia
        * rate["flows"]["blowoff"]
        * system.j["blowoff"]
        * acceleration["blowoff"]
    )
    total -= sum(rate["end_stops"][n]["force_n"] * velocity[n] for n in PISTONS)
    gas_area = piston_area(0.038)
    blowoff_gas_area = piston_area(0.045)
    volume_rate = (gas_area - area_fluid) * (
        velocity["actuator_a"] + velocity["actuator_b"]
    ) + (system.j["blowoff"] - blowoff_gas_area) * velocity["blowoff"]
    expected = (
        rate["thermal"]["external_heat_w"]
        + sum(u[n + "_force_n"] * velocity[n] for n in ("actuator_a", "actuator_b"))
        - ATMOSPHERIC_PRESSURE_PA * volume_rate
    )
    assert total == pytest.approx(expected, abs=1e-6)
    assert _record(state, rate, 0.0, system, u)[
        "truth_external_power_w"
    ] == pytest.approx(expected)
