import copy
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

from gas_spring.config import load_yaml
from gas_spring.hydraulics import (piston_area, DigressiveCurve, restriction_flow, pipe_inertance,
                                   regularized_restriction_pressure_drop, gas_pipe_mass_flow)
from gas_spring.balances import advective_flux_rates, ideal_gas_state, ideal_gas_energy_rate
from gas_spring.circuit import circuit_nodes_from_initial_state, hydraulic_rates, nitrogen_rates
from gas_spring.coupling import coupled_rates
from gas_spring.initialization import initialize_system
from gas_spring.force_input import interpolate_inputs, shaft_position_m
from gas_spring.diagnostics import isolated_actuator_warmup
from gas_spring.materials import (ATMOSPHERIC_PRESSURE_PA, liquid_density, liquid_pressure,
                                  liquid_viscosity, liquid_internal_energy, liquid_state,
                                  nitrogen_mass, nitrogen_pressure)
from gas_spring.mechanics import chamber_volumes, initial_static_equilibrium, piston_acceleration
from gas_spring.sensors import CHANNELS, add_sensor_measurements
from gas_spring.thermal import actuator_wall_cell_areas, exchange_heat_rates
from gas_spring.thermal_coupling import thermal_exchange
from gas_spring.solver import simulate


@pytest.fixture
def config():
    return load_yaml("config/physical.yaml"), load_yaml("config/run.yaml")


def test_liquid_eos_roundtrip_and_temperature_dependence(config):
    fluid = config[0]["fluid"]
    p = np.array([101325., 3.6e6, 8.1e6])
    t = np.array([20., 50., 100.])
    rho = liquid_density(p, t, fluid)
    np.testing.assert_allclose(liquid_pressure(rho, t, fluid), p, rtol=1e-11)
    assert liquid_density(3e6, 100, fluid) < liquid_density(3e6, 20, fluid)
    np.testing.assert_allclose(liquid_viscosity([20, 100], fluid), [.01, .0021])
    assert liquid_viscosity(50, fluid) < liquid_viscosity(20, fluid)


def test_nitrogen_ideal_gas_uses_absolute_pressure(config):
    gas = config[0]["gas"]
    p = ATMOSPHERIC_PRESSURE_PA + 80e5
    m = nitrogen_mass(p, .001, 25, gas)
    assert nitrogen_pressure(m, .001, 25, gas) == pytest.approx(p)
    assert nitrogen_pressure(m, .0005, 25, gas) == pytest.approx(2*p)


def test_chamber_volumes_and_differential_shaft_areas(config):
    g = config[0]["geometry"]
    zero = chamber_volumes(g)
    moved = chamber_volumes(g, actuator_a_m=.05, actuator_b_m=-.015, main_m=.007, blowoff_m=.05)
    fluid_area = piston_area(.038, .014)
    gas_area = piston_area(.038)
    assert moved["actuator_a_fluid"] == pytest.approx(fluid_area * .005)
    assert moved["actuator_b_gas"] == pytest.approx(gas_area * .005)
    assert moved["main_gas"] - zero["main_gas"] == pytest.approx(piston_area(.045)*.007)
    assert moved["blowoff_gas"] - zero["blowoff_gas"] == pytest.approx(-piston_area(.045)*.05)
    assert moved["blowoff_fluid"] - zero["blowoff_fluid"] == pytest.approx(piston_area(.045, .012)*.05)
    with pytest.raises(ValueError, match="volume"):
        chamber_volumes(g, actuator_a_m=.056)


def test_static_initialization_accepts_balance_and_rejects_conflict(config):
    physical, run = config
    state = initial_static_equilibrium(physical, run, -1729, -1729)
    assert state.hydraulic_gauge_pressure_pa == pytest.approx(35e5)
    for name in ("actuator_a", "actuator_b", "main_accumulator"):
        assert abs(state.residual_forces_n[name]) <= state.friction_limits_n[name]
    # The zero stop supports negative residual on the blow-off piston.
    assert state.residual_forces_n["blowoff"] < -state.friction_limits_n["blowoff"]
    with pytest.raises(ValueError, match="static equilibrium"):
        initial_static_equilibrium(physical, run, 0, 20000)


def make_truth():
    data = {"time_s": [0., 60., 120.], "hot_ambient_c": [20., 30., 40.],
            "cooler_region_temperature_c": [20., 22., 24.]}
    for name, _, unit, _ in CHANNELS:
        data[f"truth_{name}_{unit}"] = [1., 1., 1.]
    return pd.DataFrame(data)


def test_sensor_channels_apply_independent_ambient_and_calibration(config):
    physical, run = copy.deepcopy(config)
    for profile in physical["sensor_profiles"].values():
        profile["noise_std"] = profile["drift_std_per_sqrt_hour"] = 0
    run["sensor_calibration"]["actuator_a_displacement"] = {"gain": 2, "offset": 3, "temperature_coefficient": .1, "reference_temperature_c": 20}
    run["sensor_calibration"]["blowoff_displacement"] = {"temperature_coefficient": .5, "reference_temperature_c": 20}
    data = add_sensor_measurements(make_truth(), physical, run)
    np.testing.assert_allclose(data["sensor_actuator_a_displacement_mm"], [5, 6, 7])
    np.testing.assert_allclose(data["sensor_blowoff_displacement_mm"], [1, 2, 3])
    assert len([c for c in data if c.startswith("sensor_")]) == len(CHANNELS)


def test_sensor_channels_reproduce_and_do_not_share_noise(config):
    physical, run = config
    one = add_sensor_measurements(make_truth(), physical, run)
    two = add_sensor_measurements(make_truth(), physical, run)
    pd.testing.assert_frame_equal(one, two)
    assert not np.array_equal(one["sensor_actuator_a_force_n"], one["sensor_actuator_b_force_n"])


def test_internal_mass_and_enthalpy_fluxes_conserve_with_flow_reversal():
    dm, du = advective_flux_rates([100, 200, 300], [(0, 1, 2), (1, 2, -3)])
    np.testing.assert_allclose(dm, [-2, 5, -3])
    np.testing.assert_allclose(du, [-200, 1100, -900])
    assert sum(dm) == pytest.approx(0)
    assert sum(du) == pytest.approx(0)


def test_gas_energy_and_pressure_work(config):
    gas = config[0]["gas"]
    mass, temp_k, volume = 0.02, 300.0, .001
    cv = gas["specific_heat_cp_j_kg_k"] - gas["specific_gas_constant_j_kg_k"]
    p, temp_c = ideal_gas_state(mass, mass * cv * temp_k, volume, gas)
    assert temp_c == pytest.approx(26.85)
    assert p == pytest.approx(mass * gas["specific_gas_constant_j_kg_k"] * temp_k / volume)
    assert ideal_gas_energy_rate(p, 1e-6, 0, 0) == pytest.approx(-p*1e-6)


def test_wall_wetting_and_heat_exchange_conserve_energy():
    f, g = actuator_wall_cell_areas(.038, .055, .020, .012, 7)
    assert sum(f) == pytest.approx(np.pi*.038*.043)
    assert sum(g) == pytest.approx(np.pi*.038*.032)
    heat = exchange_heat_rates([100, 50, 25], [(0, 1, 2), (1, 2, 3)])
    np.testing.assert_allclose(heat, [-100, 25, 75])
    assert sum(heat) == pytest.approx(0)


def test_initial_state_contains_balanced_masses_and_pipe_gradients(config):
    physical, run = config
    state = initialize_system(physical, run, -1729, -1729, pipe_cells=4)
    assert state["liquid_pressure_pa"] == pytest.approx(ATMOSPHERIC_PRESSURE_PA + 35e5)
    a = state["gas_chambers"]["actuator_a_gas"]
    assert nitrogen_pressure(a["mass_kg"], a["volume_m3"], a["temperature_c"], physical["gas"]) == pytest.approx(ATMOSPHERIC_PRESSURE_PA + 15e5)
    np.testing.assert_allclose(state["pipes"]["gas"]["temperature_c"], [46.875, 40.625, 34.375, 28.125])
    assert np.all(state["pipes"]["hydraulic"]["mass_kg"] > 0)
    fluid = state["liquid_chambers"]["actuator_a_fluid"]
    p, t = liquid_state(fluid["mass_kg"], fluid["internal_energy_j"], fluid["volume_m3"], physical["fluid"])
    assert p == pytest.approx(state["liquid_pressure_pa"])
    assert t == pytest.approx(50)
    with pytest.raises(ValueError, match="static equilibrium"):
        initialize_system(physical, run, 0, 20000)


def test_liquid_energy_relation_includes_thermal_expansion_work(config):
    fluid = config[0]["fluid"]
    temp, pressure = 50., 3.6e6
    density = liquid_density(pressure, temp, fluid)
    volume = 1/density
    dv = 1e-8
    e_plus = liquid_internal_energy(temp, 1/(volume+dv), fluid)
    e_minus = liquid_internal_energy(temp, 1/(volume-dv), fluid)
    derivative = (e_plus-e_minus)/(2*dv)
    expected = (temp + 273.15)*fluid["volumetric_expansion_per_k"]*fluid["bulk_modulus_pa"]-pressure
    assert derivative == pytest.approx(expected, rel=1e-7)


def test_restriction_inverse_and_extrapolation_match_both_fit_points():
    area = piston_area(.038, .014)
    c = DigressiveCurve.from_points(area, .15, 4000, .8, 6000)
    e = DigressiveCurve.from_points(area, .15, 5000, .8, 7000)
    for flow in (area*.15, area*.8, -area*.15, -area*.8, area*10, -area*10):
        dp = regularized_restriction_pressure_drop(flow, c, e)
        assert dp*flow > 0
        assert restriction_flow(dp, c, e) == pytest.approx(flow)
    assert regularized_restriction_pressure_drop(area*.8, c, e)*area == pytest.approx(6000)
    assert regularized_restriction_pressure_drop(-area*.8, c, e)*area == pytest.approx(-7000)


def test_gas_pipe_flow_is_bidirectional_and_zero_when_balanced():
    args = (300., 296.8, 1.78e-5, 1., .004)
    forward = gas_pipe_mass_flow(1.7e6, 1.6e6, *args)
    reverse = gas_pipe_mass_flow(1.6e6, 1.7e6, *args)
    assert forward > 0
    assert reverse == pytest.approx(-forward)
    assert gas_pipe_mass_flow(1.6e6, 1.6e6, *args) == 0


def test_piston_stiction_sliding_and_stops():
    assert piston_acceleration(0, 0, 9, 10, .1, -1, 1) == 0
    assert piston_acceleration(0, 0, 12, 10, .1, -1, 1) == pytest.approx(20)
    assert piston_acceleration(0, 0, 12, 10, .1, -1, 1, kinetic_friction_limit_n=8) == pytest.approx(40)
    assert piston_acceleration(0, .1, -10, 5, .1, -1, 1) == pytest.approx(-150)
    assert piston_acceleration(-1, 0, -100, 10, .1, -1, 1) == 0
    assert piston_acceleration(1, 0, 100, 10, .1, -1, 1) == 0
    with pytest.raises(ValueError, match="impact"):
        piston_acceleration(1, .1, 0, 10, .1, -1, 1)


def test_forcing_interpolation_and_shaft_deflection():
    data = pd.DataFrame({"time_s": [0, 2], "actuator_a_force_n": [1000, 2000],
                         "actuator_b_force_n": [-100, 100], "hot_ambient_c": [100, 120],
                         "duct_air_speed_m_s": [1, 3]})
    interpolated = interpolate_inputs(data, 1)
    assert interpolated == {"actuator_a_force_n": 1500, "actuator_b_force_n": 0,
                            "hot_ambient_c": 110, "duct_air_speed_m_s": 2}
    assert shaft_position_m(.01, 2000, 20e6) == pytest.approx(.0099)
    with pytest.raises(ValueError, match="outside"):
        interpolate_inputs(data, 3)


def test_complete_circuit_rate_assembly_conserves_internal_mass_and_energy(config):
    physical, run = config
    state = initialize_system(physical, run, -1729, -1729, pipe_cells=3)
    liquid_nodes, gas_nodes = circuit_nodes_from_initial_state(state)
    flow = np.full(4, 1e-6)
    h = hydraulic_rates(liquid_nodes, flow, physical)
    g = nitrogen_rates(gas_nodes, 3, physical)
    assert sum(n["mass_rate_kg_s"] for n in h["node_rates"].values()) == pytest.approx(0, abs=1e-14)
    assert sum(n["mass_rate_kg_s"] for n in g["node_rates"].values()) == pytest.approx(0, abs=1e-14)
    assert sum(n["energy_rate_w"] for n in g["node_rates"].values()) == pytest.approx(0, abs=1e-10)
    geo = physical["geometry"]
    inertia = [pipe_inertance(physical["fluid"]["density_kg_m3_at_20c"],
                              geo["hydraulic_pipe_length_m"]/3*(.5 if i in (0, 3) else 1),
                              geo["hydraulic_pipe_diameter_m"]) for i in range(4)]
    kinetic_rate = np.dot(np.asarray(inertia)*flow, h["pipe_flow_rate_m3_s2"])
    assert sum(n["energy_rate_w"] for n in h["node_rates"].values()) + kinetic_rate == pytest.approx(0, abs=1e-8)
    # Two actuator restrictions go into their own nodes. The blow-off branch
    # and main inlet both connect at the far, downstream junction.
    pairs = {(a, b) for a, b, _ in h["mass_flows_kg_s"]}
    assert ("hydraulic_inlet_junction", "actuator_a_fluid") in pairs
    assert ("hydraulic_inlet_junction", "actuator_b_fluid") in pairs
    assert ("hydraulic_outlet_junction", "main_fluid") in pairs
    assert ("hydraulic_outlet_junction", "blowoff_fluid") in pairs


def test_reservoir_pressure_difference_reverses_gas_pipe_flow(config):
    physical, run = config
    state = initialize_system(physical, run, -1729, -1729, pipe_cells=2)
    _, gas = circuit_nodes_from_initial_state(state)
    reservoir = gas["low_pressure_reservoir_gas"]
    for field in ("mass_kg", "internal_energy_j", "boundary_energy_j"):
        reservoir[field] *= 1.01
    rates = nitrogen_rates(gas, 2, physical)
    connection = [flow for a, b, flow in rates["mass_flows_kg_s"] if b == "low_pressure_reservoir_gas"]
    assert len(connection) == 1 and connection[0] < 0


def test_coupled_rates_preserve_static_equilibrium_and_pressure_work(config):
    physical, run = config
    state = initialize_system(physical, run, -1729, -1729, pipe_cells=2)
    liquid, gas = circuit_nodes_from_initial_state(state)
    result = coupled_rates(physical, run, liquid, gas, state["pipes"]["hydraulic"]["flow_m3_s"],
                           state["position_m"], state["velocity_m_s"], state["applied_force_n"])
    assert all(value == pytest.approx(0) for value in result["velocity_rate_m_s2"].values())
    velocity = {"actuator_a": .001, "actuator_b": -.002, "main_accumulator": .001, "blowoff": .001}
    result = coupled_rates(physical, run, liquid, gas, state["pipes"]["hydraulic"]["flow_m3_s"],
                           state["position_m"], velocity, state["applied_force_n"])
    fluid_gas_energy_rate = sum(x["energy_rate_w"] for group in (result["liquid"], result["gas"])
                                for x in group["node_rates"].values())
    kinetic_rate = sum(physical["moving_masses_kg"][name] * velocity[name] * acceleration
                       for name, acceleration in result["velocity_rate_m_s2"].items())
    friction_heat = sum(result["piston_friction_heat_w"].values())
    af, ag = piston_area(.038, .014), piston_area(.038)
    bf, bg = piston_area(.045, .012), piston_area(.045)
    volume_rate = (ag-af)*(velocity["actuator_a"]+velocity["actuator_b"]) + (bf-bg)*velocity["blowoff"]
    external_power = -sum(state["applied_force_n"][name]*velocity[name] for name in ("actuator_a", "actuator_b"))
    assert fluid_gas_energy_rate + kinetic_rate + friction_heat == pytest.approx(external_power - ATMOSPHERIC_PRESSURE_PA*volume_rate, abs=1e-6)


def test_thermal_network_and_coupled_rates_balance_energy(config):
    physical, run = config
    state = initialize_system(physical, run, -1729, -1729, pipe_cells=2, actuator_wall_cells=3)
    liquid, gas = circuit_nodes_from_initial_state(state)
    # A warmer boundary and cooler core represent a nonuniform gas temperature.
    gas["actuator_a_gas"]["boundary_energy_j"] *= 1.01
    liquid["actuator_a_fluid"]["boundary_energy_j"] *= 1.01
    thermal = thermal_exchange(physical, run, liquid, gas, state["solid_temperatures_c"],
                               state["position_m"], 120, 2)
    internal_heat = sum(thermal["liquid_heat_w"].values()) + sum(thermal["gas_heat_w"].values()) + sum(thermal["solid_heat_w"].values())
    assert internal_heat == pytest.approx(thermal["external_heat_w"], abs=1e-9)
    result = coupled_rates(physical, run, liquid, gas, state["pipes"]["hydraulic"]["flow_m3_s"],
                           state["position_m"], state["velocity_m_s"], state["applied_force_n"],
                           state["solid_temperatures_c"], 120, 2)
    fluid_gas_energy = sum(x["energy_rate_w"] for group in (result["liquid"], result["gas"])
                           for x in group["node_rates"].values())
    assert fluid_gas_energy + sum(result["thermal"]["solid_heat_w"].values()) == pytest.approx(result["thermal"]["external_heat_w"], abs=1e-6)
    assert "boundary_energy_rate_w" in result["gas"]["node_rates"]["actuator_a_gas"]
    assert "boundary_energy_rate_w" in result["liquid"]["node_rates"]["actuator_a_fluid"]
    assert result["liquid"]["boundary_temperatures_c"]["actuator_a_fluid"] > result["liquid"]["temperatures_c"]["actuator_a_fluid"]


def test_short_coupled_run_generates_truth_and_sensor_channels(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=.001)
    inputs = pd.DataFrame({"time_s": [0., .001], "actuator_a_force_n": [-1729., -1729.],
                           "actuator_b_force_n": [-1729., -1729.], "hot_ambient_c": [120., 120.],
                           "duct_air_speed_m_s": [2., 2.]})
    result = simulate(physical, run, inputs)
    assert len(result) == 2
    assert result["truth_actuator_a_fluid_pressure_bar"].iloc[0] == pytest.approx(35)
    assert result["truth_low_pressure_gas_temperature_c"].iloc[0] == pytest.approx(25)
    assert result["truth_actuator_a_gas_temperature_c"].iloc[0] == pytest.approx(50)
    assert result["truth_actuator_a_displacement_mm"].iloc[0] == pytest.approx(1729/20e6*1000)
    assert result["truth_actuator_a_wall_0_temperature_c"].iloc[-1] > 50
    assert len([c for c in result if c.startswith("sensor_")]) == len(CHANNELS)


def test_cli_writes_parquet_atomically_and_protects_input(config, tmp_path, monkeypatch):
    from gas_spring.cli import main
    physical, run = copy.deepcopy(config)
    inputs = pd.DataFrame({"time_s": [0., .001], "actuator_a_force_n": [-1729., -1729.],
                           "actuator_b_force_n": [-1729., -1729.], "duct_air_speed_m_s": [2., 2.]})
    source = tmp_path / "input.parquet"
    inputs.to_parquet(source, index=False)
    physical_path, run_path = tmp_path / "physical.yaml", tmp_path / "run.yaml"
    physical_path.write_text(yaml.safe_dump(physical), encoding="utf-8")
    run.update(input_parquet="input.parquet", output_parquet="output.parquet")
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    monkeypatch.setattr("gas_spring.solver.simulate", lambda p, r, data: pd.DataFrame({"time_s": data.time_s, "truth_example": [3., 4.]}))
    monkeypatch.setattr(sys, "argv", ["gas-spring", "--physical", str(physical_path), "--run", str(run_path), "--simulate"])
    main()
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "output.parquet"), pd.DataFrame({"time_s": [0., .001], "truth_example": [3., 4.]}))
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


def test_real_cli_simulation_writes_sensor_and_truth_parquet(config, tmp_path, monkeypatch):
    from gas_spring.cli import main
    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=.001)
    run.update(input_parquet="input.parquet", output_parquet="output.parquet")
    pd.DataFrame({"time_s": [0., .001], "actuator_a_force_n": [-1729., -1729.],
                  "actuator_b_force_n": [-1729., -1729.], "duct_air_speed_m_s": [2., 2.]}).to_parquet(tmp_path / "input.parquet", index=False)
    physical_path, run_path = tmp_path / "physical.yaml", tmp_path / "run.yaml"
    physical_path.write_text(yaml.safe_dump(physical), encoding="utf-8")
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gas-spring", "--physical", str(physical_path), "--run", str(run_path), "--simulate"])
    main()
    output = pd.read_parquet(tmp_path / "output.parquet")
    assert len(output) == 2
    assert "sensor_actuator_a_force_n" in output
    assert "truth_sampled_energy_balance_residual_j" in output
    assert "truth_mass_flow_hydraulic_outlet_junction_to_blowoff_fluid_kg_s" in output


def test_coupled_blowoff_opens_and_reaches_stop(config):
    physical, run = copy.deepcopy(config)
    # Bring the initially closed blow-off close to its opening threshold so a
    # short synthetic run exercises both release and stop-impact events.
    run["initial_gas_gauge_pressure_bar"]["blowoff"] = 36
    run["blowoff_max_travel_m"] = 1e-6
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=.001, relative_tolerance=.0002)
    inputs = pd.DataFrame({"time_s": [0., .001, .002, .003],
                           "actuator_a_force_n": [-1729., -5000., -8000., -8000.],
                           "actuator_b_force_n": [-1729.]*4, "hot_ambient_c": [120.]*4,
                           "duct_air_speed_m_s": [2.]*4})
    result = simulate(physical, run, inputs)
    travel = result["truth_blowoff_piston_position_m"].to_numpy()
    assert np.min(travel) >= -1e-12
    assert np.max(travel) <= 1e-6+1e-12
    assert travel[-1] == pytest.approx(1e-6, abs=1e-10)


def test_isolated_warmup_has_expected_equilibrium_and_thermal_lag(config):
    physical, run = copy.deepcopy(config)
    run["hot_ambient"]["temperature_c"] = 50
    equilibrium = isolated_actuator_warmup(physical, run)
    assert all(value == pytest.approx(50) for value in equilibrium.values())
    run["hot_ambient"]["temperature_c"] = 120
    warming = isolated_actuator_warmup(physical, run)
    assert 100 < warming["housing_c"] < 105
    assert warming["housing_c"] > warming["fluid_c"] > warming["piston_fluid_face_c"] > 50


def test_short_forced_run_is_consistent_with_tighter_steps(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=.001, relative_tolerance=.0002)
    inputs = pd.DataFrame({"time_s": [0., .001, .002],
                           "actuator_a_force_n": [-1729., -2000., -2200.],
                           "actuator_b_force_n": [-1729.]*3, "hot_ambient_c": [120.]*3,
                           "duct_air_speed_m_s": [2.]*3})
    coarse = simulate(physical, run, inputs)
    run["solver"].update(max_step_s=.0005, relative_tolerance=.00005)
    fine = simulate(physical, run, inputs)
    assert coarse["truth_actuator_a_piston_position_m"].iloc[-1] == pytest.approx(fine["truth_actuator_a_piston_position_m"].iloc[-1], abs=1e-6)
    assert coarse["truth_actuator_a_fluid_pressure_bar"].iloc[-1] == pytest.approx(fine["truth_actuator_a_fluid_pressure_bar"].iloc[-1], abs=.2)


def test_coupled_three_minute_warmup_is_finite_and_heats_actuator(config):
    physical, run = copy.deepcopy(config)
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, max_step_s=.5, relative_tolerance=.0001)
    times = np.linspace(0, 180, 61)
    inputs = pd.DataFrame({"time_s": times, "actuator_a_force_n": -1729+times*0,
                           "actuator_b_force_n": -1729+times*0, "hot_ambient_c": 120+times*0,
                           "duct_air_speed_m_s": 2+times*0})
    output = simulate(physical, run, inputs)
    assert np.isfinite(output.select_dtypes(include=[np.number]).to_numpy()).all()
    assert 95 < output["truth_actuator_a_wall_0_temperature_c"].iloc[-1] < 110
    assert output["truth_actuator_a_fluid_temperature_c"].iloc[-1] > 80
    for channel in ("truth_liquid_mass_total_kg", "truth_nitrogen_mass_total_kg"):
        assert output[channel].iloc[-1] == pytest.approx(output[channel].iloc[0], rel=1e-8)
    energy_change = output["truth_system_energy_j"].iloc[-1] - output["truth_system_energy_j"].iloc[0]
    # This audit integrates power at the 3 s output sampling interval, so it
    # includes quadrature error as well as numerical integration error.
    assert abs(output["truth_sampled_energy_balance_residual_j"].iloc[-1]) < abs(energy_change)*.002 + 1
