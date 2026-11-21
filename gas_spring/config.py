"""Load and validate inputs and the two YAML configurations."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .hydraulics import DigressiveCurve, piston_area
from .materials import liquid_internal_energy, liquid_density, liquid_viscosity
from .mechanics import chamber_volumes, roll_spring_stiffness_n_m, seal_friction_limit

CORE_COLUMNS = ("time_s", "actuator_a_force_n", "actuator_b_force_n", "duct_air_speed_m_s")
COLUMNS = CORE_COLUMNS + ("hot_ambient_c",)


def _hot_ambient_settings(run):
    settings = run.get("hot_ambient", {"source": "parquet", "column": "hot_ambient_c"})
    if not isinstance(settings, dict):
        raise ValueError("hot_ambient must be a YAML mapping")
    source = settings.get("source", "parquet")
    if source not in ("constant", "parquet"):
        raise ValueError("hot_ambient.source must be 'constant' or 'parquet'")
    if source == "constant":
        try:
            temperature = float(settings["temperature_c"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("constant hot_ambient requires a numeric temperature_c") from exc
        if not np.isfinite(temperature) or temperature <= -273.15:
            raise ValueError("hot_ambient.temperature_c must be above absolute zero")
    else:
        column = settings.get("column", "hot_ambient_c")
        if not isinstance(column, str) or not column or column in CORE_COLUMNS:
            raise ValueError("Parquet hot_ambient requires a distinct, non-empty column name")
    return settings, source

def load_yaml(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value

def load_inputs(run, run_path):
    path = (Path(run_path).resolve().parent / run["input_parquet"]).resolve()
    frame = pd.read_parquet(path)
    settings, source = _hot_ambient_settings(run)
    ambient_column = settings.get("column", "hot_ambient_c")
    required = CORE_COLUMNS + ((ambient_column,) if source == "parquet" else ())
    missing = set(required)-set(frame.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError("input must have at least two rows")
    try:
        data_columns = required
        data = frame.loc[:, list(data_columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("input columns must be numeric") from exc
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("input values must be finite and time strictly increasing")
    if np.any(data[:, CORE_COLUMNS.index("duct_air_speed_m_s")] < 0):
        raise ValueError("duct air speed cannot be negative")
    if source == "parquet" and np.any(data[:, len(CORE_COLUMNS)] <= -273.15):
        raise ValueError("hot ambient temperature must be above absolute zero")
    if source == "constant":
        frame["hot_ambient_c"] = float(settings["temperature_c"])
    elif ambient_column != "hot_ambient_c":
        frame["hot_ambient_c"] = frame[ambient_column]
    return frame

def validate_config(physical, run):
    roll_spring_stiffness_n_m(run)
    g = physical["geometry"]
    area = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"])
    if g.get("actuator_count") != 2:
        raise ValueError("geometry.actuator_count must be 2")
    for key in ("actuator_fluid_chamber_length_at_zero_m",
                "actuator_gas_chamber_length_at_zero_m",
                "actuator_extension_travel_m", "actuator_compression_travel_m",
                "main_accumulator_piston_diameter_m", "main_gas_volume_at_zero_m3",
                "main_accumulator_expansion_stop_m", "blowoff_gas_volume_at_zero_stop_m3",
                "actuator_input_shaft_stiffness_n_m", "hydraulic_pipe_diameter_m", "hydraulic_pipe_length_m",
                "gas_pipe_diameter_m", "gas_pipe_length_m", "low_pressure_reservoir_m3"):
        if key not in g or not np.isfinite(float(g[key])) or float(g[key]) <= 0:
            raise ValueError(f"geometry.{key} must be positive")
    for key in ("actuator_fluid_dead_volume_m3", "actuator_gas_dead_volume_m3"):
        if key not in g or not np.isfinite(float(g[key])) or float(g[key]) < 0:
            raise ValueError(f"geometry.{key} must be non-negative")
    for key in ("main_accumulator_fluid_volume_at_zero_m3", "blowoff_fluid_volume_at_zero_m3",
                "hydraulic_junction_volume_m3", "gas_junction_volume_m3"):
        if key not in g or not np.isfinite(float(g[key])) or float(g[key]) <= 0:
            raise ValueError(f"geometry.{key} must be positive")
    chamber_volumes(g, actuator_a_m=g["actuator_extension_travel_m"],
                    actuator_b_m=-g["actuator_compression_travel_m"],
                    main_m=g["main_accumulator_expansion_stop_m"])
    liquid_internal_energy(20.0, physical["fluid"])
    liquid_density(20.0, physical["fluid"])
    liquid_viscosity(20.0, physical["fluid"])
    gas = physical["gas"]
    if gas.get("type") != "nitrogen" or not 0 < float(gas["specific_gas_constant_j_kg_k"]) < float(gas["specific_heat_cp_j_kg_k"]):
        raise ValueError("nitrogen gas properties must have cp > R > 0")
    for key in ("actuator", "main_accumulator", "blowoff"):
        seal_friction_limit(0, 0, physical["seal_friction"][key])
    friction_ratio = float(physical["seal_friction"]["kinetic_to_static_ratio"])
    if not np.isfinite(friction_ratio) or not 0 <= friction_ratio <= 1:
        raise ValueError("seal_friction.kinetic_to_static_ratio must be between zero and one")
    thermal = physical["thermal"]
    for key in ("gas_boundary_mass_fraction", "liquid_boundary_mass_fraction"):
        if not np.isfinite(float(thermal[key])) or not 0 < float(thermal[key]) < 1:
            raise ValueError(f"thermal.{key} must be between zero and one")
    for key in ("actuator_housing_mass_kg_each", "actuator_outer_heat_transfer_area_m2_each",
                "actuator_outer_diameter_m", "metal_specific_heat_j_kg_k", "metal_density_kg_m3",
                "pipe_outer_diameter_m", "main_accumulator_housing_mass_kg", "blowoff_housing_mass_kg",
                "low_pressure_reservoir_housing_mass_kg", "main_accumulator_exposed_area_m2",
                "blowoff_exposed_area_m2", "low_pressure_reservoir_exposed_area_m2", "cooler_region_heat_capacity_j_k"):
        if not np.isfinite(float(thermal[key])) or float(thermal[key]) <= 0:
            raise ValueError(f"thermal.{key} must be positive")
    for key in ("actuator_hot_ambient_to_housing_w_m2_k", "metal_thermal_conductivity_w_m_k",
                "fluid_to_housing_w_m2_k", "liquid_core_to_boundary_w_m2_k", "gas_to_housing_w_m2_k",
                "gas_core_to_boundary_w_m2_k", "actuator_piston_through_conductance_w_k",
                "main_accumulator_piston_through_conductance_w_k", "blowoff_piston_through_conductance_w_k",
                "hydraulic_pipe_fluid_to_wall_w_m2_k", "gas_pipe_gas_to_wall_w_m2_k", "pipe_wall_to_surroundings_w_m2_k",
                "accumulator_and_reservoir_housing_to_cooler_w_m2_k", "cooler_base_conductance_w_k",
                "cooler_air_speed_conductance_w_k_per_m_s"):
        if not np.isfinite(float(thermal[key])) or float(thermal[key]) < 0:
            raise ValueError(f"thermal.{key} must be non-negative")
    if thermal["actuator_outer_diameter_m"] <= g["actuator_piston_diameter_m"] or thermal["pipe_outer_diameter_m"] <= max(g["hydraulic_pipe_diameter_m"], g["gas_pipe_diameter_m"]):
        raise ValueError("thermal outside diameters must exceed their bores")
    for key in ("blowoff_connection_length_m", "blowoff_connection_diameter_m", "actuator_gas_connection_length_m", "actuator_gas_connection_diameter_m"):
        if not np.isfinite(float(g[key])) or float(g[key]) <= 0:
            raise ValueError(f"geometry.{key} must be positive")
    stiffness = float(physical.get("end_stops", {}).get("stiffness_n_m", 1e7))
    if not np.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("end_stops.stiffness_n_m must be positive")
    masses = physical["moving_masses_kg"]
    for key in ("actuator_a", "actuator_b", "blowoff", "main_accumulator"):
        try:
            mass = float(masses[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"moving_masses_kg.{key} must be positive") from exc
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"moving_masses_kg.{key} must be positive")
    attached_masses = physical.get("attached_masses_kg", {})
    if not isinstance(attached_masses, dict):
        raise ValueError("attached_masses_kg must be a mapping")
    unknown_attached = set(attached_masses) - {"actuator_a", "actuator_b", "blowoff", "main_accumulator"}
    if unknown_attached:
        raise ValueError(f"unknown attached_masses_kg entries: {sorted(unknown_attached)}")
    for key, value in attached_masses.items():
        try:
            mass = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"attached_masses_kg.{key} must be non-negative") from exc
        if isinstance(value, bool) or not np.isfinite(mass) or mass < 0:
            raise ValueError(f"attached_masses_kg.{key} must be non-negative")
    solver = run.get("solver", {})
    if not isinstance(solver, dict):
        raise ValueError("solver must be a mapping")
    if solver.get("friction_model", "smooth") != "smooth":
        raise ValueError("solver.friction_model must be 'smooth'")
    try:
        velocity_scale = float(solver.get("friction_velocity_scale_m_s", 0.005))
    except (TypeError, ValueError) as exc:
        raise ValueError("solver.friction_velocity_scale_m_s must be positive") from exc
    if not np.isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("solver.friction_velocity_scale_m_s must be positive")
    temperatures = run["initial_temperatures_c"]
    for key in ("actuator_a_fluid", "actuator_b_fluid", "actuator_a_gas", "actuator_b_gas",
                "main_accumulator_fluid", "blowoff_fluid", "main_accumulator_gas",
                "blowoff_gas", "low_pressure_reservoir_gas", "actuator_a_housing_and_piston",
                "actuator_b_housing_and_piston", "main_accumulator_housing_and_piston",
                "blowoff_housing_and_piston", "low_pressure_reservoir_housing",
                "cooler_region", "pipe_surroundings"):
        try:
            temperature = float(temperatures[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"initial_temperatures_c.{key} must be numeric") from exc
        if not np.isfinite(temperature) or temperature <= -273.15:
            raise ValueError(f"initial_temperatures_c.{key} must be finite and above absolute zero")
    if run.get("pipe_initial_temperature_profile") != "linear_between_connected_chambers":
        raise ValueError("unsupported pipe initial temperature profile")
    if any(run.get("sensor_locations", {}).get(key) != "low_pressure_reservoir" for key in
           ("low_pressure_gas_pressure", "low_pressure_gas_temperature")):
        raise ValueError("low-pressure gas sensors must be located in the reservoir")
    tail = float(physical["restrictions"]["high_flow_tail_fraction"])
    if not np.isfinite(tail) or not 0 < tail < 1:
        raise ValueError("restrictions.high_flow_tail_fraction must be between zero and one")
    for name in ("actuator", "accumulator_inlet"):
        for direction in ("compression", "extension"):
            p = physical["restrictions"][name][direction]
            curve = DigressiveCurve.from_points(area, p["speed_m_s"][0], p["force_n"][0], p["speed_m_s"][1], p["force_n"][1])
            if p["force_n"][1] / area > tail * curve.limit_pa:
                raise ValueError(f"{name}.{direction} high-flow tail must begin above both supplied points")
    for pressure in run["initial_gas_gauge_pressure_bar"].values():
        if not np.isfinite(float(pressure)) or pressure <= 0:
            raise ValueError("initial gas gauge pressures must be positive")
    _hot_ambient_settings(run)
    travel = run.get("blowoff_max_travel_m", 0.05)
    if not np.isfinite(float(travel)) or float(travel) <= 0:
        raise ValueError("blowoff_max_travel_m must be positive")
    initial_travel = run.get("blowoff_initial_travel_m", 0.0)
    if not np.isfinite(float(initial_travel)) or not 0 <= float(initial_travel) <= float(travel):
        raise ValueError("blowoff_initial_travel_m must be between zero and blowoff_max_travel_m")
    if float(initial_travel) != 0:
        raise ValueError("static initialization requires blow-off at its zero stop")
    chamber_volumes(g, blowoff_m=float(travel))
