"""Construct zero-flow reference states for the incompressible solver."""
import math
import numpy as np

from .force_input import transfer_force
from .materials import ATMOSPHERIC_PRESSURE_PA, liquid_density, liquid_internal_energy, nitrogen_mass
from .mechanics import chamber_volumes, initial_static_equilibrium
from .thermal import initial_pipe_temperature_profiles


def initialize_system(physical, run, first_force_a_n, first_force_b_n,
                      time_s=0.0, pipe_cells=8, actuator_wall_cells=8):
    """Build initial masses, energies, temperatures and positions.

    This is a zero-flow static initialization. It rejects forces that cannot
    be held with the configured pressure and friction assumptions. Pipe cells
    are finite volumes, with temperature specified at cell centres. No time
    integration is performed here.
    """
    if isinstance(actuator_wall_cells, bool) or not isinstance(actuator_wall_cells, int) or actuator_wall_cells <= 0:
        raise ValueError("actuator_wall_cells must be a positive integer")
    force_a = transfer_force(first_force_a_n, time_s)
    force_b = transfer_force(first_force_b_n, time_s)
    equilibrium = initial_static_equilibrium(physical, run, force_a, force_b)
    geom = physical["geometry"]
    t = run["initial_temperatures_c"]
    profiles = initial_pipe_temperature_profiles(run, pipe_cells)
    volumes = chamber_volumes(geom)
    liquid_p = ATMOSPHERIC_PRESSURE_PA + equilibrium.hydraulic_gauge_pressure_pa
    p_gas = {key: ATMOSPHERIC_PRESSURE_PA + run["initial_gas_gauge_pressure_bar"][kind] * 1e5
             for key, kind in (("actuator_a_gas", "low"), ("actuator_b_gas", "low"),
                               ("low_pressure_reservoir_gas", "low"), ("main_gas", "main"),
                               ("blowoff_gas", "blowoff"))}
    fluid_temp = {"actuator_a_fluid": t["actuator_a_fluid"],
                  "actuator_b_fluid": t["actuator_b_fluid"],
                  "main_fluid": t["main_accumulator_fluid"],
                  "blowoff_fluid": t["blowoff_fluid"]}
    gas_temp = {"actuator_a_gas": t["actuator_a_gas"], "actuator_b_gas": t["actuator_b_gas"],
                "main_gas": t["main_accumulator_gas"], "blowoff_gas": t["blowoff_gas"],
                "low_pressure_reservoir_gas": t["low_pressure_reservoir_gas"]}
    cv = physical["gas"]["specific_heat_cp_j_kg_k"] - physical["gas"]["specific_gas_constant_j_kg_k"]
    if not math.isfinite(cv) or cv <= 0:
        raise ValueError("nitrogen heat capacity at constant volume must be positive")
    liquid = {key: {"volume_m3": volumes[key], "temperature_c": temp,
                    "mass_kg": float(volumes[key] * liquid_density(temp, physical["fluid"]))}
              for key, temp in fluid_temp.items()}
    for key, temp in (("hydraulic_inlet_junction", np.mean([t["actuator_a_fluid"], t["actuator_b_fluid"]])),
                      ("hydraulic_outlet_junction", np.mean([t["main_accumulator_fluid"], t["blowoff_fluid"]]))):
        volume = geom["hydraulic_junction_volume_m3"]
        liquid[key] = {"volume_m3": volume, "temperature_c": float(temp),
                       "mass_kg": float(volume * liquid_density(temp, physical["fluid"]))}
    for chamber in liquid.values():
        chamber["internal_energy_j"] = float(chamber["mass_kg"] * liquid_internal_energy(
            chamber["temperature_c"], physical["fluid"]))
    liquid_fraction = physical["thermal"]["liquid_boundary_mass_fraction"]
    if not 0 < liquid_fraction < 1:
        raise ValueError("liquid boundary mass fraction must be between zero and one")
    for key in ("actuator_a_fluid", "actuator_b_fluid", "main_fluid", "blowoff_fluid"):
        liquid[key]["boundary_mass_fraction"] = liquid_fraction
        liquid[key]["boundary_energy_j"] = liquid_fraction * liquid[key]["internal_energy_j"]
    gas = {}
    boundary_fraction = physical["thermal"]["gas_boundary_mass_fraction"]
    if not 0 < boundary_fraction < 1:
        raise ValueError("gas boundary mass fraction must be between zero and one")
    for key, temp in gas_temp.items():
        mass = float(nitrogen_mass(p_gas[key], volumes[key], temp, physical["gas"]))
        gas[key] = {"volume_m3": volumes[key], "temperature_c": temp,
                    "mass_kg": mass, "internal_energy_j": mass * cv * (temp + 273.15),
                    "boundary_mass_fraction": boundary_fraction,
                    "boundary_energy_j": boundary_fraction * mass * cv * (temp + 273.15)}
    gas_junction_temp = float(np.mean([t["actuator_a_gas"], t["actuator_b_gas"]]))
    gas_junction_volume = geom["gas_junction_volume_m3"]
    gas_junction_mass = float(nitrogen_mass(p_gas["actuator_a_gas"], gas_junction_volume,
                                            gas_junction_temp, physical["gas"]))
    gas["actuator_gas_junction"] = {
        "volume_m3": gas_junction_volume, "temperature_c": gas_junction_temp,
        "mass_kg": gas_junction_mass,
        "internal_energy_j": gas_junction_mass * cv * (gas_junction_temp + 273.15),
    }
    pipe_result = {}
    for name, temp, diameter, length in (
        ("hydraulic", profiles["hydraulic_pipe_c"], geom["hydraulic_pipe_diameter_m"], geom["hydraulic_pipe_length_m"]),
        ("gas", profiles["gas_pipe_c"], geom["gas_pipe_diameter_m"], geom["gas_pipe_length_m"]),
    ):
        cell_volume = math.pi / 4 * diameter**2 * length / pipe_cells
        if name == "hydraulic":
            mass = cell_volume * liquid_density(temp, physical["fluid"])
            energy = mass * liquid_internal_energy(temp, physical["fluid"])
        else:
            mass = nitrogen_mass(p_gas["low_pressure_reservoir_gas"], cell_volume, temp, physical["gas"])
            energy = mass * cv * (temp + 273.15)
        pipe_result[name] = {"cell_volume_m3": cell_volume, "temperature_c": temp.copy(),
                             "wall_temperature_c": temp.copy(), "mass_kg": np.asarray(mass),
                             "internal_energy_j": energy,
                             "flow_m3_s": np.zeros(pipe_cells + 1)}
    return {
        "time_s": float(time_s), "equilibrium": equilibrium,
        "applied_force_n": {"actuator_a": force_a, "actuator_b": force_b},
        "position_m": {"actuator_a": 0.0, "actuator_b": 0.0, "main_accumulator": 0.0, "blowoff": 0.0},
        "velocity_m_s": {"actuator_a": 0.0, "actuator_b": 0.0, "main_accumulator": 0.0, "blowoff": 0.0},
        "liquid_pressure_pa": liquid_p, "liquid_chambers": liquid,
        "gas_chambers": gas, "pipes": pipe_result,
        "actuator_wall_temperature_c": {
            "actuator_a": np.full(actuator_wall_cells, t["actuator_a_housing_and_piston"], dtype=float),
            "actuator_b": np.full(actuator_wall_cells, t["actuator_b_housing_and_piston"], dtype=float),
        },
        "cooler_region_temperature_c": t["cooler_region"],
        "solid_temperatures_c": {
            "actuator_a_wall": np.full(actuator_wall_cells, t["actuator_a_housing_and_piston"], dtype=float),
            "actuator_b_wall": np.full(actuator_wall_cells, t["actuator_b_housing_and_piston"], dtype=float),
            "actuator_a_piston_fluid": t["actuator_a_housing_and_piston"],
            "actuator_a_piston_gas": t["actuator_a_housing_and_piston"],
            "actuator_b_piston_fluid": t["actuator_b_housing_and_piston"],
            "actuator_b_piston_gas": t["actuator_b_housing_and_piston"],
            "main_wall": t["main_accumulator_housing_and_piston"],
            "main_piston_fluid": t["main_accumulator_housing_and_piston"],
            "main_piston_gas": t["main_accumulator_housing_and_piston"],
            "blowoff_wall": t["blowoff_housing_and_piston"],
            "blowoff_piston_fluid": t["blowoff_housing_and_piston"],
            "blowoff_piston_gas": t["blowoff_housing_and_piston"],
            "reservoir_wall": t["low_pressure_reservoir_housing"],
            "hydraulic_pipe_wall": profiles["hydraulic_pipe_c"].copy(),
            "gas_pipe_wall": profiles["gas_pipe_c"].copy(),
            "cooler_region": t["cooler_region"],
        },
    }
