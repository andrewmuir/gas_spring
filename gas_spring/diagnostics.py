"""Small, explicitly approximate thermal calibration diagnostics."""
import numpy as np
from scipy.integrate import solve_ivp

from .hydraulics import piston_area
from .materials import ATMOSPHERIC_PRESSURE_PA, liquid_density, nitrogen_mass


def isolated_actuator_warmup(physical, run, actuator="actuator_a", duration_s=180.0):
    """Estimate static actuator temperatures with a six-node thermal model.

    The housing is one lump, and fluid and gas masses, wetted areas and heat
    capacities are held fixed. Pipes, cooler, pressure work, piston motion and
    flow are omitted. Fluid cp is used as an effective thermal capacity, so
    this is a calibration guide rather than the full coupled model's result.
    """
    if actuator not in ("actuator_a", "actuator_b") or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("invalid actuator or duration")
    if run["hot_ambient"]["source"] != "constant":
        raise ValueError("isolated warmup requires a constant hot ambient")
    g, th, fluid, gas, t = (physical["geometry"], physical["thermal"], physical["fluid"],
                             physical["gas"], run["initial_temperatures_c"])
    af = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    ag = piston_area(g["actuator_piston_diameter_m"])
    circumference = np.pi*g["actuator_piston_diameter_m"]
    fluid_wall = circumference*g["actuator_fluid_chamber_length_at_zero_m"]
    gas_wall = circumference*g["actuator_gas_chamber_length_at_zero_m"]
    fluid_volume = af*g["actuator_fluid_chamber_length_at_zero_m"] + g["actuator_fluid_dead_volume_m3"]
    gas_volume = ag*g["actuator_gas_chamber_length_at_zero_m"] + g["actuator_gas_dead_volume_m3"]
    liquid_p = ATMOSPHERIC_PRESSURE_PA + run["initial_gas_gauge_pressure_bar"]["main"]*1e5
    gas_p = ATMOSPHERIC_PRESSURE_PA + run["initial_gas_gauge_pressure_bar"]["low"]*1e5
    fluid_mass = fluid_volume*float(liquid_density(t[actuator+"_fluid"], fluid))
    gas_mass = float(nitrogen_mass(gas_p, gas_volume, t[actuator+"_gas"], gas))
    fraction = th["gas_boundary_mass_fraction"]
    cv = gas["specific_heat_cp_j_kg_k"]-gas["specific_gas_constant_j_kg_k"]
    piston_capacity = physical["moving_masses_kg"][actuator]*th["metal_specific_heat_j_kg_k"]/2
    capacity = np.array([fluid_mass*fluid["specific_heat_j_kg_k"], gas_mass*(1-fraction)*cv,
                         gas_mass*fraction*cv, th["actuator_housing_mass_kg_each"]*th["metal_specific_heat_j_kg_k"],
                         piston_capacity, piston_capacity])
    if not np.isfinite(capacity).all() or np.any(capacity <= 0):
        raise ValueError("invalid thermal capacities")
    # Ordering: fluid, gas core, gas boundary, housing, fluid face, gas face.
    links = ((0, 3, th["fluid_to_housing_w_m2_k"]*fluid_wall),
             (2, 3, th["gas_to_housing_w_m2_k"]*gas_wall),
             (1, 2, th["gas_core_to_boundary_w_m2_k"]*(gas_wall+ag)),
             (0, 4, th["fluid_to_housing_w_m2_k"]*af),
             (2, 5, th["gas_to_housing_w_m2_k"]*ag),
             (4, 5, th["actuator_piston_through_conductance_w_k"]))
    ambient = float(run["hot_ambient"]["temperature_c"])
    conductance = th["actuator_hot_ambient_to_housing_w_m2_k"]*th["actuator_outer_heat_transfer_area_m2_each"]
    start = np.array([t[actuator+"_fluid"], t[actuator+"_gas"], t[actuator+"_gas"],
                      t[actuator+"_housing_and_piston"], t[actuator+"_housing_and_piston"],
                      t[actuator+"_housing_and_piston"]], dtype=float)
    def rhs(_, temperature):
        heat = np.zeros(6)
        for a, b, conduct in links:
            power = conduct*(temperature[a]-temperature[b])
            heat[a] -= power
            heat[b] += power
        heat[3] += conductance*(ambient-temperature[3])
        return heat/capacity
    result = solve_ivp(rhs, (0, duration_s), start, method="BDF", rtol=1e-8, atol=1e-9)
    if not result.success:
        raise RuntimeError(result.message)
    labels = ("fluid_c", "gas_core_c", "gas_boundary_c", "housing_c", "piston_fluid_face_c", "piston_gas_face_c")
    return dict(zip(labels, map(float, result.y[:, -1])))
