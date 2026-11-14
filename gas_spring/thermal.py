"""Conservative thermal building blocks for the coupled solver.

All temperatures in kelvin or degrees Celsius (differences only), heat rates in W.
"""
import numpy as np

def initial_pipe_temperature_profiles(run, cell_count):
    """Return initial cell-centre temperatures, from actuator to remote end.

    Junction temperatures are approximated by the arithmetic means of the
    connected chamber temperatures. The hydraulic far end joins the main and
    blow-off chambers; the gas far end is the low-pressure reservoir.
    """
    if isinstance(cell_count, bool) or not isinstance(cell_count, (int, np.integer)) or cell_count <= 0:
        raise ValueError("cell_count must be a positive integer")
    if run.get("pipe_initial_temperature_profile") != "linear_between_connected_chambers":
        raise ValueError("unsupported pipe initial temperature profile")
    t = run["initial_temperatures_c"]
    names = ("actuator_a_fluid", "actuator_b_fluid", "main_accumulator_fluid",
             "blowoff_fluid", "actuator_a_gas", "actuator_b_gas",
             "low_pressure_reservoir_gas")
    try:
        values = np.asarray([t[name] for name in names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pipe endpoint temperatures must be numeric and present") from exc
    if not np.all(np.isfinite(values)) or np.any(values <= -273.15):
        raise ValueError("pipe endpoint temperatures must be finite and above absolute zero")
    fluid_start, fluid_end = np.mean(values[:2]), np.mean(values[2:4])
    gas_start, gas_end = np.mean(values[4:6]), values[6]
    fraction = (np.arange(cell_count) + 0.5) / cell_count
    return {
        "hydraulic_pipe_c": fluid_start + fraction * (fluid_end - fluid_start),
        "gas_pipe_c": gas_start + fraction * (gas_end - gas_start),
    }

def pipe_heat_rates(temperatures, volumetric_flow_m3_s, density_kg_m3,
                    specific_heat_j_kg_k, conductivity_w_m_k, area_m2, cell_length_m,
                    inlet_temperature):
    """Cell heat rates from upwind enthalpy transport and axial conduction.

    inlet_temperature is the upstream boundary temperature for the current flow
    direction. Axial conduction has zero-gradient boundary conditions; connection
    to chambers and heat exchange with the wall are handled separately.
    """
    t = np.asarray(temperatures, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError("temperatures must be a nonempty finite vector")
    if min(density_kg_m3, specific_heat_j_kg_k, area_m2, cell_length_m) <= 0 or conductivity_w_m_k < 0:
        raise ValueError("invalid thermal or pipe properties")
    q = volumetric_flow_m3_s
    upstream = np.r_[inlet_temperature, t[:-1]] if q >= 0 else np.r_[t[1:], inlet_temperature]
    rates = density_kg_m3*specific_heat_j_kg_k*abs(q)*(upstream-t)
    conduction = conductivity_w_m_k*area_m2/cell_length_m*np.diff(t)
    rates[:-1] += conduction
    rates[1:] -= conduction
    return rates

def actuator_wetted_wall_areas(bore_diameter_m, chamber_length_m, piston_position_m):
    """Fluid wall is from x=0 to piston; gas wall is the remaining length."""
    if bore_diameter_m <= 0 or chamber_length_m <= 0 or not 0 <= piston_position_m <= chamber_length_m:
        raise ValueError("invalid geometry or piston position")
    circumference = np.pi*bore_diameter_m
    return circumference*piston_position_m, circumference*(chamber_length_m-piston_position_m)

def actuator_wall_cell_areas(bore_diameter_m, fluid_length_at_zero_m,
                             gas_length_at_zero_m, extension_m, cell_count):
    """Fluid- and gas-wetted bore area per fixed axial wall cell, in m2.

    Extension decreases fluid-side length. A cell crossed by the piston is
    shared in proportion to the length wetted by each medium.
    """
    if isinstance(cell_count, bool) or not isinstance(cell_count, (int, np.integer)) or cell_count <= 0:
        raise ValueError("cell_count must be a positive integer")
    total = fluid_length_at_zero_m + gas_length_at_zero_m
    boundary = fluid_length_at_zero_m - extension_m
    if not np.isfinite([bore_diameter_m, total, boundary]).all() or bore_diameter_m <= 0 or total <= 0 or not 0 <= boundary <= total:
        raise ValueError("invalid actuator geometry or piston position")
    edges = np.linspace(0, total, cell_count + 1)
    fluid_lengths = np.clip(boundary - edges[:-1], 0.0, np.diff(edges))
    circumference = np.pi * bore_diameter_m
    return circumference * fluid_lengths, circumference * (np.diff(edges) - fluid_lengths)


def exchange_heat_rates(temperatures, links):
    """Heat rates for pairwise (index_a, index_b, conductance W/K) links.

    Heat leaving one internal node enters the other, including piston and
    gas-to-wall links. External ambient sources must be accounted for separately.
    """
    t = np.asarray(temperatures, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError("temperatures must be a finite 1D array")
    rates = np.zeros_like(t)
    for a, b, conductance in links:
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or not 0 <= a < len(t) or not 0 <= b < len(t) or a == b or not np.isfinite(conductance) or conductance < 0:
            raise ValueError("invalid heat-exchange link")
        heat = conductance * (t[a] - t[b])
        rates[a] -= heat
        rates[b] += heat
    return rates

def interface_heat_rate(coefficient_w_m2_k, area_m2, temperature_a, temperature_b):
    """Positive means heat enters B from A."""
    if coefficient_w_m2_k < 0 or area_m2 < 0:
        raise ValueError("coefficient and area must be non-negative")
    return coefficient_w_m2_k*area_m2*(temperature_a-temperature_b)
