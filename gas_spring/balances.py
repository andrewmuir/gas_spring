"""Mass and energy balance primitives for connected finite volumes."""
import math
import numpy as np


def advective_flux_rates(specific_enthalpies_j_kg, links):
    """Return mass and energy rates for internal links with signed kg/s flow.

    Each link is (node_a, node_b, mass_flow_a_to_b). Upwind specific enthalpy
    is transported, including on flow reversal. Both totals sum to zero for a
    closed set of nodes. External sources are deliberately not included.
    """
    h = np.asarray(specific_enthalpies_j_kg, dtype=float)
    if h.ndim != 1 or not np.isfinite(h).all():
        raise ValueError("specific enthalpies must be a finite 1D array")
    dm = np.zeros_like(h)
    du = np.zeros_like(h)
    for a, b, flow in links:
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or not 0 <= a < len(h) or not 0 <= b < len(h) or a == b or not np.isfinite(flow):
            raise ValueError("invalid mass-flow link")
        enthalpy_flow = flow * (h[a] if flow >= 0 else h[b])
        dm[a] -= flow
        dm[b] += flow
        du[a] -= enthalpy_flow
        du[b] += enthalpy_flow
    return dm, du


def ideal_gas_state(mass_kg, internal_energy_j, volume_m3, gas):
    """Return absolute pressure and Celsius temperature from mass and U.

    Constant heat capacities and an ideal-gas equation are approximations.
    Internal energy is measured relative to zero kelvin.
    """
    r = float(gas["specific_gas_constant_j_kg_k"])
    cv = float(gas["specific_heat_cp_j_kg_k"]) - r
    if all(np.isscalar(x) for x in (mass_kg, internal_energy_j, volume_m3)):
        m, u, v = float(mass_kg), float(internal_energy_j), float(volume_m3)
        if not all(math.isfinite(x) and x > 0 for x in (m, u, v, r, cv)):
            raise ValueError("invalid ideal gas state or heat capacities")
        temperature_k = u/(m*cv)
        return m*r*temperature_k/v, temperature_k-273.15
    m, u, v = (np.asarray(x, dtype=float) for x in (mass_kg, internal_energy_j, volume_m3))
    if np.any(~np.isfinite(m)) or np.any(m <= 0) or np.any(~np.isfinite(u)) or np.any(u <= 0) or np.any(~np.isfinite(v)) or np.any(v <= 0) or not np.isfinite([r, cv]).all() or r <= 0 or cv <= 0:
        raise ValueError("invalid ideal gas state or heat capacities")
    temperature_k = u / (m * cv)
    return m * r * temperature_k / v, temperature_k - 273.15


def ideal_gas_energy_rate(pressure_pa, volume_rate_m3_s, heat_rate_w,
                          advected_enthalpy_rate_w):
    """First-law gas energy rate dU/dt = Qdot - p*dV/dt + enthalpy flux."""
    values = np.asarray([pressure_pa, volume_rate_m3_s, heat_rate_w, advected_enthalpy_rate_w], dtype=float)
    if not np.isfinite(values).all() or pressure_pa <= 0:
        raise ValueError("invalid gas energy balance inputs")
    return heat_rate_w - pressure_pa * volume_rate_m3_s + advected_enthalpy_rate_w
