"""Provisional material equations in SI units; temperatures are degrees C."""

import math
import numpy as np

ATMOSPHERIC_PRESSURE_PA = 101325.0
KELVIN_OFFSET = 273.15


def liquid_density(temperature_c, properties):
    """Pressure-independent density with linear thermal specific-volume expansion."""
    t = np.asarray(temperature_c, dtype=float)
    rho = float(properties["density_kg_m3_at_20c"])
    alpha = float(properties["volumetric_expansion_per_k"])
    factor = 1 + alpha * (t - 20)
    if (
        not np.isfinite([rho, alpha]).all()
        or rho <= 0
        or alpha < 0
        or np.any(~np.isfinite(t))
        or np.any(t <= -273.15)
        or np.any(factor <= 0)
    ):
        raise ValueError("invalid incompressible liquid properties or temperature")
    return rho / factor


def liquid_internal_energy(temperature_c, properties):
    """Sensible specific energy, J/kg relative to 20 C, using constant heat capacity."""
    cp = float(properties["specific_heat_j_kg_k"])
    t = np.asarray(temperature_c, dtype=float)
    if (
        not math.isfinite(cp)
        or cp <= 0
        or np.any(~np.isfinite(t))
        or np.any(t <= -273.15)
    ):
        raise ValueError("invalid liquid heat capacity or temperature")
    return cp * (t - 20)


def liquid_temperatures(
    mass_kg, energy_j, properties, boundary_energy_j=None, fraction=None
):
    """Mean/core/boundary temperatures; pressure is determined by the circuit."""
    cp = float(properties["specific_heat_j_kg_k"])
    if mass_kg <= 0 or not math.isfinite(mass_kg) or cp <= 0:
        raise ValueError("invalid liquid mass or heat capacity")
    mean = 20 + energy_j / (mass_kg * cp)
    if boundary_energy_j is None:
        core = boundary = mean
    else:
        if not 0 < fraction < 1:
            raise ValueError("invalid liquid boundary fraction")
        core = 20 + (energy_j - boundary_energy_j) / ((1 - fraction) * mass_kg * cp)
        boundary = 20 + boundary_energy_j / (fraction * mass_kg * cp)
    if not all(math.isfinite(t) and t > -273.15 for t in (mean, core, boundary)):
        raise ValueError("invalid liquid temperature")
    return mean, core, boundary


def liquid_viscosity(temperature_c, properties):
    """Log-linear interpolation between the 20 C guess and 100 C reference.

    Temperatures outside the tabulated range are clamped; extrapolation is not
    justified by the two values. The 20 C value is provisional.
    """
    mu20 = float(properties["dynamic_viscosity_pa_s"])
    mu100 = float(properties["dynamic_viscosity_pa_s_at_100c"])
    if np.isscalar(temperature_c):
        t = float(temperature_c)
        if (
            not all(math.isfinite(x) for x in (t, mu20, mu100))
            or t <= -KELVIN_OFFSET
            or min(mu20, mu100) <= 0
        ):
            raise ValueError("invalid temperature or viscosity")
        f = min(1.0, max(0.0, (t - 20.0) / 80.0))
        return math.exp((1 - f) * math.log(mu20) + f * math.log(mu100))
    t = np.asarray(temperature_c, dtype=float)
    if (
        np.any(~np.isfinite(t))
        or np.any(t <= -KELVIN_OFFSET)
        or min(mu20, mu100) <= 0
        or not np.isfinite([mu20, mu100]).all()
    ):
        raise ValueError("invalid temperature or viscosity")
    f = np.clip((t - 20.0) / 80.0, 0.0, 1.0)
    return np.exp((1.0 - f) * np.log(mu20) + f * np.log(mu100))


def nitrogen_mass(pressure_pa, volume_m3, temperature_c, properties):
    """Ideal-gas mass; pressure is absolute. Real-gas effects are omitted."""
    p, v, t = (
        np.asarray(x, dtype=float) for x in (pressure_pa, volume_m3, temperature_c)
    )
    r = float(properties["specific_gas_constant_j_kg_k"])
    if (
        np.any(~np.isfinite(p))
        or np.any(p <= 0)
        or np.any(~np.isfinite(v))
        or np.any(v <= 0)
        or np.any(~np.isfinite(t))
        or np.any(t <= -KELVIN_OFFSET)
        or not np.isfinite(r)
        or r <= 0
    ):
        raise ValueError("invalid nitrogen state")
    return p * v / (r * (t + KELVIN_OFFSET))


def nitrogen_pressure(mass_kg, volume_m3, temperature_c, properties):
    """Ideal-gas absolute pressure from mass, volume and temperature."""
    m, v, t = (np.asarray(x, dtype=float) for x in (mass_kg, volume_m3, temperature_c))
    r = float(properties["specific_gas_constant_j_kg_k"])
    if (
        np.any(~np.isfinite(m))
        or np.any(m <= 0)
        or np.any(~np.isfinite(v))
        or np.any(v <= 0)
        or np.any(~np.isfinite(t))
        or np.any(t <= -KELVIN_OFFSET)
        or not np.isfinite(r)
        or r <= 0
    ):
        raise ValueError("invalid nitrogen state")
    return m * r * (t + KELVIN_OFFSET) / v
