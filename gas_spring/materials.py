"""Provisional material equations in SI units; temperatures are degrees C."""
import math
import numpy as np

ATMOSPHERIC_PRESSURE_PA = 101325.0
KELVIN_OFFSET = 273.15


def liquid_density(pressure_pa, temperature_c, properties):
    """Compressible liquid EOS with constant bulk modulus and expansion.

    Pressure is absolute. rho = rho_ref exp((p-p_ref)/K - alpha*(T-T_ref)).
    The model neglects the temperature and pressure dependence of K and alpha.
    """
    p = np.asarray(pressure_pa, dtype=float)
    t = np.asarray(temperature_c, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p <= 0) or np.any(~np.isfinite(t)) or np.any(t <= -KELVIN_OFFSET):
        raise ValueError("pressure must be positive and temperature above absolute zero")
    rho0 = float(properties["density_kg_m3_at_20c"])
    bulk = float(properties["bulk_modulus_pa"])
    alpha = float(properties["volumetric_expansion_per_k"])
    if not np.isfinite([rho0, bulk, alpha]).all() or rho0 <= 0 or bulk <= 0 or alpha < 0:
        raise ValueError("invalid liquid material properties")
    return rho0 * np.exp((p - ATMOSPHERIC_PRESSURE_PA) / bulk - alpha * (t - 20.0))


def liquid_pressure(density_kg_m3, temperature_c, properties):
    """Invert liquid_density; report an invalid non-positive absolute pressure."""
    rho = np.asarray(density_kg_m3, dtype=float)
    t = np.asarray(temperature_c, dtype=float)
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0) or np.any(~np.isfinite(t)) or np.any(t <= -KELVIN_OFFSET):
        raise ValueError("density must be positive and temperature above absolute zero")
    rho0 = float(properties["density_kg_m3_at_20c"])
    bulk = float(properties["bulk_modulus_pa"])
    alpha = float(properties["volumetric_expansion_per_k"])
    if not np.isfinite([rho0, bulk, alpha]).all() or rho0 <= 0 or bulk <= 0 or alpha < 0:
        raise ValueError("invalid liquid material properties")
    pressure = ATMOSPHERIC_PRESSURE_PA + bulk * (np.log(rho / rho0) + alpha * (t - 20.0))
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("liquid equation of state gives non-positive absolute pressure")
    return pressure


def liquid_viscosity(temperature_c, properties):
    """Log-linear interpolation between the 20 C guess and 100 C reference.

    Temperatures outside the tabulated range are clamped; extrapolation is not
    justified by the two values. The 20 C value is provisional.
    """
    mu20 = float(properties["dynamic_viscosity_pa_s"])
    mu100 = float(properties["dynamic_viscosity_pa_s_at_100c"])
    if np.isscalar(temperature_c):
        t = float(temperature_c)
        if not all(math.isfinite(x) for x in (t, mu20, mu100)) or t <= -KELVIN_OFFSET or min(mu20, mu100) <= 0:
            raise ValueError("invalid temperature or viscosity")
        f = min(1.0, max(0.0, (t-20.0)/80.0))
        return math.exp((1-f)*math.log(mu20)+f*math.log(mu100))
    t = np.asarray(temperature_c, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t <= -KELVIN_OFFSET) or min(mu20, mu100) <= 0 or not np.isfinite([mu20, mu100]).all():
        raise ValueError("invalid temperature or viscosity")
    f = np.clip((t - 20.0) / 80.0, 0.0, 1.0)
    return np.exp((1.0 - f) * np.log(mu20) + f * np.log(mu100))


def liquid_cv(properties):
    """Constant cv chosen to match configured cp at the 20 C reference state.

    cp-cv = T*alpha**2*K/rho at the reference. Constant cv and the exponential
    EOS form a consistent provisional caloric model; cp elsewhere is implied
    by that model and is not constant.
    """
    rho0 = float(properties["density_kg_m3_at_20c"])
    bulk = float(properties["bulk_modulus_pa"])
    alpha = float(properties["volumetric_expansion_per_k"])
    cp = float(properties["specific_heat_j_kg_k"])
    if not all(math.isfinite(x) for x in (rho0, bulk, alpha, cp)) or rho0 <= 0 or bulk <= 0 or alpha < 0 or cp <= 0:
        raise ValueError("invalid liquid thermal properties")
    cv = cp - (20.0 + KELVIN_OFFSET) * alpha**2 * bulk / rho0
    if cv <= 0:
        raise ValueError("liquid reference cv must be positive")
    return cv


def liquid_internal_energy(temperature_c, density_kg_m3, properties):
    """Specific internal energy J/kg relative to liquid at 20 C and 1 atm.

    This caloric relation is consistent with liquid_density's constant K and
    alpha and constant cv: de = cv*dT + (T*alpha*K-p)*d(1/rho).
    It is an approximation, not measured DOT 4 property data.
    """
    t, rho = (np.asarray(x, dtype=float) for x in (temperature_c, density_kg_m3))
    if np.any(~np.isfinite(t)) or np.any(t <= -KELVIN_OFFSET) or np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("invalid liquid temperature or density")
    cv = liquid_cv(properties)
    rho0 = float(properties["density_kg_m3_at_20c"])
    bulk = float(properties["bulk_modulus_pa"])
    alpha = float(properties["volumetric_expansion_per_k"])
    v0 = 1.0 / rho0
    log_ratio = np.log(rho0 / rho)
    dv = v0 * np.expm1(log_ratio)
    curvature = v0 * (log_ratio + np.expm1(log_ratio) * (log_ratio - 1.0))
    return cv * (t - 20.0) + (alpha * bulk * (20.0 + KELVIN_OFFSET) - ATMOSPHERIC_PRESSURE_PA) * dv + bulk * curvature


def liquid_state(mass_kg, internal_energy_j, volume_m3, properties):
    """Return absolute pressure and Celsius temperature from mass, U and V."""
    if all(np.isscalar(x) for x in (mass_kg, internal_energy_j, volume_m3)):
        mass, u, volume = float(mass_kg), float(internal_energy_j), float(volume_m3)
        if not all(math.isfinite(x) for x in (mass, u, volume)) or mass <= 0 or volume <= 0:
            raise ValueError("invalid liquid mass, energy or volume")
        rho0 = float(properties["density_kg_m3_at_20c"])
        bulk = float(properties["bulk_modulus_pa"])
        alpha = float(properties["volumetric_expansion_per_k"])
        cv = liquid_cv(properties)
        log_ratio = math.log(rho0*volume/mass)
        exp_delta = math.expm1(log_ratio)
        dv = exp_delta / rho0
        curvature = (log_ratio + exp_delta*(log_ratio-1.0)) / rho0
        reference_e = (alpha*bulk*(20.0+KELVIN_OFFSET)-ATMOSPHERIC_PRESSURE_PA)*dv + bulk*curvature
        temperature = 20.0+(u/mass-reference_e)/cv
        if not math.isfinite(temperature) or temperature <= -KELVIN_OFFSET:
            raise ValueError("liquid temperature must be above absolute zero")
        pressure = ATMOSPHERIC_PRESSURE_PA + bulk*(-log_ratio + alpha*(temperature-20.0))
        if not math.isfinite(pressure) or pressure <= 0:
            raise ValueError("liquid equation of state gives non-positive absolute pressure")
        return pressure, temperature
    mass, u, volume = (np.asarray(x, dtype=float) for x in (mass_kg, internal_energy_j, volume_m3))
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0) or np.any(~np.isfinite(u)) or np.any(~np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("invalid liquid mass, energy or volume")
    rho = mass / volume
    reference_e = liquid_internal_energy(20.0, rho, properties)
    temperature = 20.0 + (u / mass - reference_e) / liquid_cv(properties)
    return liquid_pressure(rho, temperature, properties), temperature


def liquid_zone_temperatures(mass_kg, internal_energy_j, volume_m3,
                             boundary_energy_j, boundary_mass_fraction, properties):
    """Common pressure, core and boundary temperatures for a two-zone liquid.

    The common pressure uses the aggregate mass, energy and volume. The zones
    share aggregate density when evaluating temperature and enthalpy; local
    thermal-expansion differences are therefore approximated. This allows a
    wall-to-core temperature lag without inventing a liquid pressure jump.
    """
    mass, total, volume = float(mass_kg), float(internal_energy_j), float(volume_m3)
    boundary, f = float(boundary_energy_j), float(boundary_mass_fraction)
    if not all(math.isfinite(x) for x in (mass, total, volume, boundary, f)) or mass <= 0 or volume <= 0 or not 0 < f < 1:
        raise ValueError("invalid two-zone liquid state")
    pressure, bulk_temperature = liquid_state(mass, total, volume, properties)
    cv = liquid_cv(properties)
    reference_e = total/mass - cv*(bulk_temperature-20.0)
    boundary_t = 20.0 + (boundary/(f*mass)-reference_e)/cv
    core_t = 20.0 + ((total-boundary)/((1-f)*mass)-reference_e)/cv
    if not all(math.isfinite(x) and x > -KELVIN_OFFSET for x in (boundary_t, core_t)):
        raise ValueError("invalid two-zone liquid temperatures")
    return pressure, core_t, boundary_t


def nitrogen_mass(pressure_pa, volume_m3, temperature_c, properties):
    """Ideal-gas mass; pressure is absolute. Real-gas effects are omitted."""
    p, v, t = (np.asarray(x, dtype=float) for x in (pressure_pa, volume_m3, temperature_c))
    r = float(properties["specific_gas_constant_j_kg_k"])
    if np.any(~np.isfinite(p)) or np.any(p <= 0) or np.any(~np.isfinite(v)) or np.any(v <= 0) or np.any(~np.isfinite(t)) or np.any(t <= -KELVIN_OFFSET) or not np.isfinite(r) or r <= 0:
        raise ValueError("invalid nitrogen state")
    return p * v / (r * (t + KELVIN_OFFSET))


def nitrogen_pressure(mass_kg, volume_m3, temperature_c, properties):
    """Ideal-gas absolute pressure from mass, volume and temperature."""
    m, v, t = (np.asarray(x, dtype=float) for x in (mass_kg, volume_m3, temperature_c))
    r = float(properties["specific_gas_constant_j_kg_k"])
    if np.any(~np.isfinite(m)) or np.any(m <= 0) or np.any(~np.isfinite(v)) or np.any(v <= 0) or np.any(~np.isfinite(t)) or np.any(t <= -KELVIN_OFFSET) or not np.isfinite(r) or r <= 0:
        raise ValueError("invalid nitrogen state")
    return m * r * (t + KELVIN_OFFSET) / v
