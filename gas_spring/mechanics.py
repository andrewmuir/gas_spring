"""Geometry, seal friction and static initialization building blocks.

Actuator x > 0 means extension. Main accumulator x > 0 expands its gas.
Blow-off x > 0 compresses its gas, measured from its zero stop.
"""
from dataclasses import dataclass
import math

from .hydraulics import piston_area


def chamber_volumes(geometry, actuator_a_m=0.0, actuator_b_m=0.0,
                    main_m=0.0, blowoff_m=0.0):
    """Volumes in m3 using current piston positions and zero-reference sizes.

    This function does not enforce mechanical stops: it rejects only volumes
    that cease to be positive. Stop handling belongs to the motion solver.
    """
    g = geometry
    a_full = piston_area(g["actuator_piston_diameter_m"])
    a_fluid = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    main_area = piston_area(g["main_accumulator_piston_diameter_m"])
    blow_gas_area = piston_area(g["blowoff_piston_diameter_m"])
    blow_fluid_area = piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"])
    result = {}
    for label, x in (("actuator_a", actuator_a_m), ("actuator_b", actuator_b_m)):
        result[label + "_fluid"] = g["actuator_fluid_dead_volume_m3"] + a_fluid * (g["actuator_fluid_chamber_length_at_zero_m"] - x)
        result[label + "_gas"] = g["actuator_gas_dead_volume_m3"] + a_full * (g["actuator_gas_chamber_length_at_zero_m"] + x)
    result["main_fluid"] = g["main_accumulator_fluid_volume_at_zero_m3"] - main_area * main_m
    result["main_gas"] = g["main_gas_volume_at_zero_m3"] + main_area * main_m
    result["blowoff_fluid"] = g["blowoff_fluid_volume_at_zero_m3"] + blow_fluid_area * blowoff_m
    result["blowoff_gas"] = g["blowoff_gas_volume_at_zero_stop_m3"] - blow_gas_area * blowoff_m
    result["low_pressure_reservoir_gas"] = g["low_pressure_reservoir_m3"]
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("piston position gives non-positive or non-finite chamber volume")
    return result


def seal_friction_limit(fluid_gauge_pa, gas_gauge_pa, spec):
    """Static/Coulomb friction from a provisional two-side pressure law."""
    values = (fluid_gauge_pa, gas_gauge_pa, spec["constant_n"], spec["pressure_coefficient_n_pa"])
    if not all(math.isfinite(float(v)) for v in values) or spec["constant_n"] < 0 or spec["pressure_coefficient_n_pa"] < 0:
        raise ValueError("invalid friction parameters or pressure")
    return spec["constant_n"] + spec["pressure_coefficient_n_pa"] * (max(0.0, fluid_gauge_pa) + max(0.0, gas_gauge_pa))


def piston_acceleration(position_m, velocity_m_s, drive_force_n, friction_limit_n,
                        mass_kg, lower_stop_m=None, upper_stop_m=None, *, kinetic_friction_limit_n=None):
    """Coulomb friction and unilateral stop acceleration away from impacts.

    Velocity is in the positive position direction. At rest, the piston sticks
    until |drive| exceeds the friction limit. At a stop, an outward drive is
    held by the stop. Impacts and zero crossings must be detected by the time
    integrator, which must update velocity at the event; this function does not
    project a moving state through a stop or infer an impact restitution.
    """
    if kinetic_friction_limit_n is None:
        kinetic_friction_limit_n = friction_limit_n
    if not all(math.isfinite(v) for v in (position_m, velocity_m_s, drive_force_n, friction_limit_n, kinetic_friction_limit_n, mass_kg)) or mass_kg <= 0 or friction_limit_n < 0 or not 0 <= kinetic_friction_limit_n <= friction_limit_n:
        raise ValueError("invalid piston state or mechanical parameters")
    if lower_stop_m is not None and (not math.isfinite(lower_stop_m) or position_m < lower_stop_m):
        raise ValueError("position is below lower stop")
    if upper_stop_m is not None and (not math.isfinite(upper_stop_m) or position_m > upper_stop_m):
        raise ValueError("position is above upper stop")
    if lower_stop_m is not None and upper_stop_m is not None and lower_stop_m >= upper_stop_m:
        raise ValueError("invalid stop interval")
    if (lower_stop_m is not None and position_m == lower_stop_m and velocity_m_s < 0) or (upper_stop_m is not None and position_m == upper_stop_m and velocity_m_s > 0):
        raise ValueError("outward impact velocity must be resolved at the stop")
    if velocity_m_s == 0:
        if abs(drive_force_n) <= friction_limit_n:
            return 0.0
        net = drive_force_n - math.copysign(kinetic_friction_limit_n, drive_force_n)
        if (lower_stop_m is not None and position_m == lower_stop_m and net < 0) or (upper_stop_m is not None and position_m == upper_stop_m and net > 0):
            return 0.0
    else:
        net = drive_force_n - math.copysign(kinetic_friction_limit_n, velocity_m_s)
    return net / mass_kg


@dataclass(frozen=True)
class InitialEquilibrium:
    hydraulic_gauge_pressure_pa: float
    allowable_pressure_interval_pa: tuple
    residual_forces_n: dict
    friction_limits_n: dict


def initial_static_equilibrium(physical, run, first_force_a_n, first_force_b_n):
    """Find a common zero-flow hydraulic pressure compatible with static seals.

    Actuators and main accumulator start at their zero reference, the blow-off
    at its lower stop. At the stop a negative residual is supported by the stop;
    a positive residual must remain within seal friction. Pressures here are
    gauge, and the search is restricted to non-negative hydraulic gauge pressure.
    This is a feasibility calculation, not a dynamic initialization solver.
    """
    if float(run.get("blowoff_initial_travel_m", 0.0)) != 0:
        raise ValueError("static initializer currently requires blow-off at its zero stop")
    if not all(math.isfinite(float(v)) for v in (first_force_a_n, first_force_b_n)):
        raise ValueError("initial forces must be finite")
    g, seals = physical["geometry"], physical["seal_friction"]
    pg = run["initial_gas_gauge_pressure_bar"]
    low, main, blow = (float(pg[k]) * 1e5 for k in ("low", "main", "blowoff"))
    if min(low, main, blow) < 0 or not all(math.isfinite(x) for x in (low, main, blow)):
        raise ValueError("initial gas gauge pressures must be finite and non-negative")
    aa = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    ag = piston_area(g["actuator_piston_diameter_m"])
    ma = piston_area(g["main_accumulator_piston_diameter_m"])
    ba = piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"])
    bg = piston_area(g["blowoff_piston_diameter_m"])
    # Each residual is a*p_h+b, with positive force in the position direction.
    cases = {
        # Extension shrinks the fluid chamber and expands the gas chamber.
        # Fluid pressure therefore pushes toward compression; gas pressure
        # pushes toward extension. Input force is positive in compression.
        "actuator_a": (-aa, ag * low - float(first_force_a_n), low, seals["actuator"], False),
        "actuator_b": (-aa, ag * low - float(first_force_b_n), low, seals["actuator"], False),
        "main_accumulator": (-ma, ma * main, main, seals["main_accumulator"], False),
        "blowoff": (ba, -bg * blow, blow, seals["blowoff"], True),
    }
    lower, upper = 0.0, math.inf

    def constrain(coefficient, rhs):
        nonlocal lower, upper
        if abs(coefficient) < 1e-18:
            if rhs < 0:
                raise ValueError("initial forces and pressures cannot be held in static equilibrium")
        elif coefficient > 0:
            upper = min(upper, rhs / coefficient)
        else:
            lower = max(lower, rhs / coefficient)

    for a, b, gas_pressure, spec, lower_stop in cases.values():
        seal_friction_limit(0.0, gas_pressure, spec)  # validate parameters
        c, k = float(spec["constant_n"]), float(spec["pressure_coefficient_n_pa"])
        # a*p+b <= c+k*(p+gas), and the opposite inequality except at a lower stop.
        constrain(a-k, c+k*gas_pressure-b)
        if not lower_stop:
            constrain(-a-k, c+k*gas_pressure+b)
    if upper < lower or upper < 0:
        raise ValueError("initial forces and pressures cannot be held in static equilibrium")
    chosen = min(max(main, lower), upper)
    residuals = {name: a*chosen+b for name, (a, b, _, _, _) in cases.items()}
    limits = {name: seal_friction_limit(chosen, gas_p, spec)
              for name, (_, _, gas_p, spec, _) in cases.items()}
    return InitialEquilibrium(chosen, (lower, upper), residuals, limits)
