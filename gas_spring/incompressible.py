"""Volume and flow constraints for an incompressible hydraulic circuit.

Used by the coupled incompressible time integrator.
Positive actuator/main piston velocity is extension; positive blow-off velocity
opens its liquid chamber. Thermal source rates are increases in liquid volume.
"""

import math
import numpy as np

from .hydraulics import (
    DigressiveCurve,
    pipe_inertance,
    pipe_pressure_drop,
    piston_area,
    regularized_restriction_pressure_drop,
)
from .materials import ATMOSPHERIC_PRESSURE_PA, liquid_viscosity
from .mechanics import (
    effective_piston_mass_kg,
    seal_friction_limit,
    smooth_friction_force,
)


def liquid_piston_areas(geometry):
    """Return the actuator, main and blow-off liquid-side piston areas."""
    actuator = piston_area(
        geometry["actuator_piston_diameter_m"], geometry["actuator_shaft_diameter_m"]
    )
    main = piston_area(geometry["main_accumulator_piston_diameter_m"])
    blowoff = piston_area(
        geometry["blowoff_piston_diameter_m"], geometry["blowoff_shaft_diameter_m"]
    )
    return actuator, main, blowoff


def accumulator_position_m(
    geometry, actuator_a_m, actuator_b_m, blowoff_m, thermal_expansion_m3=0.0
):
    """Return main piston position required by the global volume constraint.

    All piston coordinates and thermal expansion are measured from the same
    initial state. Physical travel stops must be enforced by the caller.
    """
    values = (actuator_a_m, actuator_b_m, blowoff_m, thermal_expansion_m3)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("positions and thermal expansion must be finite")
    actuator, main, blowoff = liquid_piston_areas(geometry)
    return (
        -actuator * (actuator_a_m + actuator_b_m)
        + blowoff * blowoff_m
        - thermal_expansion_m3
    ) / main


def accumulator_velocity_m_s(
    geometry, actuator_a_m_s, actuator_b_m_s, blowoff_m_s, thermal_expansion_m3_s=0.0
):
    """Return the main velocity required by incompressible volume continuity."""
    return accumulator_position_m(
        geometry, actuator_a_m_s, actuator_b_m_s, blowoff_m_s, thermal_expansion_m3_s
    )


def hydraulic_flows_m3_s(geometry, velocities_m_s, thermal_sources_m3_s, pipe_cells):
    """Calculate the unique tree-network flows for prescribed piston velocities.

    Branch flows are positive from the shared line into the chamber; pipe flows
    are positive from inlet to outlet. Thermal sources give volumetric expansion
    in each named liquid node. The four piston velocities must already satisfy
    the global volume constraint, including all node thermal sources.
    """
    if (
        isinstance(pipe_cells, bool)
        or not isinstance(pipe_cells, int)
        or pipe_cells < 1
    ):
        raise ValueError("pipe_cells must be a positive integer")
    piston_names = ("actuator_a", "actuator_b", "main_accumulator", "blowoff")
    node_names = (
        "actuator_a_fluid",
        "actuator_b_fluid",
        "main_fluid",
        "blowoff_fluid",
        "hydraulic_inlet_junction",
        "hydraulic_outlet_junction",
        *(f"hydraulic_pipe_{i}" for i in range(pipe_cells)),
    )
    if set(thermal_sources_m3_s) - set(node_names):
        raise ValueError("unknown liquid thermal expansion source")
    try:
        v = {name: float(velocities_m_s[name]) for name in piston_names}
        s = {name: float(thermal_sources_m3_s.get(name, 0.0)) for name in node_names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "invalid piston velocities or thermal expansion sources"
        ) from exc
    if not all(math.isfinite(value) for value in (*v.values(), *s.values())):
        raise ValueError("piston velocities and thermal sources must be finite")
    actuator, main, blowoff = liquid_piston_areas(geometry)
    flow_a = -actuator * v["actuator_a"] - s["actuator_a_fluid"]
    flow_b = -actuator * v["actuator_b"] - s["actuator_b_fluid"]
    flow_main = -main * v["main_accumulator"] - s["main_fluid"]
    flow_blowoff = blowoff * v["blowoff"] - s["blowoff_fluid"]
    pipe = [s["hydraulic_inlet_junction"] - flow_a - flow_b]
    for i in range(pipe_cells):
        pipe.append(pipe[-1] + s[f"hydraulic_pipe_{i}"])
    residual = pipe[-1] + s["hydraulic_outlet_junction"] - flow_main - flow_blowoff
    return {
        "actuator_a": flow_a,
        "actuator_b": flow_b,
        "main_accumulator": flow_main,
        "blowoff": flow_blowoff,
        "pipe": pipe,
        "continuity_residual_m3_s": residual,
    }


def constrained_accelerations(
    physical,
    velocities_m_s,
    gas_pressures_pa,
    actuator_forces_n,
    thermal_sources_m3_s,
    thermal_source_rates_m3_s2,
    pipe_cells,
    stopped=(),
    temperatures_c=None,
    friction_velocity_scale_m_s=0.005,
    allow_nonpositive_pressure=False,
    source_rate_function=None,
    external_piston_forces_n=None,
):
    """Solve piston accelerations and pressure under liquid continuity.

    Positive applied actuator load drives extension, expelling high-side liquid.
    Liquid is incompressible apart from the supplied thermal volume sources.
    The pipe's inertance and its nonlinear pressure loss are retained. Gas
    pressures are absolute; the returned chamber pressures are also absolute.
    ``stopped`` optionally prescribes fixed coordinates for fixture calculations.
    The time integrator uses elastic end-stop forces and leaves all pistons free.
    Thermal volume sources and their differentiated rates are supplied by the caller.
    An optional source-rate function supplies expansion acceleration as a
    function of piston acceleration for the differentiated thermal constraint.
    """
    geometry, fluid = physical["geometry"], physical["fluid"]
    names = ("actuator_a", "actuator_b", "main_accumulator", "blowoff")
    stopped = set(stopped)
    if stopped - set(names) or len(stopped) == len(names):
        raise ValueError("invalid or fully constrained piston set")
    if any(abs(float(velocities_m_s[name])) > 1e-12 for name in stopped):
        raise ValueError("a stopped piston must have zero velocity")
    sources = dict(thermal_sources_m3_s)
    source_rates = dict(thermal_source_rates_m3_s2)
    flows = hydraulic_flows_m3_s(geometry, velocities_m_s, sources, pipe_cells)
    # Validate the source-rate names with the same circuit topology.
    hydraulic_flows_m3_s(
        geometry, {name: 0.0 for name in names}, source_rates, pipe_cells
    )
    if abs(flows["continuity_residual_m3_s"]) > 1e-12:
        raise ValueError("piston velocities violate incompressible volume continuity")
    temperatures = temperatures_c or {}

    def temp(name):
        return float(temperatures.get(name, 20.0))

    actuator_area, main_area, blowoff_area = liquid_piston_areas(geometry)
    actuator_gas_area = piston_area(geometry["actuator_piston_diameter_m"])
    blowoff_gas_area = piston_area(geometry["blowoff_piston_diameter_m"])
    area_vector = np.array([-actuator_area, -actuator_area, -main_area, blowoff_area])
    masses = np.array([effective_piston_mass_kg(physical, name) for name in names])
    gas = {name: float(gas_pressures_pa[name]) for name in names}
    if not all(math.isfinite(value) and value > 0 for value in gas.values()):
        raise ValueError("gas pressures must be positive and finite")
    inputs = {
        name: float(actuator_forces_n[name]) for name in ("actuator_a", "actuator_b")
    }
    if not all(math.isfinite(value) for value in inputs.values()):
        raise ValueError("actuator forces must be finite")
    ratio = float(physical["seal_friction"]["kinetic_to_static_ratio"])
    if not 0 <= ratio <= 1:
        raise ValueError("kinetic_to_static_ratio must be between zero and one")
    tail = physical["restrictions"]["high_flow_tail_fraction"]

    def curves(label):
        result = []
        for direction in ("compression", "extension"):
            points = physical["restrictions"][label][direction]
            result.append(
                DigressiveCurve.from_points(
                    actuator_area,
                    points["speed_m_s"][0],
                    points["force_n"][0],
                    points["speed_m_s"][1],
                    points["force_n"][1],
                )
            )
        return result

    actuator_curves, main_curves = curves("actuator"), curves("accumulator_inlet")
    branch_dp = {
        "actuator_a": regularized_restriction_pressure_drop(
            flows["actuator_a"], *actuator_curves, tail, zero_flow_transition_m3_s=1e-11
        ),
        "actuator_b": regularized_restriction_pressure_drop(
            flows["actuator_b"], *actuator_curves, tail, zero_flow_transition_m3_s=1e-11
        ),
        "main_accumulator": regularized_restriction_pressure_drop(
            flows["main_accumulator"],
            *main_curves,
            tail,
            zero_flow_transition_m3_s=1e-11,
        ),
    }
    density = float(fluid["density_kg_m3_at_20c"])
    branch_dp["blowoff"] = pipe_pressure_drop(
        flows["blowoff"],
        density,
        float(
            liquid_viscosity(
                (temp("hydraulic_outlet_junction") + temp("blowoff_fluid")) / 2, fluid
            )
        ),
        geometry["blowoff_connection_length_m"],
        geometry["blowoff_connection_diameter_m"],
        fluid["pipe_roughness_m"],
    )
    blowoff_inertance = pipe_inertance(
        density,
        geometry["blowoff_connection_length_m"],
        geometry["blowoff_connection_diameter_m"],
    )
    path = [
        "hydraulic_inlet_junction",
        *(f"hydraulic_pipe_{i}" for i in range(pipe_cells)),
        "hydraulic_outlet_junction",
    ]
    segment_length = geometry["hydraulic_pipe_length_m"] / pipe_cells
    pipe_resistance, pipe_inertia = [], []
    for i, flow in enumerate(flows["pipe"]):
        length = segment_length / 2 if i in (0, pipe_cells) else segment_length
        viscosity = float(
            liquid_viscosity((temp(path[i]) + temp(path[i + 1])) / 2, fluid)
        )
        pipe_resistance.append(
            pipe_pressure_drop(
                flow,
                density,
                viscosity,
                length,
                geometry["hydraulic_pipe_diameter_m"],
                fluid["pipe_roughness_m"],
            )
        )
        pipe_inertia.append(
            pipe_inertance(density, length, geometry["hydraulic_pipe_diameter_m"])
        )
    target_volume_acceleration = sum(float(value) for value in source_rates.values())

    def evaluate(acceleration, outlet_pressure):
        active_rates = (
            source_rates
            if source_rate_function is None
            else source_rate_function(acceleration)
        )
        upstream = sum(
            float(active_rates.get(name, 0.0))
            for name in (
                "actuator_a_fluid",
                "actuator_b_fluid",
                "hydraulic_inlet_junction",
            )
        )
        current_pipe_rates = []
        for i in range(pipe_cells + 1):
            if i:
                upstream += float(active_rates.get(f"hydraulic_pipe_{i-1}", 0.0))
            current_pipe_rates.append(upstream)
        qdot_base = actuator_area * (acceleration[0] + acceleration[1])
        blowoff_inertial_drop = blowoff_inertance * (
            blowoff_area * acceleration[3]
            - float(active_rates.get("blowoff_fluid", 0.0))
        )
        pipe_drops = [
            resistance + inertance * (qdot_base + source_rate)
            for resistance, inertance, source_rate in zip(
                pipe_resistance, pipe_inertia, current_pipe_rates
            )
        ]
        pipe_dp = sum(pipe_drops)
        inlet_pressure = outlet_pressure + pipe_dp
        pressures = {
            "actuator_a": inlet_pressure - branch_dp["actuator_a"],
            "actuator_b": inlet_pressure - branch_dp["actuator_b"],
            "main_accumulator": outlet_pressure - branch_dp["main_accumulator"],
            "blowoff": outlet_pressure - branch_dp["blowoff"] - blowoff_inertial_drop,
        }
        drive = {
            "actuator_a": (gas["actuator_a"] - ATMOSPHERIC_PRESSURE_PA)
            * actuator_gas_area
            - (pressures["actuator_a"] - ATMOSPHERIC_PRESSURE_PA) * actuator_area
            + inputs["actuator_a"],
            "actuator_b": (gas["actuator_b"] - ATMOSPHERIC_PRESSURE_PA)
            * actuator_gas_area
            - (pressures["actuator_b"] - ATMOSPHERIC_PRESSURE_PA) * actuator_area
            + inputs["actuator_b"],
            "main_accumulator": (
                gas["main_accumulator"] - pressures["main_accumulator"]
            )
            * main_area,
            "blowoff": (pressures["blowoff"] - ATMOSPHERIC_PRESSURE_PA) * blowoff_area
            - (gas["blowoff"] - ATMOSPHERIC_PRESSURE_PA) * blowoff_gas_area,
        }
        for name in names:
            drive[name] += float((external_piston_forces_n or {}).get(name, 0.0))
        friction = {}
        for name in names:
            seal = physical["seal_friction"][
                "actuator" if name.startswith("actuator_") else name
            ]
            limit = seal_friction_limit(
                pressures[name] - ATMOSPHERIC_PRESSURE_PA,
                gas[name] - ATMOSPHERIC_PRESSURE_PA,
                seal,
            )
            friction[name] = smooth_friction_force(
                float(velocities_m_s[name]), ratio * limit, friction_velocity_scale_m_s
            )
            drive[name] -= friction[name]
        node_pressures = {path[-1]: outlet_pressure}
        for i in range(len(pipe_drops) - 1, -1, -1):
            node_pressures[path[i]] = node_pressures[path[i + 1]] + pipe_drops[i]
        node_pressures.update(
            {
                "actuator_a_fluid": pressures["actuator_a"],
                "actuator_b_fluid": pressures["actuator_b"],
                "main_fluid": pressures["main_accumulator"],
                "blowoff_fluid": pressures["blowoff"],
            }
        )
        return pressures, drive, friction, node_pressures

    def equations(unknown):
        acceleration, outlet_pressure = unknown[:4], unknown[4] * 1e5
        _, drive, _, _ = evaluate(acceleration, outlet_pressure)
        force_residual = [
            (
                masses[i] * acceleration[i]
                if name in stopped
                else masses[i] * acceleration[i] - drive[name]
            )
            / 1000.0
            for i, name in enumerate(names)
        ]
        target = (
            target_volume_acceleration
            if source_rate_function is None
            else sum(source_rate_function(acceleration).values())
        )
        continuity = (float(area_vector @ acceleration) - target) / actuator_area
        return np.array([*force_residual, continuity])

    # Assemble the piecewise-linear mechanical system analytically. Taking
    # numerical differences of almost-balanced forces loses precision near a
    # loaded stop and corrupts the time integrator's numerical Jacobian.
    zero = np.zeros(4)
    rate_base = (
        source_rates if source_rate_function is None else source_rate_function(zero)
    )
    rate_jac = {name: np.zeros(4) for name in sources}
    for name in source_rates:
        rate_jac.setdefault(name, np.zeros(4))
    if source_rate_function is not None:
        for i in range(4):
            unit = np.zeros(4)
            unit[i] = 1.0
            shifted = source_rate_function(unit)
            for name in shifted:
                rate_jac.setdefault(name, np.zeros(4))[i] = shifted[
                    name
                ] - rate_base.get(name, 0.0)
    upstream_names = [
        "actuator_a_fluid",
        "actuator_b_fluid",
        "hydraulic_inlet_junction",
    ]
    pipe_offset = sum(pipe_resistance)
    pipe_acceleration = np.zeros(4)
    for i, inertance in enumerate(pipe_inertia):
        if i:
            upstream_names.append(f"hydraulic_pipe_{i-1}")
        pipe_offset += inertance * sum(rate_base.get(n, 0.0) for n in upstream_names)
        pipe_acceleration += inertance * (
            np.array([actuator_area, actuator_area, 0.0, 0.0])
            + sum((rate_jac.get(n, zero) for n in upstream_names), start=zero.copy())
        )
    pressure_offset = np.array(
        [
            pipe_offset - branch_dp["actuator_a"],
            pipe_offset - branch_dp["actuator_b"],
            -branch_dp["main_accumulator"],
            -branch_dp["blowoff"]
            + blowoff_inertance * rate_base.get("blowoff_fluid", 0.0),
        ]
    )
    pressure_acceleration = np.vstack(
        [
            pipe_acceleration,
            pipe_acceleration,
            zero,
            blowoff_inertance
            * (
                rate_jac.get("blowoff_fluid", zero)
                - np.array([0.0, 0.0, 0.0, blowoff_area])
            ),
        ]
    )
    ambient = ATMOSPHERIC_PRESSURE_PA
    drive_constant = np.array(
        [
            (gas["actuator_a"] - ambient) * actuator_gas_area
            + ambient * actuator_area
            + inputs["actuator_a"],
            (gas["actuator_b"] - ambient) * actuator_gas_area
            + ambient * actuator_area
            + inputs["actuator_b"],
            gas["main_accumulator"] * main_area,
            -ambient * blowoff_area - (gas["blowoff"] - ambient) * blowoff_gas_area,
        ]
    )
    drive_constant += np.array(
        [float((external_piston_forces_n or {}).get(n, 0.0)) for n in names]
    )
    gamma = np.array(
        [
            smooth_friction_force(velocities_m_s[n], ratio, friction_velocity_scale_m_s)
            for n in names
        ]
    )
    seals = [
        physical["seal_friction"]["actuator" if n.startswith("actuator_") else n]
        for n in names
    ]
    coefficient = np.array([seal["pressure_coefficient_n_pa"] for seal in seals])
    friction_base = gamma * np.array(
        [
            seal["constant_n"]
            + seal["pressure_coefficient_n_pa"] * max(0.0, gas[n] - ambient)
            for n, seal in zip(names, seals)
        ]
    )
    active = np.ones(4, dtype=bool)
    unknown = np.zeros(5)
    for _ in range(12):
        slope = area_vector - gamma * coefficient * active
        constant = (
            drive_constant - friction_base + gamma * coefficient * active * ambient
        )
        matrix = np.zeros((5, 5))
        rhs = np.zeros(5)
        for i, name in enumerate(names):
            matrix[i, i] = masses[i]
            if name not in stopped:
                matrix[i, :4] -= slope[i] * pressure_acceleration[i]
                matrix[i, 4] = -slope[i] * 1e5
                rhs[i] = constant[i] + slope[i] * pressure_offset[i]
            matrix[i] /= 1000.0
            rhs[i] /= 1000.0
        matrix[4, :4] = (
            area_vector - sum(rate_jac.values(), start=zero.copy())
        ) / actuator_area
        rhs[4] = sum(rate_base.values()) / actuator_area
        unknown = np.linalg.solve(matrix, rhs)
        solved_pressure = (
            pressure_offset + pressure_acceleration @ unknown[:4] + unknown[4] * 1e5
        )
        updated = solved_pressure > ambient
        if np.array_equal(active, updated):
            break
        active = updated
    if (
        not np.isfinite(unknown).all()
        or np.linalg.norm(equations(unknown), ord=np.inf) > 1e-9
    ):
        raise RuntimeError("incompressible pressure/acceleration solve failed")
    acceleration = unknown[:4]
    pressures, drive, friction, node_pressures = evaluate(
        acceleration, unknown[4] * 1e5
    )
    if not allow_nonpositive_pressure and min(node_pressures.values()) <= 0:
        raise ValueError(
            "incompressible circuit requires non-positive absolute liquid pressure"
        )
    return {
        "acceleration_m_s2": dict(zip(names, acceleration)),
        "pressures_pa": pressures,
        "stop_reactions_n": {name: -drive[name] for name in stopped},
        "net_drive_n": drive,
        "friction_force_n": friction,
        "flows_m3_s": flows,
        "node_pressures_pa": node_pressures,
        "branch_dissipation_w": {
            name: flows[name] * branch_dp[name] for name in branch_dp
        },
        "pipe_dissipation_w": [
            flow * loss for flow, loss in zip(flows["pipe"], pipe_resistance)
        ],
        "volume_acceleration_residual_m3_s2": float(
            area_vector @ acceleration
            - (
                target_volume_acceleration
                if source_rate_function is None
                else sum(source_rate_function(acceleration).values())
            )
        ),
    }
