"""Finite-volume hydraulic and nitrogen circuit balance assembly.

These functions evaluate instantaneous rates, not a time simulation. Every
node contains mass_kg, internal_energy_j, volume_m3 and optionally
volume_rate_m3_s and heat_rate_w. Pressures are absolute. The time integrator
must supply current piston-dependent volumes and handle stops and friction.
"""

import math
import numpy as np

from .balances import advective_flux_rates, ideal_gas_state
from .hydraulics import gas_pipe_mass_flow


def circuit_nodes_from_initial_state(state):
    """Convert initialized chamber and pipe states into mutable circuit nodes."""
    liquid = {name: dict(node) for name, node in state["liquid_chambers"].items()}
    gas = {name: dict(node) for name, node in state["gas_chambers"].items()}
    for name, destination in (("hydraulic", liquid), ("gas", gas)):
        pipe = state["pipes"][name]
        for i, (mass, energy) in enumerate(
            zip(pipe["mass_kg"], pipe["internal_energy_j"])
        ):
            destination[f"{name}_pipe_{i}"] = {
                "mass_kg": float(mass),
                "internal_energy_j": float(energy),
                "volume_m3": float(pipe["cell_volume_m3"]),
            }
    return liquid, gas


def _rates(nodes, links, enthalpies, pressures):
    names = list(nodes)
    index = {name: i for i, name in enumerate(names)}
    dm, du = advective_flux_rates(
        [enthalpies[name] for name in names],
        [(index[a], index[b], flow) for a, b, flow in links],
        flow_regularization_kg_s=1e-8,
    )
    result = {}
    for i, name in enumerate(names):
        node = nodes[name]
        heat = float(node.get("heat_rate_w", 0.0))
        vdot = float(node.get("volume_rate_m3_s", 0.0))
        if not np.isfinite([heat, vdot]).all():
            raise ValueError(f"invalid heat or volume rate for {name}")
        result[name] = {
            "mass_rate_kg_s": dm[i],
            "energy_rate_w": du[i] + heat - pressures[name] * vdot,
        }
    return result


def nitrogen_rates(
    nodes,
    pipe_cell_count,
    physical,
    pressure_reference_pa=None,
    pressure_offsets_pa=None,
):
    """Rates for the cross-connected actuator gas chambers and reservoir.

    Main and blow-off chambers are closed gas nodes. Gas pipe and short
    actuator connection flows use quasi-steady mean-density Darcy loss; gas
    pipe inertia and choking are not yet modeled.
    """
    if (
        isinstance(pipe_cell_count, bool)
        or not isinstance(pipe_cell_count, int)
        or pipe_cell_count < 1
    ):
        raise ValueError("pipe_cell_count must be positive")
    pipe_names = [f"gas_pipe_{i}" for i in range(pipe_cell_count)]
    required = {
        "actuator_a_gas",
        "actuator_b_gas",
        "actuator_gas_junction",
        "low_pressure_reservoir_gas",
        "main_gas",
        "blowoff_gas",
        *pipe_names,
    }
    if set(nodes) != required:
        raise ValueError(f"gas nodes must be exactly {sorted(required)}")
    gas, geom = physical["gas"], physical["geometry"]
    pressures, temperatures, boundary_temperatures, enthalpies = {}, {}, {}, {}
    cv = gas["specific_heat_cp_j_kg_k"] - gas["specific_gas_constant_j_kg_k"]
    for name, node in nodes.items():
        p, t = ideal_gas_state(
            node["mass_kg"], node["internal_energy_j"], node["volume_m3"], gas
        )
        pressures[name] = (
            pressure_reference_pa + pressure_offsets_pa[name]
            if pressure_offsets_pa is not None and name in pressure_offsets_pa
            else float(p)
        )
        if "boundary_mass_fraction" in node:
            f = float(node["boundary_mass_fraction"])
            boundary_u = float(node["boundary_energy_j"])
            total_u = float(node["internal_energy_j"])
            mass = float(node["mass_kg"])
            if not 0 < f < 1 or not 0 < boundary_u < total_u:
                raise ValueError(f"invalid gas boundary state for {name}")
            boundary_temperatures[name] = boundary_u / (f * mass * cv) - 273.15
            t = (total_u - boundary_u) / ((1 - f) * mass * cv) - 273.15
        temperatures[name] = float(t)
        enthalpies[name] = gas["specific_heat_cp_j_kg_k"] * (float(t) + 273.15)
    links = []

    def connect(a, b, length, diameter):
        flow = gas_pipe_mass_flow(
            pressures[a],
            pressures[b],
            (temperatures[a] + temperatures[b]) / 2 + 273.15,
            gas["specific_gas_constant_j_kg_k"],
            gas["dynamic_viscosity_pa_s_at_25c"],
            length,
            diameter,
            pressure_difference_pa=(
                pressure_offsets_pa[a] - pressure_offsets_pa[b]
                if pressure_offsets_pa is not None
                and a in pressure_offsets_pa
                and b in pressure_offsets_pa
                else None
            ),
        )
        links.append((a, b, flow))

    for actuator in ("actuator_a_gas", "actuator_b_gas"):
        connect(
            actuator,
            "actuator_gas_junction",
            geom["actuator_gas_connection_length_m"],
            geom["actuator_gas_connection_diameter_m"],
        )
    path = ["actuator_gas_junction", *pipe_names, "low_pressure_reservoir_gas"]
    cell_length = geom["gas_pipe_length_m"] / pipe_cell_count
    for i in range(pipe_cell_count + 1):
        length = cell_length / 2 if i in (0, pipe_cell_count) else cell_length
        connect(path[i], path[i + 1], length, geom["gas_pipe_diameter_m"])
    rates = _rates(nodes, links, enthalpies, pressures)
    for name, boundary_t in boundary_temperatures.items():
        node = nodes[name]
        fraction = float(node["boundary_mass_fraction"])
        exchange = fraction * rates[name]["mass_rate_kg_s"]
        weight = 0.5 * (1 + math.tanh(exchange / 1e-8))
        source_h = weight * enthalpies[name] + (1 - weight) * gas[
            "specific_heat_cp_j_kg_k"
        ] * (boundary_t + 273.15)
        # Unless specified otherwise, external heat enters the boundary zone.
        # core_to_boundary_heat_w is internal and cancels from total gas energy.
        rates[name]["boundary_energy_rate_w"] = (
            exchange * source_h
            - fraction * pressures[name] * float(node.get("volume_rate_m3_s", 0))
            + float(node.get("boundary_heat_rate_w", node.get("heat_rate_w", 0)))
            + float(node.get("core_to_boundary_heat_w", 0))
        )
    return {
        "node_rates": rates,
        "pressures_pa": pressures,
        "temperatures_c": temperatures,
        "boundary_temperatures_c": boundary_temperatures,
        "mass_flows_kg_s": links,
    }
