"""Finite-volume hydraulic and nitrogen circuit balance assembly.

These functions evaluate instantaneous rates, not a time simulation. Every
node contains mass_kg, internal_energy_j, volume_m3 and optionally
volume_rate_m3_s and heat_rate_w. Pressures are absolute. The time integrator
must supply current piston-dependent volumes and handle stops and friction.
"""
import numpy as np

from .balances import advective_flux_rates, ideal_gas_state
from .hydraulics import (DigressiveCurve, gas_pipe_mass_flow, liquid_pipe_flow,
                         pipe_inertance, pipe_pressure_drop, piston_area, restriction_flow)
from .materials import liquid_internal_energy, liquid_state, liquid_viscosity, liquid_zone_temperatures


def circuit_nodes_from_initial_state(state):
    """Convert initialized chamber and pipe states into mutable circuit nodes."""
    liquid = {name: dict(node) for name, node in state["liquid_chambers"].items()}
    gas = {name: dict(node) for name, node in state["gas_chambers"].items()}
    for name, destination in (("hydraulic", liquid), ("gas", gas)):
        pipe = state["pipes"][name]
        for i, (mass, energy) in enumerate(zip(pipe["mass_kg"], pipe["internal_energy_j"])):
            destination[f"{name}_pipe_{i}"] = {
                "mass_kg": float(mass), "internal_energy_j": float(energy),
                "volume_m3": float(pipe["cell_volume_m3"]),
            }
    return liquid, gas


def _rates(nodes, links, enthalpies, pressures):
    names = list(nodes)
    index = {name: i for i, name in enumerate(names)}
    dm, du = advective_flux_rates([enthalpies[name] for name in names],
                                  [(index[a], index[b], flow) for a, b, flow in links])
    result = {}
    for i, name in enumerate(names):
        node = nodes[name]
        heat = float(node.get("heat_rate_w", 0.0))
        vdot = float(node.get("volume_rate_m3_s", 0.0))
        if not np.isfinite([heat, vdot]).all():
            raise ValueError(f"invalid heat or volume rate for {name}")
        result[name] = {"mass_rate_kg_s": dm[i],
                        "energy_rate_w": du[i] + heat - pressures[name] * vdot}
    return result


def _curves(physical, name):
    geom = physical["geometry"]
    area = piston_area(geom["actuator_piston_diameter_m"], geom["actuator_shaft_diameter_m"])
    curves = []
    for direction in ("compression", "extension"):
        points = physical["restrictions"][name][direction]
        curves.append(DigressiveCurve.from_points(area, points["speed_m_s"][0], points["force_n"][0],
                                                   points["speed_m_s"][1], points["force_n"][1]))
    return curves


def hydraulic_rates(nodes, pipe_flows_m3_s, physical):
    """Rates for two actuator restrictions, inertial pipe and blow-off branch.

    Pipe flows run from the actuator-side junction to the far junction. The
    blow-off has a provisional short, resistive connection. The accumulator
    inlet restriction comes after that branch. Flow inertia uses constant
    reference density; its kinetic-energy rate is taken from the adjacent
    liquid energies so this discrete internal network balances energy.
    """
    q = np.asarray(pipe_flows_m3_s, dtype=float)
    if q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("pipe flows must be a finite vector for at least one pipe cell")
    count = len(q)-1
    pipe_names = [f"hydraulic_pipe_{i}" for i in range(count)]
    required = {"actuator_a_fluid", "actuator_b_fluid", "hydraulic_inlet_junction",
                "hydraulic_outlet_junction", "main_fluid", "blowoff_fluid", *pipe_names}
    if set(nodes) != required:
        raise ValueError(f"hydraulic nodes must be exactly {sorted(required)}")
    fluid, geom = physical["fluid"], physical["geometry"]
    pressures, temperatures, boundary_temperatures, densities, enthalpies = {}, {}, {}, {}, {}
    for name, node in nodes.items():
        mass, volume = float(node["mass_kg"]), float(node["volume_m3"])
        p, t = liquid_state(mass, node["internal_energy_j"], volume, fluid)
        if "boundary_mass_fraction" in node:
            p, t, boundary_t = liquid_zone_temperatures(mass, node["internal_energy_j"], volume,
                node["boundary_energy_j"], node["boundary_mass_fraction"], fluid)
            boundary_temperatures[name] = boundary_t
            fraction = node["boundary_mass_fraction"]
            specific_e = (node["internal_energy_j"]-node["boundary_energy_j"])/((1-fraction)*mass)
        else:
            specific_e = float(liquid_internal_energy(t, mass/volume, fluid))
        pressures[name], temperatures[name], densities[name] = float(p), float(t), mass/volume
        enthalpies[name] = float(specific_e + p/(mass/volume))
    inlet, outlet = "hydraulic_inlet_junction", "hydraulic_outlet_junction"
    tail = physical["restrictions"]["high_flow_tail_fraction"]
    actuator_curve = _curves(physical, "actuator")
    main_curve = _curves(physical, "accumulator_inlet")
    # Positive actuator restriction flow is from the shared line into the actuator
    # (compression); positive main-inlet flow is into the main accumulator.
    volume_links = []
    for a in ("actuator_a_fluid", "actuator_b_fluid"):
        volume_links.append((inlet, a, restriction_flow(pressures[inlet]-pressures[a], *actuator_curve, tail)))
    volume_links.append((outlet, "main_fluid", restriction_flow(pressures[outlet]-pressures["main_fluid"], *main_curve, tail)))
    branch_temp = (temperatures[outlet]+temperatures["blowoff_fluid"])/2
    branch_density = (densities[outlet]+densities["blowoff_fluid"])/2
    branch_flow = liquid_pipe_flow(pressures[outlet]-pressures["blowoff_fluid"], branch_density,
                                   float(liquid_viscosity(branch_temp, fluid)), geom["blowoff_connection_length_m"],
                                   geom["blowoff_connection_diameter_m"], fluid["pipe_roughness_m"])
    volume_links.append((outlet, "blowoff_fluid", branch_flow))
    path = [inlet, *pipe_names, outlet]
    for i, flow in enumerate(q):
        volume_links.append((path[i], path[i+1], flow))
    mass_links = [(a, b, flow * densities[a if flow >= 0 else b]) for a, b, flow in volume_links]
    rates = _rates(nodes, mass_links, enthalpies, pressures)
    qdot = np.empty_like(q)
    density_ref = float(fluid["density_kg_m3_at_20c"])
    cell_length = geom["hydraulic_pipe_length_m"] / count
    for i, flow in enumerate(q):
        a, b = path[i], path[i+1]
        segment_length = cell_length / 2 if i in (0, count) else cell_length
        inertance = pipe_inertance(density_ref, segment_length, geom["hydraulic_pipe_diameter_m"])
        rho = (densities[a]+densities[b])/2
        mu = float(liquid_viscosity((temperatures[a]+temperatures[b])/2, fluid))
        loss = pipe_pressure_drop(flow, rho, mu, segment_length, geom["hydraulic_pipe_diameter_m"], fluid["pipe_roughness_m"])
        qdot[i] = (pressures[a]-pressures[b]-loss)/inertance
        kinetic_rate = inertance * flow * qdot[i]
        rates[a]["energy_rate_w"] -= kinetic_rate/2
        rates[b]["energy_rate_w"] -= kinetic_rate/2
    for name in boundary_temperatures:
        node = nodes[name]
        fraction = float(node["boundary_mass_fraction"])
        exchange = fraction*rates[name]["mass_rate_kg_s"]
        boundary_enthalpy = node["boundary_energy_j"]/(fraction*node["mass_kg"]) + pressures[name]/densities[name]
        source_h = enthalpies[name] if exchange >= 0 else boundary_enthalpy
        rates[name]["boundary_energy_rate_w"] = (
            exchange*source_h - fraction*pressures[name]*float(node.get("volume_rate_m3_s", 0))
            + float(node.get("boundary_heat_rate_w", node.get("heat_rate_w", 0)))
            + float(node.get("core_to_boundary_heat_w", 0)))
    return {"node_rates": rates, "pressures_pa": pressures, "temperatures_c": temperatures,
            "boundary_temperatures_c": boundary_temperatures,
            "pipe_flow_rate_m3_s2": qdot, "mass_flows_kg_s": mass_links}


def nitrogen_rates(nodes, pipe_cell_count, physical):
    """Rates for the cross-connected actuator gas chambers and reservoir.

    Main and blow-off chambers are closed gas nodes. Gas pipe and short
    actuator connection flows use quasi-steady mean-density Darcy loss; gas
    pipe inertia and choking are not yet modeled.
    """
    if isinstance(pipe_cell_count, bool) or not isinstance(pipe_cell_count, int) or pipe_cell_count < 1:
        raise ValueError("pipe_cell_count must be positive")
    pipe_names = [f"gas_pipe_{i}" for i in range(pipe_cell_count)]
    required = {"actuator_a_gas", "actuator_b_gas", "actuator_gas_junction",
                "low_pressure_reservoir_gas", "main_gas", "blowoff_gas", *pipe_names}
    if set(nodes) != required:
        raise ValueError(f"gas nodes must be exactly {sorted(required)}")
    gas, geom = physical["gas"], physical["geometry"]
    pressures, temperatures, boundary_temperatures, enthalpies = {}, {}, {}, {}
    cv = gas["specific_heat_cp_j_kg_k"] - gas["specific_gas_constant_j_kg_k"]
    for name, node in nodes.items():
        p, t = ideal_gas_state(node["mass_kg"], node["internal_energy_j"], node["volume_m3"], gas)
        pressures[name] = float(p)
        if "boundary_mass_fraction" in node:
            f = float(node["boundary_mass_fraction"])
            boundary_u = float(node["boundary_energy_j"])
            total_u = float(node["internal_energy_j"])
            mass = float(node["mass_kg"])
            if not 0 < f < 1 or not 0 < boundary_u < total_u:
                raise ValueError(f"invalid gas boundary state for {name}")
            boundary_temperatures[name] = boundary_u / (f*mass*cv) - 273.15
            t = (total_u - boundary_u) / ((1-f)*mass*cv) - 273.15
        temperatures[name] = float(t)
        enthalpies[name] = gas["specific_heat_cp_j_kg_k"]*(float(t)+273.15)
    links = []
    def connect(a, b, length, diameter):
        flow = gas_pipe_mass_flow(pressures[a], pressures[b], (temperatures[a]+temperatures[b])/2+273.15,
                                  gas["specific_gas_constant_j_kg_k"], gas["dynamic_viscosity_pa_s_at_25c"],
                                  length, diameter)
        links.append((a, b, flow))
    for actuator in ("actuator_a_gas", "actuator_b_gas"):
        connect(actuator, "actuator_gas_junction", geom["actuator_gas_connection_length_m"], geom["actuator_gas_connection_diameter_m"])
    path = ["actuator_gas_junction", *pipe_names, "low_pressure_reservoir_gas"]
    cell_length = geom["gas_pipe_length_m"] / pipe_cell_count
    for i in range(pipe_cell_count+1):
        length = cell_length/2 if i in (0, pipe_cell_count) else cell_length
        connect(path[i], path[i+1], length, geom["gas_pipe_diameter_m"])
    rates = _rates(nodes, links, enthalpies, pressures)
    for name, boundary_t in boundary_temperatures.items():
        node = nodes[name]
        fraction = float(node["boundary_mass_fraction"])
        exchange = fraction * rates[name]["mass_rate_kg_s"]
        source_h = enthalpies[name] if exchange >= 0 else gas["specific_heat_cp_j_kg_k"]*(boundary_t+273.15)
        # Unless specified otherwise, external heat enters the boundary zone.
        # core_to_boundary_heat_w is internal and cancels from total gas energy.
        rates[name]["boundary_energy_rate_w"] = (
            exchange * source_h - fraction * pressures[name] * float(node.get("volume_rate_m3_s", 0))
            + float(node.get("boundary_heat_rate_w", node.get("heat_rate_w", 0)))
            + float(node.get("core_to_boundary_heat_w", 0)))
    return {"node_rates": rates, "pressures_pa": pressures, "temperatures_c": temperatures,
            "boundary_temperatures_c": boundary_temperatures, "mass_flows_kg_s": links}
