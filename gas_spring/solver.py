"""Experimental stiff, event-driven integration of the coupled reference model.

The model and solver remain provisional and require convergence and physical
validation before their output is used for correlation studies.
"""
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .balances import ideal_gas_state
from .circuit import circuit_nodes_from_initial_state, hydraulic_rates, nitrogen_rates
from .coupling import coupled_rates
from .force_input import interpolate_inputs, shaft_position_m
from .hydraulics import piston_area, pipe_inertance
from .initialization import initialize_system
from .materials import ATMOSPHERIC_PRESSURE_PA, liquid_state, liquid_zone_temperatures
from .mechanics import chamber_volumes, seal_friction_limit
from .sensors import add_sensor_measurements
from .thermal_coupling import thermal_exchange

PISTONS = ("actuator_a", "actuator_b", "main_accumulator", "blowoff")


class _Layout:
    """Pack heterogeneous finite-volume states into a numerical solver vector."""
    def __init__(self, state):
        self.liquid, self.gas = circuit_nodes_from_initial_state(state)
        self.solid = state["solid_temperatures_c"]
        self.fields, values, tolerances = [], [], []
        def add(group, name, field, value, tolerance):
            self.fields.append((group, name, field))
            values.append(float(value))
            tolerances.append(tolerance)
        for name in PISTONS:
            add("position", name, None, state["position_m"][name], 1e-9)
            add("velocity", name, None, state["velocity_m_s"][name], 1e-7)
        for i, value in enumerate(state["pipes"]["hydraulic"]["flow_m3_s"]):
            add("flow", i, None, value, 1e-12)
        for group, nodes in (("liquid", self.liquid), ("gas", self.gas)):
            for name, node in nodes.items():
                add(group, name, "mass_kg", node["mass_kg"], 1e-12)
                add(group, name, "internal_energy_j", node["internal_energy_j"], 1e-5)
                if "boundary_energy_j" in node:
                    add(group, name, "boundary_energy_j", node["boundary_energy_j"], 1e-5)
        for name, value in self.solid.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i, x in enumerate(value):
                    add("solid", name, i, x, 1e-5)
            else:
                add("solid", name, None, value, 1e-5)
        self.y0 = np.asarray(values)
        self.atol = np.asarray(tolerances)
        self.indices = {field: i for i, field in enumerate(self.fields)}

    def unpack(self, y):
        result = {"position": {}, "velocity": {}, "flow": np.empty(sum(f[0] == "flow" for f in self.fields)),
                  "liquid": {k: dict(v) for k, v in self.liquid.items()},
                  "gas": {k: dict(v) for k, v in self.gas.items()},
                  "solid": {k: np.array(v, dtype=float, copy=True) if isinstance(v, (list, tuple, np.ndarray)) else float(v)
                            for k, v in self.solid.items()}}
        for value, (group, name, field) in zip(y, self.fields):
            if group in ("position", "velocity"):
                result[group][name] = float(value)
            elif group == "flow":
                result[group][name] = float(value)
            elif group == "solid":
                if field is None:
                    result[group][name] = float(value)
                else:
                    result[group][name][field] = float(value)
            else:
                result[group][name][field] = float(value)
        return result

    def derivative(self, rate):
        out = np.empty(len(self.fields))
        for i, (group, name, field) in enumerate(self.fields):
            if group == "position":
                out[i] = rate["position_rate_m_s"][name]
            elif group == "velocity":
                out[i] = rate["velocity_rate_m_s2"][name]
            elif group == "flow":
                out[i] = rate["liquid"]["pipe_flow_rate_m3_s2"][name]
            elif group == "solid":
                value = rate["thermal"]["solid_temperature_rate_c_s"][name]
                out[i] = value if field is None else value[field]
            else:
                key = {"mass_kg": "mass_rate_kg_s", "internal_energy_j": "energy_rate_w",
                       "boundary_energy_j": "boundary_energy_rate_w"}[field]
                out[i] = rate[group]["node_rates"][name][key]
        return out


def _force_state(physical, run, state, inputs):
    """Compute piston drives and friction for event detection without flows."""
    g = physical["geometry"]
    x = state["position"].copy()
    for name in ("actuator_a", "actuator_b"):
        x[name] = min(max(x[name], -g["actuator_compression_travel_m"]), g["actuator_extension_travel_m"])
    x["blowoff"] = min(max(x["blowoff"], 0), run["blowoff_max_travel_m"])
    x["main_accumulator"] = min(x["main_accumulator"], g["main_accumulator_expansion_stop_m"])
    volumes = chamber_volumes(g, x["actuator_a"], x["actuator_b"], x["main_accumulator"], x["blowoff"])
    pressures = {}
    for name in ("actuator_a_fluid", "actuator_b_fluid", "main_fluid", "blowoff_fluid"):
        node = state["liquid"][name]
        pressures[name] = float(liquid_state(node["mass_kg"], node["internal_energy_j"], volumes[name], physical["fluid"])[0])-ATMOSPHERIC_PRESSURE_PA
    for name in ("actuator_a_gas", "actuator_b_gas", "main_gas", "blowoff_gas"):
        node = state["gas"][name]
        pressures[name] = float(ideal_gas_state(node["mass_kg"], node["internal_energy_j"], volumes[name], physical["gas"])[0])-ATMOSPHERIC_PRESSURE_PA
    af = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    ag = piston_area(g["actuator_piston_diameter_m"])
    am = piston_area(g["main_accumulator_piston_diameter_m"])
    bf = piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"])
    bg = piston_area(g["blowoff_piston_diameter_m"])
    force = {name: pressures[name+"_gas"]*ag-pressures[name+"_fluid"]*af-inputs[name+"_force_n"] for name in ("actuator_a", "actuator_b")}
    force["main_accumulator"] = (pressures["main_gas"]-pressures["main_fluid"])*am
    force["blowoff"] = pressures["blowoff_fluid"]*bf-pressures["blowoff_gas"]*bg
    friction = {name: seal_friction_limit(pressures[name+"_fluid"], pressures[name+"_gas"], physical["seal_friction"]["actuator"]) for name in ("actuator_a", "actuator_b")}
    friction["main_accumulator"] = seal_friction_limit(pressures["main_fluid"], pressures["main_gas"], physical["seal_friction"]["main_accumulator"])
    friction["blowoff"] = seal_friction_limit(pressures["blowoff_fluid"], pressures["blowoff_gas"], physical["seal_friction"]["blowoff"])
    return force, friction


def _record(layout, y, time_s, physical, run, inputs):
    s = layout.unpack(y)
    result = {"time_s": float(time_s), "hot_ambient_c": inputs["hot_ambient_c"],
              "duct_air_speed_m_s": inputs["duct_air_speed_m_s"],
              "cooler_region_temperature_c": s["solid"]["cooler_region"]}
    g = physical["geometry"]
    output_stops = {
        "actuator_a": (-g["actuator_compression_travel_m"], g["actuator_extension_travel_m"]),
        "actuator_b": (-g["actuator_compression_travel_m"], g["actuator_extension_travel_m"]),
        "main_accumulator": (None, g["main_accumulator_expansion_stop_m"]),
        "blowoff": (0.0, run["blowoff_max_travel_m"]),
    }
    for name, (lo, hi) in output_stops.items():
        x = s["position"][name]
        if (lo is not None and x < lo-1e-9) or (hi is not None and x > hi+1e-9):
            raise RuntimeError(f"{name} exceeded its travel stop")
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        s["position"][name] = x
    vols = chamber_volumes(g, s["position"]["actuator_a"], s["position"]["actuator_b"],
                           s["position"]["main_accumulator"], s["position"]["blowoff"])
    for nodes in (s["liquid"], s["gas"]):
        for name, node in nodes.items():
            if name in vols:
                node["volume_m3"] = vols[name]
    for group in ("liquid", "gas"):
        for name, node in s[group].items():
            volume = vols.get(name, node["volume_m3"])
            if group == "liquid":
                pressure, temp = liquid_state(node["mass_kg"], node["internal_energy_j"], volume, physical["fluid"])
                if "boundary_energy_j" in node:
                    pressure, temp, boundary = liquid_zone_temperatures(node["mass_kg"], node["internal_energy_j"],
                        volume, node["boundary_energy_j"], node["boundary_mass_fraction"], physical["fluid"])
                    result[f"truth_{name}_boundary_temperature_c"] = boundary
                    result[f"truth_{name}_boundary_energy_j"] = node["boundary_energy_j"]
            else:
                pressure, temp = ideal_gas_state(node["mass_kg"], node["internal_energy_j"], volume, physical["gas"])
                if "boundary_energy_j" in node:
                    cv = physical["gas"]["specific_heat_cp_j_kg_k"] - physical["gas"]["specific_gas_constant_j_kg_k"]
                    f = node["boundary_mass_fraction"]
                    temp = (node["internal_energy_j"] - node["boundary_energy_j"])/((1-f)*node["mass_kg"]*cv)-273.15
                    result[f"truth_{name}_boundary_temperature_c"] = node["boundary_energy_j"]/(f*node["mass_kg"]*cv)-273.15
                    result[f"truth_{name}_boundary_energy_j"] = node["boundary_energy_j"]
            result[f"truth_{name}_pressure_bar"] = (float(pressure)-ATMOSPHERIC_PRESSURE_PA)/1e5
            result[f"truth_{name}_temperature_c"] = float(temp)
            result[f"truth_{name}_mass_kg"] = node["mass_kg"]
            result[f"truth_{name}_internal_energy_j"] = node["internal_energy_j"]
            result[f"truth_{name}_volume_m3"] = volume
    for channel, node in (("low_pressure_gas", "low_pressure_reservoir_gas"),):
        for kind in ("pressure_bar", "temperature_c"):
            result[f"truth_{channel}_{kind}"] = result[f"truth_{node}_{kind}"]
    for name in PISTONS:
        result[f"truth_{name}_piston_position_m"] = s["position"][name]
        result[f"truth_{name}_piston_velocity_m_s"] = s["velocity"][name]
    drives, friction = _force_state(physical, run, s, inputs)
    for name in PISTONS:
        result[f"truth_{name}_drive_force_n"] = drives[name]
        result[f"truth_{name}_friction_limit_n"] = friction[name]
    for name in ("actuator_a", "actuator_b"):
        force = inputs[name+"_force_n"]
        result[f"truth_{name}_force_n"] = force
        result[f"truth_{name}_displacement_mm"] = 1000*shaft_position_m(s["position"][name], force, g["actuator_input_shaft_stiffness_n_m"])
    result["truth_blowoff_displacement_mm"] = 1000*s["position"]["blowoff"]
    for i, flow in enumerate(s["flow"]):
        result[f"truth_hydraulic_pipe_flow_{i}_m3_s"] = flow
    # Include algebraic branch and gas-pipe flows, in addition to inertial
    # hydraulic pipe flow states. These are not separately measured channels.
    for rates in (hydraulic_rates(s["liquid"], s["flow"], physical),
                  nitrogen_rates(s["gas"], len(s["flow"])-1, physical)):
        for a, b, flow in rates["mass_flows_kg_s"]:
            result[f"truth_mass_flow_{a}_to_{b}_kg_s"] = flow
    for name, value in s["solid"].items():
        if isinstance(value, np.ndarray):
            for i, temp in enumerate(value):
                result[f"truth_{name}_{i}_temperature_c"] = temp
        else:
            result[f"truth_{name}_temperature_c"] = value
    thermal = thermal_exchange(physical, run, s["liquid"], s["gas"], s["solid"],
                               s["position"], inputs["hot_ambient_c"], inputs["duct_air_speed_m_s"])
    solid_energy = sum(capacity*result[f"truth_{name}_temperature_c"]
                       for name, capacity in thermal["solid_heat_capacity_j_k"].items())
    piston_energy = sum(.5*physical["moving_masses_kg"][name]*s["velocity"][name]**2 for name in PISTONS)
    count = len(s["flow"])-1
    cell_length = g["hydraulic_pipe_length_m"]/count
    pipe_energy = sum(.5*pipe_inertance(physical["fluid"]["density_kg_m3_at_20c"],
                                       cell_length/2 if i in (0, count) else cell_length,
                                       g["hydraulic_pipe_diameter_m"])*flow**2
                      for i, flow in enumerate(s["flow"]))
    result["truth_liquid_mass_total_kg"] = sum(node["mass_kg"] for node in s["liquid"].values())
    result["truth_nitrogen_mass_total_kg"] = sum(node["mass_kg"] for node in s["gas"].values())
    result["truth_system_energy_j"] = (solid_energy + piston_energy + pipe_energy
        + sum(node["internal_energy_j"] for group in (s["liquid"], s["gas"]) for node in group.values()))
    # The massless input shafts are algebraic; their elastic energy is recorded
    # separately. The energy audit uses force power at the piston coordinates.
    result["truth_input_shaft_strain_energy_j"] = sum(
        inputs[name+"_force_n"]**2/(2*g["actuator_input_shaft_stiffness_n_m"])
        for name in ("actuator_a", "actuator_b"))
    af = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    ag = piston_area(g["actuator_piston_diameter_m"])
    bf = piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"])
    bg = piston_area(g["blowoff_piston_diameter_m"])
    volume_rate = (ag-af)*(s["velocity"]["actuator_a"]+s["velocity"]["actuator_b"]) + (bf-bg)*s["velocity"]["blowoff"]
    input_power = -sum(inputs[name+"_force_n"]*s["velocity"][name] for name in ("actuator_a", "actuator_b"))
    result["truth_external_power_w"] = thermal["external_heat_w"] + input_power - ATMOSPHERIC_PRESSURE_PA*volume_rate
    return result


def simulate(physical, run, inputs, event_callback=None):
    """Integrate the provisional coupled model and return truth and sensors.

    Samples are emitted at the supplied input times. Inputs are linearly
    interpolated between samples. Stick, release and ideal zero-restitution
    stop impacts use event-driven modes; impact energy enters the piston.
    This implementation must be checked for time-step and spatial convergence
    before its results are used as a reference model.
    """
    options = run.get("solver", {})
    cells, wall_cells = options.get("pipe_cells", 3), options.get("actuator_wall_cells", 6)
    rtol, max_step = float(options.get("relative_tolerance", 1e-5)), float(options.get("max_step_s", .05))
    max_events = options.get("max_events", 10000)
    integer_options = (cells, wall_cells, max_events)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in integer_options) or not 0 < rtol < 1 or not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("invalid numerical solver settings")
    cells, wall_cells, max_events = int(cells), int(wall_cells), int(max_events)
    times = np.asarray(inputs["time_s"], dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("input time must be strictly increasing and finite")
    first = inputs.iloc[0]
    state = initialize_system(physical, run, first["actuator_a_force_n"], first["actuator_b_force_n"],
                              time_s=times[0], pipe_cells=cells, actuator_wall_cells=wall_cells)
    layout, modes = _Layout(state), {name: 0 for name in PISTONS}
    g = physical["geometry"]
    stops = {"actuator_a": (-g["actuator_compression_travel_m"], g["actuator_extension_travel_m"]),
             "actuator_b": (-g["actuator_compression_travel_m"], g["actuator_extension_travel_m"]),
             "main_accumulator": (None, g["main_accumulator_expansion_stop_m"]),
             "blowoff": (0.0, run["blowoff_max_travel_m"])}
    position_idx = {name: layout.indices[("position", name, None)] for name in PISTONS}
    velocity_idx = {name: layout.indices[("velocity", name, None)] for name in PISTONS}
    state_y, now, sample_idx, records, event_count = layout.y0.copy(), float(times[0]), 0, [], 0
    # A small breakaway margin avoids chatter at the stick/release boundary.
    force_tolerance_n = float(options.get("stiction_release_margin_n", 1e-6))
    if not np.isfinite(force_tolerance_n) or force_tolerance_n < 0:
        raise ValueError("stiction_release_margin_n must be non-negative")

    def rhs(t, y):
        s = layout.unpack(y)
        u = interpolate_inputs(inputs, t)
        rate = coupled_rates(physical, run, s["liquid"], s["gas"], s["flow"], s["position"], s["velocity"],
                             {"actuator_a": u["actuator_a_force_n"], "actuator_b": u["actuator_b_force_n"]},
                             s["solid"], u["hot_ambient_c"], u["duct_air_speed_m_s"], modes)
        return layout.derivative(rate)

    while now < times[-1]:
        events, metadata = [], []
        for name in PISTONS:
            mode = modes[name]
            if mode == 0:
                def release(t, y, label=name):
                    s = layout.unpack(y)
                    drives, limits = _force_state(physical, run, s, interpolate_inputs(inputs, t))
                    x, (lo, hi) = s["position"][label], stops[label]
                    drive = drives[label]
                    if lo is not None and x <= lo + 1e-10:
                        available = drive
                    elif hi is not None and x >= hi - 1e-10:
                        available = -drive
                    else:
                        available = abs(drive)
                    return available - limits[label] - force_tolerance_n
                release.terminal, release.direction = True, 1
                events.append(release)
                metadata.append((name, "release"))
            else:
                def zero_velocity(t, y, label=name, direction=mode):
                    return direction*y[velocity_idx[label]]
                zero_velocity.terminal, zero_velocity.direction = True, -1
                events.append(zero_velocity)
                metadata.append((name, "zero"))
                lo, hi = stops[name]
                if (mode > 0 and hi is not None) or (mode < 0 and lo is not None):
                    stop = hi if mode > 0 else lo
                    def impact(t, y, label=name, direction=mode, bound=stop):
                        return direction*(bound-y[position_idx[label]])
                    impact.terminal, impact.direction = True, -1
                    events.append(impact)
                    metadata.append((name, "impact"))
        solution = solve_ivp(rhs, (now, float(times[-1])), state_y, method="BDF", rtol=rtol,
                             atol=layout.atol, max_step=max_step, events=events, dense_output=True)
        if not solution.success:
            raise RuntimeError(f"coupled integration failed at {solution.t[-1]:g} s: {solution.message}")
        end = float(solution.t[-1])
        triggered = next((i for i, points in enumerate(solution.t_events) if len(points)), None)
        # Samples coinciding with an event are recorded after its state reset.
        while sample_idx < len(times) and (times[sample_idx] < end-1e-11 or (triggered is None and times[sample_idx] <= end)):
            stamp = float(times[sample_idx])
            records.append(_record(layout, solution.sol(stamp), stamp, physical, run, interpolate_inputs(inputs, stamp)))
            sample_idx += 1
        state_y = solution.y[:, -1].copy()
        now = end
        if triggered is None:
            break
        event_count += 1
        if event_count > max_events:
            raise RuntimeError(f"coupled integration exceeded the event limit at {now:.9g} s")
        name, kind = metadata[triggered]
        if event_callback is not None:
            event_callback(now, name, kind)
        if kind == "impact":
            old_velocity = state_y[velocity_idx[name]]
            bound = stops[name][1] if modes[name] > 0 else stops[name][0]
            state_y[position_idx[name]] = bound
            state_y[velocity_idx[name]] = 0.0
            energy = .5*physical["moving_masses_kg"][name]*old_velocity**2
            prefix = "main" if name == "main_accumulator" else name
            for side in ("fluid", "gas"):
                idx = layout.indices[("solid", f"{prefix}_piston_{side}", None)]
                state_y[idx] += energy/(physical["moving_masses_kg"][name]*physical["thermal"]["metal_specific_heat_j_kg_k"])
            modes[name] = 0
        else:
            state_y[velocity_idx[name]] = 0.0
            s = layout.unpack(state_y)
            drives, friction = _force_state(physical, run, s, interpolate_inputs(inputs, now))
            drive = drives[name]
            lo, hi = stops[name]
            x = s["position"][name]
            allowed = not ((lo is not None and x <= lo+1e-10 and drive < 0) or
                           (hi is not None and x >= hi-1e-10 and drive > 0))
            if allowed and (kind == "release" or abs(drive) > friction[name]+force_tolerance_n):
                modes[name] = 1 if drive > 0 else -1
            else:
                modes[name] = 0
    # An event exactly at the end is recorded after its state reset.
    while sample_idx < len(times) and abs(times[sample_idx]-now) <= 1e-11:
        stamp = float(times[sample_idx])
        records.append(_record(layout, state_y, stamp, physical, run, interpolate_inputs(inputs, stamp)))
        sample_idx += 1
    if sample_idx != len(times):
        raise RuntimeError("integration did not produce every requested sample")
    truth = pd.DataFrame.from_records(records)
    power = truth["truth_external_power_w"].to_numpy()
    intervals = np.diff(truth["time_s"].to_numpy())
    accumulated = np.r_[0.0, np.cumsum((power[1:]+power[:-1])*intervals/2)]
    truth["truth_sampled_energy_balance_residual_j"] = (truth["truth_system_energy_j"]
        - truth["truth_system_energy_j"].iloc[0] - accumulated)
    return add_sensor_measurements(truth, physical, run)
