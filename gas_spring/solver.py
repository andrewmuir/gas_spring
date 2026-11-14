"""Incompressible integration with elastic end-stop contact."""

from collections import Counter
import copy
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from .circuit import circuit_nodes_from_initial_state
from .coupling import IncompressibleSystem, PISTONS
from .force_input import interpolate_inputs, shaft_position_m
from .initialization import initialize_system
from .materials import ATMOSPHERIC_PRESSURE_PA, liquid_temperatures
from .mechanics import effective_piston_mass_kg
from .hydraulics import pipe_inertance
from .sensors import add_sensor_measurements


class _Layout:
    """Only independent piston positions/velocities are integration states."""

    def __init__(self, state, system, dependent):
        self.template, self.system, self.dependent = (
            copy.deepcopy(state),
            system,
            dependent,
        )
        self.fields, values, tolerances = [], [], []
        self.reference_gas = "low_pressure_reservoir_gas"
        self.difference_gas = set(state["gas"]) - {
            self.reference_gas,
            "main_gas",
            "blowoff_gas",
        }

        def add(group, name, field, value, tolerance):
            self.fields.append((group, name, field))
            values.append(value)
            tolerances.append(tolerance)

        for name in PISTONS:
            if name != dependent:
                add("position", name, None, state["position"][name], 1e-9)
                add("velocity", name, None, state["velocity"][name], 1e-7)
        for group in ("liquid", "gas"):
            for name, node in state[group].items():
                for field, tol in (
                    ("mass_kg", 1e-12),
                    ("internal_energy_j", 1e-5),
                    ("boundary_energy_j", 1e-5),
                ):
                    if field in node:
                        value = node[field]
                        if (
                            group == "gas"
                            and field == "internal_energy_j"
                            and name in self.difference_gas
                        ):
                            reference = state["gas"][self.reference_gas]
                            value -= (
                                reference["internal_energy_j"]
                                * node["volume_m3"]
                                / reference["volume_m3"]
                            )
                            # Keep an energy tolerance comparable to the
                            # physical gas-energy states. A nanajoule limit
                            # would unnecessarily resolve sub-millipascal
                            # pressure transients in the connecting pipes.
                            tol = 1e-6
                        add(group, name, field, value, tol)
        for name, value in state["solid"].items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i, v in enumerate(value):
                    add("solid", name, i, v, 1e-5)
            else:
                add("solid", name, None, value, 1e-5)
        self.y0, self.atol = np.array(values, dtype=float), np.array(tolerances)
        self.indices = {field: i for i, field in enumerate(self.fields)}

    def unpack(self, y):
        s = copy.deepcopy(self.template)
        for value, (group, name, field) in zip(y, self.fields):
            if group in ("position", "velocity"):
                s[group][name] = float(value)
            elif group == "solid" and field is None:
                s[group][name] = float(value)
            else:
                s[group][name][field] = float(value)
        j = self.system.j
        d = self.dependent
        s["position"][d] = (
            self.system.expansion(s["liquid"])
            - sum(j[n] * s["position"][n] for n in PISTONS if n != d)
        ) / j[d]
        self.system.volumes(s)
        reference = s["gas"][self.reference_gas]
        for name in self.difference_gas:
            s["gas"][name]["internal_energy_j"] += (
                reference["internal_energy_j"]
                * s["gas"][name]["volume_m3"]
                / reference["volume_m3"]
            )
        return s

    def derivative(self, rate, state):
        out = np.empty(len(self.fields))
        for i, (group, name, field) in enumerate(self.fields):
            if group == "position":
                out[i] = rate["position_rate_m_s"][name]
            elif group == "velocity":
                out[i] = rate["velocity_rate_m_s2"][name]
            elif group == "solid":
                value = rate["thermal"]["solid_temperature_rate_c_s"][name]
                out[i] = value if field is None else value[field]
            else:
                key = {
                    "mass_kg": "mass_rate_kg_s",
                    "internal_energy_j": "energy_rate_w",
                    "boundary_energy_j": "boundary_energy_rate_w",
                }[field]
                out[i] = rate[group]["node_rates"][name][key]
                if (
                    group == "gas"
                    and field == "internal_energy_j"
                    and name in self.difference_gas
                ):
                    reference = state["gas"][self.reference_gas]
                    node = state["gas"][name]
                    out[i] -= (
                        rate["gas"]["node_rates"][self.reference_gas]["energy_rate_w"]
                        * node["volume_m3"]
                        / reference["volume_m3"]
                        + reference["internal_energy_j"]
                        * node.get("volume_rate_m3_s", 0.0)
                        / reference["volume_m3"]
                    )
        return out


def _record(s, rate, t, system, u):
    p, run = system.physical, system.run
    result = {
        "time_s": t,
        "hot_ambient_c": u["hot_ambient_c"],
        "duct_air_speed_m_s": u["duct_air_speed_m_s"],
        "cooler_region_temperature_c": s["solid"]["cooler_region"],
    }
    for group in ("liquid", "gas"):
        for name, node in s[group].items():
            if group == "liquid":
                _, temp, boundary = liquid_temperatures(
                    node["mass_kg"],
                    node["internal_energy_j"],
                    p["fluid"],
                    node.get("boundary_energy_j"),
                    node.get("boundary_mass_fraction"),
                )
                pressure = rate["mechanics"]["node_pressures_pa"][name]
            else:
                pressure = rate["gas"]["pressures_pa"][name]
                temp = rate["gas"]["temperatures_c"][name]
                boundary = rate["gas"]["boundary_temperatures_c"].get(name, temp)
            for label, value in (
                ("pressure_bar", (pressure - ATMOSPHERIC_PRESSURE_PA) / 1e5),
                ("temperature_c", temp),
                ("mass_kg", node["mass_kg"]),
                ("internal_energy_j", node["internal_energy_j"]),
                ("volume_m3", node["volume_m3"]),
            ):
                result[f"truth_{name}_{label}"] = float(value)
            if "boundary_energy_j" in node:
                result[f"truth_{name}_boundary_temperature_c"] = boundary
                result[f"truth_{name}_boundary_energy_j"] = node["boundary_energy_j"]
    for kind in ("pressure_bar", "temperature_c"):
        result[f"truth_low_pressure_gas_{kind}"] = result[
            f"truth_low_pressure_reservoir_gas_{kind}"
        ]
    for name in PISTONS:
        result[f"truth_{name}_piston_position_m"] = s["position"][name]
        result[f"truth_{name}_piston_velocity_m_s"] = s["velocity"][name]
        result[f"truth_{name}_drive_force_n"] = (
            rate["mechanics"]["net_drive_n"][name]
            + rate["mechanics"]["friction_force_n"][name]
            - rate["end_stops"][name]["force_n"]
        )
    for name, response in rate["end_stops"].items():
        result[f"truth_{name}_end_stop_force_n"] = response["force_n"]
        result[f"truth_{name}_lower_stop_deflection_m"] = response["lower_deflection_m"]
        result[f"truth_{name}_upper_stop_deflection_m"] = response["upper_deflection_m"]
        result[f"truth_{name}_end_stop_energy_j"] = response["energy_j"]
    for name in ("actuator_a", "actuator_b"):
        force = u[name + "_force_n"]
        result[f"truth_{name}_force_n"] = force
        result[f"truth_{name}_displacement_mm"] = 1000 * shaft_position_m(
            s["position"][name], force, system.g["actuator_input_shaft_stiffness_n_m"]
        )
    result["truth_blowoff_displacement_mm"] = 1000 * s["position"]["blowoff"]
    for i, q in enumerate(rate["flows"]["pipe"]):
        result[f"truth_hydraulic_pipe_flow_{i}_m3_s"] = q
    for group in ("liquid", "gas"):
        for a, b, q in rate[group]["mass_flows_kg_s"]:
            result[f"truth_mass_flow_{a}_to_{b}_kg_s"] = q
    solid_energy = 0.0
    for name, value in s["solid"].items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, temp in enumerate(value):
                result[f"truth_{name}_{i}_temperature_c"] = temp
                solid_energy += (
                    temp * rate["thermal"]["solid_heat_capacity_j_k"][f"{name}_{i}"]
                )
        else:
            result[f"truth_{name}_temperature_c"] = value
            solid_energy += value * rate["thermal"]["solid_heat_capacity_j_k"][name]
    piston_energy = sum(
        0.5 * effective_piston_mass_kg(p, n) * s["velocity"][n] ** 2 for n in PISTONS
    )
    length = system.g["hydraulic_pipe_length_m"] / system.cells
    pipe_energy = sum(
        0.5
        * pipe_inertance(
            p["fluid"]["density_kg_m3_at_20c"],
            length * (0.5 if i in (0, system.cells) else 1),
            system.g["hydraulic_pipe_diameter_m"],
        )
        * q
        * q
        for i, q in enumerate(rate["flows"]["pipe"])
    )
    pipe_energy += (
        0.5
        * pipe_inertance(
            p["fluid"]["density_kg_m3_at_20c"],
            system.g["blowoff_connection_length_m"],
            system.g["blowoff_connection_diameter_m"],
        )
        * rate["flows"]["blowoff"] ** 2
    )
    result["truth_liquid_mass_total_kg"] = sum(
        n["mass_kg"] for n in s["liquid"].values()
    )
    result["truth_nitrogen_mass_total_kg"] = sum(
        n["mass_kg"] for n in s["gas"].values()
    )
    result["truth_system_energy_j"] = (
        solid_energy
        + piston_energy
        + pipe_energy
        + sum(stop["energy_j"] for stop in rate["end_stops"].values())
        + sum(
            n["internal_energy_j"]
            for group in ("liquid", "gas")
            for n in s[group].values()
        )
    )
    result["truth_input_shaft_strain_energy_j"] = sum(
        u[n + "_force_n"] ** 2 / (2 * system.g["actuator_input_shaft_stiffness_n_m"])
        for n in ("actuator_a", "actuator_b")
    )
    from .hydraulics import piston_area

    ag = piston_area(system.g["actuator_piston_diameter_m"])
    bg = piston_area(system.g["blowoff_piston_diameter_m"])
    # Total external volume includes the temperature-induced liquid expansion.
    volume_rate = (ag + system.j["actuator_a"]) * (
        s["velocity"]["actuator_a"] + s["velocity"]["actuator_b"]
    ) + (system.j["blowoff"] - bg) * s["velocity"]["blowoff"]
    result["truth_external_power_w"] = (
        rate["thermal"]["external_heat_w"]
        - sum(
            u[n + "_force_n"] * s["velocity"][n] for n in ("actuator_a", "actuator_b")
        )
        - ATMOSPHERIC_PRESSURE_PA * volume_rate
    )
    result["truth_thermal_expansion_work_w"] = sum(
        rate["mechanics"]["node_pressures_pa"][n] * q
        for n, q in rate["sources"].items()
    )
    result["truth_thermal_expansion_m3"] = system.expansion(s["liquid"])
    result["truth_liquid_volume_constraint_residual_m3"] = sum(
        system.j[n] * s["position"][n] for n in PISTONS
    ) - system.expansion(s["liquid"])
    result["truth_liquid_continuity_residual_m3_s"] = rate["flows"][
        "continuity_residual_m3_s"
    ]
    result["truth_minimum_liquid_absolute_pressure_pa"] = min(
        rate["mechanics"]["node_pressures_pa"].values()
    )
    return result


def simulate(
    physical,
    run,
    inputs,
    event_callback=None,
    progress_callback=None,
    diagnostics_callback=None,
):
    """Integrate incompressible flow, gas, smooth seal friction and thermal states."""
    opts = run.get("solver", {})
    cells, wall_cells = opts.get("pipe_cells", 3), opts.get("actuator_wall_cells", 6)
    rtol, max_step = float(opts.get("relative_tolerance", 1e-5)), float(
        opts.get("max_step_s", 0.05)
    )
    interval = float(opts.get("progress_interval_s", 0.5))
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (cells, wall_cells, opts.get("max_events", 10000))
        )
        or not 0 < rtol < 1
        or not np.isfinite([max_step, interval]).all()
        or min(max_step, interval) <= 0
    ):
        raise ValueError("invalid numerical solver settings")
    if opts.get("friction_model", "smooth") != "smooth":
        raise ValueError("only smooth seal friction is supported")
    times = inputs.time_s.to_numpy(dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("input time must increase")
    first = interpolate_inputs(inputs, times[0])
    initial = initialize_system(
        physical,
        run,
        first["actuator_a_force_n"],
        first["actuator_b_force_n"],
        time_s=times[0],
        pipe_cells=cells,
        actuator_wall_cells=wall_cells,
    )
    liquid, gas = circuit_nodes_from_initial_state(initial)
    s = {
        "position": initial["position_m"],
        "velocity": initial["velocity_m_s"],
        "liquid": liquid,
        "gas": gas,
        "solid": initial["solid_temperatures_c"],
    }
    system = IncompressibleSystem(
        physical, run, cells, liquid, initial["liquid_pressure_pa"]
    )
    dep = "main_accumulator"
    layout = _Layout(s, system, dep)
    now = float(times[0])
    sample, count = 0, 0
    records = []
    checkpoint = (
        min(now + interval, float(times[-1]))
        if progress_callback or diagnostics_callback
        else float(times[-1])
    )
    window_start = now
    stats, event_counts = Counter(), Counter()
    steps = []
    last_heartbeat = time.perf_counter()

    def notify(t):
        if progress_callback:
            progress_callback(t, count)

    def diagnostics(t):
        nonlocal window_start
        if diagnostics_callback:
            diagnostics_callback(
                dict(
                    window_start_s=window_start,
                    time_s=t,
                    solve_calls=stats["calls"],
                    accepted_steps=stats["steps"],
                    nfev=stats["nfev"],
                    njev=stats["njev"],
                    nlu=stats["nlu"],
                    solve_wall_s=stats["wall"],
                    min_step_s=min(steps, default=0),
                    max_step_s=max(steps, default=0),
                    event_counts=dict(event_counts),
                    event_intervals_s={},
                    warnings={},
                )
            )
        stats.clear()
        event_counts.clear()
        steps.clear()
        window_start = t

    def rhs(t, y):
        nonlocal last_heartbeat
        state = layout.unpack(y)
        rate = system.evaluate(state, interpolate_inputs(inputs, t), dep)
        if time.perf_counter() - last_heartbeat > 2:
            notify(now)
            last_heartbeat = time.perf_counter()
        return layout.derivative(rate, state)

    jacobian_cache = {}

    def jacobian(t, y):
        # Near a loaded stop, velocities can be almost zero. SciPy's default
        # perturbation is then too small to resolve the pressure/friction
        # solve accurately. Use physical perturbation scales for this model.
        state = layout.unpack(y)
        contacts = tuple(
            (response["lower_deflection_m"] > 0, response["upper_deflection_m"] > 0)
            for response in system.stop_responses(state["position"]).values()
        )
        if (
            jacobian_cache
            and abs(t - jacobian_cache["time"]) < 0.05
            and contacts == jacobian_cache["contacts"]
            and jacobian_cache["uses"] < 8
        ):
            jacobian_cache["uses"] += 1
            return jacobian_cache["matrix"]
        base = rhs(t, y)
        jac = np.empty((len(y), len(y)))
        for i, (group, name, field) in enumerate(layout.fields):
            if group == "position":
                step = max(abs(y[i]) * 1e-7, 1e-8)
            elif group == "velocity":
                step = max(abs(y[i]) * 1e-7, 1e-7)
            elif group == "gas":
                if field == "internal_energy_j":
                    # Resolve small gas pressure differences without jumping
                    # across the regularized zero-flow transport interval.
                    step = max(abs(y[i]) * 1e-11, 1e-10)
                else:
                    step = max(abs(y[i]) * 1e-8, 1e-14 if field == "mass_kg" else 1e-7)
            else:
                step = max(abs(y[i]) * 1e-6, 1e-8 if field == "mass_kg" else 1e-3)
            shifted = y.copy()
            shifted[i] += step
            jac[:, i] = (rhs(t, shifted) - base) / step
        jacobian_cache.update(time=t, contacts=contacts, uses=0, matrix=jac)
        return jac

    # Contacts are continuous spring forces. These roots only report contact
    # and separation; they never reset position, velocity or thermal energy.
    events, metadata = [], []
    for name, (lo, hi) in system.stop_bounds.items():
        for side, bound in (("lower", lo), ("upper", hi)):
            if bound is None:
                continue
            surface = bound + (
                system.lower_preloads[name] / system.stop_stiffness
                if side == "lower"
                else 0.0
            )

            def contact(t, y, label=name, limit=surface):
                return layout.unpack(y)["position"][label] - limit

            contact.terminal = False
            contact.direction = 0
            events.append(contact)
            metadata.append((name, side))
    notify(now)
    state_y = layout.y0
    while now < times[-1] - 1e-12:
        started = time.perf_counter()
        solution = solve_ivp(
            rhs,
            (now, checkpoint),
            state_y,
            method="Radau",
            jac=jacobian,
            rtol=rtol,
            atol=layout.atol,
            max_step=max_step,
            events=events,
            dense_output=True,
        )
        stats["calls"] += 1
        stats["steps"] += len(solution.t) - 1
        stats["nfev"] += solution.nfev
        stats["njev"] += solution.njev
        stats["nlu"] += solution.nlu
        stats["wall"] += time.perf_counter() - started
        steps.extend(np.diff(solution.t).tolist())
        if not solution.success:
            raise RuntimeError(
                f"incompressible solve failed at {solution.t[-1]:g} s: {solution.message}"
            )
        end = float(solution.t[-1])
        contact_events = []
        for i, stamps in enumerate(solution.t_events):
            name, side = metadata[i]
            for stamp in stamps:
                if stamp <= now and now > times[0]:
                    continue
                state = layout.unpack(solution.sol(stamp))
                rate = system.evaluate(state, interpolate_inputs(inputs, stamp), dep)
                entering = state["velocity"][name] * (1 if side == "upper" else -1) > 0
                kind = ("contact_" if entering else "release_") + side
                contact_events.append((float(stamp), name, kind))
        for stamp, name, kind in sorted(contact_events):
            count += 1
            event_counts[(name, kind)] += 1
            if event_callback:
                event_callback(stamp, name, kind)
        if count > opts.get("max_events", 10000):
            raise RuntimeError(f"too many end-stop contact events at {end:g} s")
        while sample < len(times) and times[sample] <= end + 1e-12:
            stamp = float(times[sample])
            state = layout.unpack(solution.sol(stamp))
            u = interpolate_inputs(inputs, stamp)
            rate = system.evaluate(state, u, dep)
            records.append(_record(state, rate, stamp, system, u))
            sample += 1
        state_y = solution.y[:, -1].copy()
        now = end
        notify(now)
        diagnostics(now)
        checkpoint = min(checkpoint + interval, float(times[-1]))
    if len(records) != len(times):
        raise RuntimeError("not all requested output samples were generated")
    truth = pd.DataFrame(records)
    dt = np.diff(times)
    power = truth.truth_external_power_w.to_numpy()
    work = np.r_[0.0, np.cumsum((power[:-1] + power[1:]) * dt / 2)]
    truth["truth_sampled_energy_balance_residual_j"] = (
        truth.truth_system_energy_j - truth.truth_system_energy_j.iloc[0] - work
    )
    return add_sensor_measurements(truth, physical, run)
