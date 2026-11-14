"""Command-line entry point for validation and the provisional simulator."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from .config import load_yaml, load_inputs, validate_config
from .initialization import initialize_system
from .mechanics import chamber_volumes


def print_diagnostic_model_context(physical, run, data):
    """Print configured solver settings and physical parameter context."""
    geometry = physical["geometry"]
    volumes = chamber_volumes(geometry)
    gas = physical["gas"]
    gamma = float(gas["specific_heat_cp_j_kg_k"]) / (
        float(gas["specific_heat_cp_j_kg_k"]) - float(gas["specific_gas_constant_j_kg_k"])
    )
    pressure = run["initial_gas_gauge_pressure_bar"]
    masses = physical["moving_masses_kg"]
    attached_masses = physical.get("attached_masses_kg", {})
    effective_masses = {name: float(mass) + float(attached_masses.get(name, 0.0))
                        for name, mass in masses.items()}
    print("Diagnostic parameter context (zero travel):")
    solver_options = run.get("solver", {})
    print(f"  Radau settings: rtol={solver_options.get('relative_tolerance', 1e-5):g}, "
          f"max_step={solver_options.get('max_step_s', .05):g} s, "
          f"progress interval={solver_options.get('progress_interval_s', .5):g} s, "
          f"max_events={solver_options.get('max_events', 10000)}.")
    print(f"  Nitrogen gamma: {gamma:.4g}.")
    print(f"  End-stop stiffness: {physical.get('end_stops', {}).get('stiffness_n_m', 1e7):g} N/m per stop (10 kN/mm at the default).")
    print("  Piston masses (kg): " + ", ".join(f"{name}={value:g}" for name, value in masses.items()))
    print("  Attached masses (kg): " + ", ".join(
        f"{name}={float(attached_masses.get(name, 0.0)):g}" for name in masses))
    print("  Effective moving masses (kg): " + ", ".join(
        f"{name}={value:g}" for name, value in effective_masses.items()))
    print("  Zero-travel volumes (mL): "
          f"actuator fluid={1e6*volumes['actuator_a_fluid']:.3g}, "
          f"actuator gas={1e6*volumes['actuator_a_gas']:.3g}, "
          f"main fluid={1e6*volumes['main_fluid']:.3g}, "
          f"main gas={1e6*volumes['main_gas']:.3g}.")
    sample_intervals = data.time_s.diff().dropna().to_numpy()
    if len(sample_intervals):
        mean_dt = float(sample_intervals.mean())
        if max(abs(float(dt) - mean_dt) for dt in sample_intervals) <= max(1e-12, mean_dt*1e-6):
            print(f"  Uniform input spacing: {mean_dt:g} s "
                  f"(nominal Nyquist {0.5/mean_dt:.0f} Hz); Radau max step: "
                  f"{run.get('solver', {}).get('max_step_s', .05):g} s.")
    seals = physical["seal_friction"]
    print("  Kinetic/static friction ratio: "
          f"{seals['kinetic_to_static_ratio']:g}; configured seal constants: "
          + ", ".join(f"{name}={spec['constant_n']:g} N"
                       for name, spec in seals.items() if isinstance(spec, dict)))
    actuator_seal = seals["actuator"]
    main_seal = seals["main_accumulator"]
    actuator_limit = (actuator_seal["constant_n"] + actuator_seal["pressure_coefficient_n_pa"]
                      * (float(pressure["low"]) + float(pressure["main"])) * 1e5)
    main_limit = (main_seal["constant_n"] + main_seal["pressure_coefficient_n_pa"]
                  * 2 * float(pressure["main"]) * 1e5)
    print(f"  Approximate initial static seal limits: actuators={actuator_limit:.1f} N; "
          f"main accumulator={main_limit:.1f} N.")

def main():
    parser = argparse.ArgumentParser(description="Validate gas spring model inputs")
    parser.add_argument("--physical", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--initialize", action="store_true",
                        help="also check static equilibrium with the first input forces")
    parser.add_argument("--simulate", action="store_true",
                        help="run the incompressible model and write output Parquet")
    parser.add_argument("--diagnostics", action="store_true",
                        help="show per-window integration, event, and warning diagnostics")
    parser.add_argument("--thermal-estimate", action="store_true",
                        help="print the approximate isolated 180 s actuator warm-up")
    args = parser.parse_args()
    physical, run = load_yaml(args.physical), load_yaml(args.run)
    validate_config(physical, run)
    data = load_inputs(run, args.run)
    print(f"Validated {len(data)} input samples ({data.time_s.iloc[0]:g} to {data.time_s.iloc[-1]:g} s).")
    if args.thermal_estimate:
        from .diagnostics import isolated_actuator_warmup
        estimate = isolated_actuator_warmup(physical, run)
        print("Isolated 180 s thermal estimate (no pipe or flow effects):")
        for name, temperature in estimate.items():
            print(f"  {name}: {temperature:.2f} C")
    if args.initialize:
        first = data.iloc[0]
        state = initialize_system(physical, run, first.actuator_a_force_n,
                                  first.actuator_b_force_n, time_s=first.time_s)
        eq = state["equilibrium"]
        print(f"Static initial hydraulic pressure: {eq.hydraulic_gauge_pressure_pa / 1e5:.4f} bar gauge.")
        print(f"Allowed static range: {eq.allowable_pressure_interval_pa[0] / 1e5:.4f} to {eq.allowable_pressure_interval_pa[1] / 1e5:.4f} bar gauge.")
    if args.simulate:
        from .solver import simulate
        output = (Path(args.run).resolve().parent / run["output_parquet"]).resolve()
        source = (Path(args.run).resolve().parent / run["input_parquet"]).resolve()
        if output in (source, Path(args.physical).resolve(), Path(args.run).resolve()):
            raise ValueError("output Parquet must not overwrite the input Parquet or either YAML file")
        start_s, end_s = float(data.time_s.iloc[0]), float(data.time_s.iloc[-1])

        def show_progress(time_s, event_count):
            fraction = (time_s - start_s) / (end_s - start_s)
            fraction = min(max(fraction, 0.0), 1.0)
            sys.stdout.write(
                f"\rSimulation: {100*fraction:5.1f}% "
                f"({time_s-start_s:.2f}/{end_s-start_s:.2f} s; {event_count} events)"
            )
            sys.stdout.flush()

        def show_diagnostics(snapshot):
            start = snapshot["window_start_s"]
            end = snapshot["time_s"]
            print("\nSolver diagnostics "
                  f"[{start:.3f}-{end:.3f} s]: "
                  f"{snapshot['solve_calls']} Radau calls, "
                  f"{snapshot['accepted_steps']} accepted steps, "
                  f"nfev/njev/nlu={snapshot['nfev']}/{snapshot['njev']}/{snapshot['nlu']}, "
                  f"{snapshot['solve_wall_s']:.1f} s solver wall time.")
            if snapshot["accepted_steps"]:
                print(f"  Accepted step size: {snapshot['min_step_s']:.3g} to "
                      f"{snapshot['max_step_s']:.3g} s.")
            event_counts = snapshot["event_counts"]
            if event_counts:
                event_summary = []
                for piston in ("actuator_a", "actuator_b", "main_accumulator", "blowoff"):
                    counts = [f"{kind}={count}" for (name, kind), count in event_counts.items()
                              if name == piston]
                    if counts:
                        event_summary.append(f"{piston}({', '.join(counts)})")
                intervals = snapshot["event_intervals_s"]
                spacing_by_piston = [
                    f"{piston}={min(spacing):.3g} s"
                    for piston, spacing in intervals.items() if spacing
                ]
                spacing = ("; min spacing by piston: " + ", ".join(spacing_by_piston)
                           if spacing_by_piston else "")
                print(f"  Events in window: {sum(event_counts.values())}: "
                      f"{'; '.join(event_summary)}{spacing}.")
            warning_counts = snapshot["warnings"]
            for (filename, line, category, message), count in warning_counts.items():
                print(f"  {category} {filename}:{line} x{count}: {message}")

        print("Running incompressible simulation...", flush=True)
        try:
            if args.diagnostics:
                print_diagnostic_model_context(physical, run, data)
            result = simulate(
                physical, run, data, progress_callback=show_progress,
                diagnostics_callback=show_diagnostics if args.diagnostics else None,
            )
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()
        pressure_channel = "truth_minimum_liquid_absolute_pressure_pa"
        if pressure_channel in result and result[pressure_channel].min() <= 0:
            minimum = float(result[pressure_channel].min())
            print(f"Model diagnostic: the loads require liquid tension "
                  f"(minimum absolute pressure {minimum/1e5:.2f} bar). "
                  "Check force signs, initial pressures and travel stops; cavitation is not modeled.")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Write a complete temporary file before replacing the requested output.
        with tempfile.NamedTemporaryFile(prefix=".gas_spring_", suffix=".parquet", dir=output.parent, delete=False) as file:
            temp_path = Path(file.name)
        try:
            result.to_parquet(temp_path, index=False)
            os.replace(temp_path, output)
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"Wrote provisional simulation output: {output}")
    else:
        print("No simulation was run. Use --simulate to run the provisional model.")

if __name__ == "__main__":
    main()
