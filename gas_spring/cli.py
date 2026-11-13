"""Command-line entry point for validation and the provisional simulator."""
import argparse
import os
from pathlib import Path
import tempfile
from .config import load_yaml, load_inputs, validate_config
from .initialization import initialize_system

def main():
    parser = argparse.ArgumentParser(description="Validate gas spring model inputs")
    parser.add_argument("--physical", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--initialize", action="store_true",
                        help="also check static equilibrium with the first input forces")
    parser.add_argument("--simulate", action="store_true",
                        help="run the provisional coupled model and write output Parquet")
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
        result = simulate(physical, run, data)
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
