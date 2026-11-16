"""Fit restriction gains to terminal pressure-drop targets in the full input.

Only the force ordinates of the existing digressive curves are scaled. Each
flow direction keeps its speed points and force ratio. Candidate configurations
and full-run results are retained; the production YAML is not overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "examples/benchmarks/restriction_calibration"
sys.path.insert(0, str(ROOT))
from gas_spring.config import load_yaml, load_inputs, validate_config
from gas_spring.materials import liquid_density

TARGETS = {"actuator": 1.0, "accumulator_inlet": 4.0}
DIRECTIONS = {"compression": 1.0, "extension": -1.0}


def terminal_peaks(frame, physical, window_s=1.0):
    terminal = frame.loc[frame.time_s >= float(frame.time_s.iloc[-1]) - window_s]
    result = {}
    for label, branches in [
        ("actuator", ("actuator_a", "actuator_b")),
        ("accumulator_inlet", ("main_accumulator",)),
    ]:
        junction = (
            "hydraulic_inlet_junction"
            if label == "actuator"
            else "hydraulic_outlet_junction"
        )
        result[label] = {}
        for direction, sign in DIRECTIONS.items():
            best = None
            for branch in branches:
                chamber = branch + "_fluid" if label == "actuator" else "main_fluid"
                pressure = (
                    terminal[f"truth_{junction}_pressure_bar"]
                    - terminal[f"truth_{chamber}_pressure_bar"]
                )
                index = (sign * pressure).idxmax()
                row = terminal.loc[index]
                magnitude = float(sign * pressure.loc[index])
                mass_flow = float(row[f"truth_mass_flow_{junction}_to_{chamber}_kg_s"])
                donor = junction if mass_flow >= 0 else chamber
                q = mass_flow / float(
                    liquid_density(
                        row[f"truth_{donor}_temperature_c"], physical["fluid"]
                    )
                )
                item = dict(
                    peak_bar=magnitude,
                    time_s=float(row.time_s),
                    branch=branch,
                    piston_velocity_m_s=float(
                        row[f"truth_{branch}_piston_velocity_m_s"]
                    ),
                    volume_flow_m3_s=q,
                )
                if best is None or magnitude > best["peak_bar"]:
                    best = item
            if best["peak_bar"] <= 0:
                raise ValueError(
                    f"No {direction} flow through {label} in the terminal window"
                )
            result[label][direction] = best
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", default=str(ROOT / "examples/benchmarks/gas_full19.parquet")
    )
    parser.add_argument(
        "--starting-physical", default=str(DIRECTORY / "starting_physical.yaml")
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--fast-fit",
        action="store_true",
        help="use looser trial solves, then verify at production tolerances",
    )
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    base = load_yaml(args.starting_physical)
    run = load_yaml(ROOT / "config/run.yaml")
    inputs = load_inputs(run, ROOT / "config/run.yaml")
    input_hash = hashlib.sha256(
        inputs[
            [
                "time_s",
                "actuator_a_force_n",
                "actuator_b_force_n",
                "hot_ambient_c",
                "duct_air_speed_m_s",
            ]
        ]
        .to_numpy(dtype=float)
        .tobytes()
    ).hexdigest()
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((ROOT / "gas_spring").glob("*.py")))
    ).hexdigest()
    baseline = terminal_peaks(pd.read_parquet(args.reference), base)
    keys = [(label, direction) for label in TARGETS for direction in DIRECTIONS]
    previous_gains = np.ones(len(keys))
    previous_pressure = np.array(
        [baseline[label][direction]["peak_bar"] for label, direction in keys]
    )
    target = np.array([TARGETS[label] for label, direction in keys])
    gains = target / previous_pressure
    history = []

    def run_trial(candidate, physical_path, label, loose):
        report_path = ROOT / f"examples/benchmarks/{label}.json"
        output_path = report_path.with_suffix(".parquet")
        expected_solver = dict(run["solver"])
        if loose:
            expected_solver["relative_tolerance"] = 0.001
        absolute_scale = 100.0 if loose else 1.0
        expected_run = json.loads(json.dumps(run))
        expected_run["solver"] = expected_solver
        cached = json.loads(report_path.read_text()) if report_path.exists() else None
        if not (
            cached
            and cached["physical"] == candidate
            and cached["source_sha256"] == source_hash
            and cached["solver"] == expected_solver
            and cached.get("run_configuration") == expected_run
            and cached.get("input_sha256") == input_hash
            and cached["absolute_tolerance_scale"] == absolute_scale
            and output_path.exists()
        ):
            command = [
                sys.executable,
                str(ROOT / "examples/benchmark.py"),
                "--physical",
                str(physical_path),
                "--run",
                str(ROOT / "config/run.yaml"),
                "--label",
                label,
            ]
            if loose:
                command += [
                    "--set",
                    "run.solver.relative_tolerance=0.001",
                    "--absolute-tolerance-scale",
                    "100",
                ]
            print(
                f'Running {label} ({"trial" if loose else "production tolerances"})',
                flush=True,
            )
            with (DIRECTORY / f"{label}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_s,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"Calibration solve failed; see {DIRECTORY}/{label}.log"
                )
        return pd.read_parquet(output_path)

    for iteration in range(1, args.max_iterations + 1):
        candidate = json.loads(json.dumps(base))
        for (label, direction), gain in zip(keys, gains):
            candidate["restrictions"][label][direction]["force_n"] = [
                float(force * gain)
                for force in base["restrictions"][label][direction]["force_n"]
            ]
        validate_config(candidate, run)
        physical_path = DIRECTORY / f"iteration_{iteration}.yaml"
        physical_path.write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        label = f"restriction_calibration_{iteration}"
        loose = args.fast_fit and iteration > 1
        frame = run_trial(candidate, physical_path, label, loose)
        peaks = terminal_peaks(frame, candidate)
        pressure = np.array(
            [peaks[label][direction]["peak_bar"] for label, direction in keys]
        )
        maximum_error = float(max(abs(pressure / target - 1)))
        entry = dict(
            iteration=iteration,
            gains=gains.tolist(),
            peaks=peaks,
            maximum_relative_target_error=maximum_error,
            minimum_absolute_liquid_pressure_bar=float(
                frame.truth_minimum_liquid_absolute_pressure_pa.min() / 1e5
            ),
        )
        history.append(entry)
        (DIRECTORY / "fit_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(entry), flush=True)
        if maximum_error <= args.tolerance and loose:
            label = f"restriction_calibrated_{iteration}"
            frame = run_trial(candidate, physical_path, label, False)
            peaks = terminal_peaks(frame, candidate)
            pressure = np.array(
                [peaks[label][direction]["peak_bar"] for label, direction in keys]
            )
            maximum_error = float(max(abs(pressure / target - 1)))
            entry = dict(
                iteration=iteration,
                verification=True,
                peaks=peaks,
                maximum_relative_target_error=maximum_error,
                minimum_absolute_liquid_pressure_bar=float(
                    frame.truth_minimum_liquid_absolute_pressure_pa.min() / 1e5
                ),
            )
            history.append(entry)
            (DIRECTORY / "fit_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            print(json.dumps(entry), flush=True)
        if maximum_error <= args.tolerance:
            if frame.truth_minimum_liquid_absolute_pressure_pa.min() <= 0:
                raise RuntimeError(
                    "Fitted case requires liquid tension; inspect pressures before adopting the fit."
                )
            (DIRECTORY / "final_physical.yaml").write_text(
                yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
            )
            (DIRECTORY / "result.json").write_text(
                json.dumps(
                    dict(
                        converged=True,
                        iteration=iteration,
                        label=label,
                        physical_path=str(physical_path),
                        terminal_window_s=1.0,
                        targets_bar=TARGETS,
                        maximum_relative_target_error=maximum_error,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("Restriction targets reached.", flush=True)
            return
        # Diagonal log-secant updates account for the change in velocity when
        # restrictions are opened. Limit extrapolation if cross-coupling dominates.
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.log(pressure / previous_pressure) / np.log(
                gains / previous_gains
            )
        slope = np.where(
            np.isfinite(slope) & (slope > 0.25), np.clip(slope, 0.5, 1.5), 1.0
        )
        correction = np.clip(np.exp(np.log(target / pressure) / slope), 0.25, 4.0)
        previous_pressure, previous_gains = pressure, gains.copy()
        gains = gains * correction
    raise RuntimeError(
        "Targets not yet converged; inspect fit_history.json before further fitting."
    )


if __name__ == "__main__":
    main()
