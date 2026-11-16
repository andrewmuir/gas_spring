"""Compare the frozen and reduced gas formulations on identical load histories."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "examples/benchmarks/gas_validation"
ARCHIVE = ROOT / "examples/benchmarks/gas_reference_before_reduction.zip"
CASES = {
    "profile_5s": None,
    "4hz": (4.0, 4, 0.0),
    "25hz": (25.0, 10, 0.0),
    "25hz_asymmetric": (25.0, 6, np.pi / 2),
}


def generate(name, base):
    run = json.loads(json.dumps(base))
    case = CASES[name]
    if case is None:
        frame = pd.read_parquet(ROOT / "examples/inputs.parquet")
        frame = frame.loc[frame.time_s <= 5.0].reset_index(drop=True)
    else:
        frequency, cycles, phase_b = case
        ramp = 0.25
        t = np.r_[
            np.linspace(0.0, ramp, 51),
            ramp + np.arange(1, cycles * 80 + 1) / (frequency * 80),
        ]
        tau = np.maximum(t - ramp, 0.0)
        envelope = 0.5 * (1 - np.cos(np.pi * np.minimum(tau * frequency, 1.0)))
        initial = float(
            pd.read_parquet(ROOT / "examples/inputs.parquet").actuator_a_force_n.iloc[0]
        )
        ramp_force = initial + (5000.0 - initial) * t / ramp
        force_a = np.where(
            t <= ramp,
            ramp_force,
            5000.0 + 3000.0 * envelope * np.sin(2 * np.pi * frequency * tau),
        )
        force_b = np.where(
            t <= ramp,
            ramp_force,
            5000.0 + 3000.0 * envelope * np.sin(2 * np.pi * frequency * tau + phase_b),
        )
        frame = pd.DataFrame(
            dict(
                time_s=t,
                actuator_a_force_n=force_a,
                actuator_b_force_n=force_b,
                hot_ambient_c=120.0,
                duct_air_speed_m_s=2.0,
            )
        )
        run["solver"]["max_step_s"] = min(
            run["solver"]["max_step_s"], 1 / (40 * frequency)
        )
    frame.to_parquet(DIRECTORY / f"{name}_input.parquet", index=False)
    run.update(
        input_parquet=f"{name}_input.parquet", output_parquet=f"{name}_output.parquet"
    )
    path = DIRECTORY / f"{name}.yaml"
    path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
    return path


def differences(name):
    paths = [
        ROOT / f"examples/benchmarks/gas_validation_{name}_{model}.parquet"
        for model in ("reference", "reduced")
    ]
    if not all(p.exists() for p in paths):
        return None
    reference, reduced = [pd.read_parquet(p) for p in paths]
    np.testing.assert_array_equal(reference.time_s, reduced.time_s)
    result = {}
    for suffix in ("piston_position_m", "pressure_bar", "temperature_c"):
        columns = [
            c for c in reference if c.startswith("truth_") and c.endswith(suffix)
        ]
        result["max_difference_" + suffix] = float(
            (reference[columns] - reduced[columns]).abs().max().max()
        )
    hydraulic = [
        c
        for c in reference
        if c.startswith("truth_") and c.endswith("pressure_bar") and "gas" not in c
    ]
    result["max_hydraulic_pressure_difference_bar"] = float(
        (reference[hydraulic] - reduced[hydraulic]).abs().max().max()
    )
    result["max_estimated_omitted_pressure_drop_fraction"] = float(
        reduced.truth_gas_network_estimated_pressure_drop_fraction.max()
    )
    result["max_estimated_gas_mach"] = float(
        reduced.truth_gas_network_estimated_max_mach_number.max()
    )
    # A separate estimate of omitted line inertia. Agreement with the frozen
    # reference alone cannot validate acoustics, which that model also omitted.
    physical = yaml.safe_load((ROOT / "config/physical.yaml").read_text())
    geom, gas = physical["geometry"], physical["gas"]
    cells = sum(
        c.startswith("truth_gas_pipe_") and c.endswith("_volume_m3") for c in reduced
    )
    path = [
        "actuator_gas_junction",
        *[f"gas_pipe_{i}" for i in range(cells)],
        "low_pressure_reservoir_gas",
    ]
    edges = [
        (
            "actuator_a_gas",
            path[0],
            geom["actuator_gas_connection_length_m"],
            geom["actuator_gas_connection_diameter_m"],
        ),
        (
            "actuator_b_gas",
            path[0],
            geom["actuator_gas_connection_length_m"],
            geom["actuator_gas_connection_diameter_m"],
        ),
    ]
    for i, (a, b) in enumerate(zip(path, path[1:])):
        edges.append(
            (
                a,
                b,
                geom["gas_pipe_length_m"] / cells * (0.5 if i in (0, cells) else 1.0),
                geom["gas_pipe_diameter_m"],
            )
        )
    pressure = reduced.truth_low_pressure_gas_pressure_bar.to_numpy() * 1e5 + 101325.0
    offsets = {
        n: np.zeros(len(reduced)) for n in ("actuator_a_gas", "actuator_b_gas", *path)
    }
    for a, b, length, diameter in reversed(edges):
        temperature = (
            reduced[f"truth_{a}_temperature_c"] + reduced[f"truth_{b}_temperature_c"]
        ).to_numpy() / 2 + 273.15
        density = pressure / (gas["specific_gas_constant_j_kg_k"] * temperature)
        flow = reduced[f"truth_mass_flow_{a}_to_{b}_kg_s"].to_numpy() / density
        drop = (
            density
            * length
            / (np.pi * diameter**2 / 4)
            * np.gradient(flow, reduced.time_s.to_numpy(), edge_order=2)
        )
        offsets[a] = offsets[b] + drop
    inertial = np.max(np.abs(np.array(list(offsets.values()))), axis=0)
    result["max_estimated_inertial_gas_pressure_fraction"] = float(
        max(inertial / pressure)
    )
    if CASES[name]:
        frequency, cycles, _ = CASES[name]
        mask = reference.time_s >= 0.25 + (cycles / 2) / frequency
        t = reference.loc[mask, "time_s"].to_numpy()
        design = np.column_stack(
            [
                np.ones_like(t),
                t - t.mean(),
                np.sin(2 * np.pi * frequency * (t - 0.25)),
                np.cos(2 * np.pi * frequency * (t - 0.25)),
            ]
        )
        amplitudes = []
        for frame in (reference, reduced):
            c = np.linalg.lstsq(
                design,
                frame.loc[mask, "truth_actuator_a_piston_position_m"],
                rcond=None,
            )[0]
            amplitudes.append(complex(c[2], c[3]))
        result["actuator_a_fundamental_gain_change_fraction"] = (
            abs(amplitudes[1]) / abs(amplitudes[0]) - 1
        )
        result["actuator_a_fundamental_phase_change_deg"] = float(
            np.angle(amplitudes[1] / amplitudes[0], deg=True)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("reference", "reduced"),
        default=["reference", "reduced"],
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "config/run.yaml").read_text())
    journal = DIRECTORY / "runs.json"
    outcomes = json.loads(journal.read_text()) if journal.exists() else []
    for name in args.cases:
        run_path = generate(name, base)
        if args.generate_only:
            continue
        for model in args.models:
            label = f"gas_validation_{name}_{model}"
            command = [
                sys.executable,
                str(ROOT / "examples/benchmark.py"),
                "--label",
                label,
                "--run",
                str(run_path),
            ]
            if model == "reference":
                command += ["--reference-archive", str(ARCHIVE)]
            started = time.perf_counter()
            print(f"Running {name}: {model}", flush=True)
            try:
                with (DIRECTORY / f"{name}_{model}.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout_s,
                    )
                outcome = dict(
                    case=name,
                    model=model,
                    exit_code=completed.returncode,
                    external_elapsed_s=time.perf_counter() - started,
                )
            except subprocess.TimeoutExpired:
                outcome = dict(
                    case=name,
                    model=model,
                    timed_out=True,
                    external_elapsed_s=time.perf_counter() - started,
                )
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
            (DIRECTORY / "runs.json").write_text(
                json.dumps(outcomes, indent=2), encoding="utf-8"
            )
        result = differences(name)
        if result:
            (DIRECTORY / f"{name}_comparison.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
