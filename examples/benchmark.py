"""Time the short input without modifying the full-run input or output."""

import time

COMMAND_STARTED = time.perf_counter()
COMMAND_CPU_STARTED = time.process_time()

import argparse
import cProfile
import hashlib
import json
import platform
from pathlib import Path
import pstats
import re
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gas_spring.config import load_inputs, load_yaml, validate_config
from gas_spring.solver import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(ROOT / "config/run_5s.yaml"))
    parser.add_argument("--physical", default=str(ROOT / "config/physical.yaml"))
    parser.add_argument("--label", default="timing")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--auto-jacobian", action="store_true", help="test SciPy numerical Jacobian"
    )
    parser.add_argument("--method", choices=("Radau", "BDF"), default="BDF")
    parser.add_argument("--absolute-tolerance-scale", type=float, default=1.0)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SCOPE.KEY=VALUE",
        help="temporary physical.KEY or run.KEY override (JSON value)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain only letters, digits, underscores and hyphens")
    directory = ROOT / "examples/benchmarks"
    directory.mkdir(exist_ok=True)
    physical, run = load_yaml(args.physical), load_yaml(args.run)
    for override in args.set:
        key, value = override.split("=", 1)
        scope, *path = key.split(".")
        target = {"physical": physical, "run": run}[scope]
        for part in path[:-1]:
            target = target[part]
        if path[-1] not in target:
            parser.error(f"unknown override: {key}")
        target[path[-1]] = json.loads(value)
    validate_config(physical, run)
    inputs = load_inputs(run, args.run)
    if (
        not np.isfinite(args.absolute_tolerance_scale)
        or args.absolute_tolerance_scale <= 0
    ):
        parser.error("absolute tolerance scale must be positive")
    import gas_spring.solver as solver_module

    original_solve = solver_module.solve_ivp

    def configured_solve(*positional, **keywords):
        if args.auto_jacobian:
            keywords.pop("jac", None)
        keywords["method"] = args.method
        keywords["atol"] = keywords["atol"] * args.absolute_tolerance_scale
        return original_solve(*positional, **keywords)

    solver_module.solve_ivp = configured_solve
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((ROOT / "gas_spring").glob("*.py")))
    ).hexdigest()
    snapshots = []

    def diagnostics(snapshot):
        snapshots.append(snapshot)
        print(
            f"{snapshot['time_s']:.2f}/{inputs.time_s.iloc[-1]:g} s: "
            f"{snapshot['accepted_steps']} steps, {snapshot['rhs_calls']} total RHS calls, "
            f"{snapshot['njev']} Jacobians, {snapshot['solve_wall_s']:.2f} s integration this window; "
            f"total elapsed {time.perf_counter()-COMMAND_STARTED:.2f} s",
            flush=True,
        )

    profiler = cProfile.Profile() if args.profile else None
    wall, cpu = time.perf_counter(), time.process_time()
    if profiler:
        profiler.enable()
    result = simulate(physical, run, inputs, diagnostics_callback=diagnostics)
    if profiler:
        profiler.disable()
    timings = dict(
        wall_time_s=time.perf_counter() - wall,
        process_cpu_time_s=time.process_time() - cpu,
    )
    writing_started = time.perf_counter()
    result.to_parquet(directory / f"{args.label}.parquet", index=False)
    writing_wall_s = time.perf_counter() - writing_started
    report_started = time.perf_counter()
    report = dict(
        **timings,
        wall_time_scope="simulate() only; excludes imports, input loading and result writing",
        simulation_wall_time_s=timings["wall_time_s"],
        preparation_wall_time_s=wall - COMMAND_STARTED,
        output_write_wall_time_s=writing_wall_s,
        profiled=args.profile,
        auto_jacobian=args.auto_jacobian,
        method=args.method,
        overrides=args.set,
        absolute_tolerance_scale=args.absolute_tolerance_scale,
        samples=len(result),
        simulated_time_s=float(inputs.time_s.iloc[-1] - inputs.time_s.iloc[0]),
        solver=run["solver"],
        physical=physical,
        solver_totals={
            k: sum(s[k] for s in snapshots)
            for k in (
                "solve_calls",
                "accepted_steps",
                "nfev",
                "rhs_calls",
                "jacobian_rhs_calls",
                "njev",
                "nlu",
                "solve_wall_s",
            )
        },
        windows=[
            {
                k: s[k]
                for k in (
                    "window_start_s",
                    "time_s",
                    "accepted_steps",
                    "nfev",
                    "njev",
                    "nlu",
                    "solve_wall_s",
                    "cumulative_solve_wall_s",
                    "simulation_elapsed_wall_s",
                    "min_step_s",
                    "max_step_s",
                )
            }
            for s in snapshots
        ],
        checks=dict(
            all_finite=bool(
                np.isfinite(result.select_dtypes(include=[np.number])).all().all()
            ),
            maximum_volume_residual_m3=float(
                abs(result.truth_liquid_volume_constraint_residual_m3).max()
            ),
            liquid_mass_drift_kg=float(
                result.truth_liquid_mass_total_kg.iloc[-1]
                - result.truth_liquid_mass_total_kg.iloc[0]
            ),
            nitrogen_mass_drift_kg=float(
                result.truth_nitrogen_mass_total_kg.iloc[-1]
                - result.truth_nitrogen_mass_total_kg.iloc[0]
            ),
        ),
        source_sha256=source_hash,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
    )
    if profiler:
        profiler.dump_stats(str(directory / f"{args.label}.prof"))
        with (directory / f"{args.label}_profile.txt").open(
            "w", encoding="utf-8"
        ) as file:
            pstats.Stats(profiler, stream=file).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(40)
    report["report_generation_wall_time_s"] = time.perf_counter() - report_started
    report["total_elapsed_wall_s"] = time.perf_counter() - COMMAND_STARTED
    report["total_process_cpu_time_s"] = time.process_time() - COMMAND_CPU_STARTED
    (directory / f"{args.label}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total_elapsed_wall_s": report["total_elapsed_wall_s"],
                "simulation_wall_time_s": timings["wall_time_s"],
                "integration_wall_time_s": report["solver_totals"]["solve_wall_s"],
                "total_process_cpu_time_s": report["total_process_cpu_time_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
