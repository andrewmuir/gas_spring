"""Prepare and run a YAML matrix of lap simulations."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter

import pandas as pd
import yaml

from .config import load_inputs, load_yaml, validate_config
from .hydraulics import piston_area
from .lap_input import build_model_inputs
from .run_catalog import output_run_metadata
from .sensors import CHANNELS, add_sensor_measurements


_FILE_KEYS = {"run_name", "input_parquet", "output_parquet"}
_SENSOR_KEYS = {"sensor_calibration", "seed"}
_CALIBRATION_FIELDS = {"gain", "offset", "temperature_coefficient", "reference_temperature_c"}
_CHANNEL_UNITS = {name: unit for name, _, unit, _ in CHANNELS}
_CALIBRATION_CHANNELS = {"default", *_CHANNEL_UNITS}


def _digest(*values):
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()[:16]


def _relative(path, directory):
    try:
        return os.path.relpath(path, directory)
    except ValueError:
        return str(path)


def _atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".batch_", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml(path, settings):
    document = yaml.safe_dump(settings, sort_keys=False)
    _atomic_write(path, lambda temp: temp.write_text(document, encoding="utf-8"))


def _write_parquet(path, frame):
    _atomic_write(path, lambda temp: frame.to_parquet(temp, index=False))


def _path(directory, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a non-empty path")
    return (directory / value).resolve()


def _tracks(spec, directory):
    if isinstance(spec, dict):
        entries = [(name, _path(directory, path)) for name, path in spec.items()]
    elif isinstance(spec, list):
        entries = []
        for pattern in spec:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("each track path or pattern must be a non-empty string")
            exact = _path(directory, pattern)
            matches = ([exact] if exact.is_file() else
                       sorted(Path(name).resolve() for name in glob(str(directory / pattern))))
            if not matches:
                raise ValueError(f"no lap files matched {pattern}")
            entries.extend((path.stem, path) for path in matches)
    else:
        raise ValueError("tracks must be a mapping from names to paths, or a list of paths or patterns")
    if not entries:
        raise ValueError("at least one track is required")
    names = set()
    for name, path in entries:
        if not isinstance(name, str) or not name.strip() or name.casefold() in names:
            raise ValueError(f"track names must be non-empty and unique: {name!r}")
        if not path.is_file() or path.suffix.lower() not in (".csv", ".parquet"):
            raise ValueError(f"expected a lap CSV or Parquet file: {path}")
        names.add(name.casefold())
    return entries


def _set_override(settings, key, value):
    parts = key.split(".")
    if parts[0] in _FILE_KEYS:
        raise ValueError(f"{key} is assigned by the harness")
    if parts[0] == "sensor_calibration":
        if (len(parts) != 3 or parts[1] not in _CALIBRATION_CHANNELS
                or parts[2] not in _CALIBRATION_FIELDS):
            raise ValueError(f"unknown sensor calibration setting: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        calibration = settings.setdefault("sensor_calibration", {})
        calibration.setdefault(parts[1], {})[parts[2]] = value
        return
    parent = settings
    for part in parts[:-1]:
        parent = parent.get(part) if isinstance(parent, dict) else None
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise ValueError(f"unknown run setting: {key}")
    parent[parts[-1]] = value


def _override_label(key, value):
    parts = key.split(".")
    if len(parts) == 2 and parts[0] == "initial_gas_gauge_pressure_bar":
        name = {"low": "Low side", "main": "Main", "blowoff": "Blow-off"}.get(parts[1], parts[1])
        return f"{name} precharge {value} bar"
    if parts[0] == "sensor_calibration":
        channel, field = parts[1:]
        name = channel.replace("_", " ").capitalize()
        name = re.sub(r"^Actuator ([ab]) ", lambda match: f"Actuator {match[1].upper()} ", name)
        unit = {"bar": "bar", "mm": "mm", "n": "N", "c": "°C"}.get(_CHANNEL_UNITS.get(channel), "")
        suffix = f" {unit}" if field == "offset" and unit else ""
        return f"{name} {field.replace('_', ' ')} {value}{suffix}"
    return f"{key}={value}"


def _balance_preload(lap_settings, physical, run):
    if not lap_settings.get("preload"):
        return
    geometry = physical["geometry"]
    bore = geometry["actuator_piston_diameter_m"]
    shaft = geometry["actuator_shaft_diameter_m"]
    pressure = run["initial_gas_gauge_pressure_bar"]
    lap_settings["preload"]["force_per_actuator_n"] = 1e5 * (
        float(pressure["main"]) * piston_area(bore, shaft)
        - float(pressure["low"]) * piston_area(bore)
    )


@dataclass
class Case:
    name: str
    lap: Path
    lap_settings: dict
    input_path: Path
    run: dict
    config_path: Path
    output_path: Path
    dynamics_id: str
    aero_scale: float
    overrides: dict


def plan(config_path):
    """Expand the sweep without generating inputs or running the solver."""
    directory = Path(config_path).resolve().parent
    settings = load_yaml(config_path)
    required = {"output_dir", "physical", "run_template", "lap_settings", "tracks"}
    missing = required - settings.keys()
    unknown = settings.keys() - required - {"aero_scales", "run_overrides", "balance_preload"}
    if missing or unknown:
        raise ValueError(f"batch YAML: missing settings {sorted(missing)}; unknown settings {sorted(unknown, key=str)}")
    destination = _path(directory, settings["output_dir"])
    physical = load_yaml(_path(directory, settings["physical"]))
    template = load_yaml(_path(directory, settings["run_template"]))
    base_lap = load_yaml(_path(directory, settings["lap_settings"]))
    tracks = _tracks(settings["tracks"], directory)
    scales = settings.get("aero_scales", [1.0])
    if (not isinstance(scales, list) or not scales
            or any(isinstance(scale, bool) or not isinstance(scale, (int, float))
                   or not math.isfinite(scale) or scale < 0 for scale in scales)):
        raise ValueError("aero_scales must be a non-empty list of finite, non-negative numbers")
    overrides = settings.get("run_overrides", {})
    if (not isinstance(overrides, dict)
            or any(not isinstance(key, str) or not isinstance(values, list) or not values
                   for key, values in overrides.items())):
        raise ValueError("run_overrides must map setting paths to non-empty lists of values")
    for key in overrides:
        if any(other.startswith(f"{key}.") for other in overrides):
            raise ValueError(f"cannot sweep both {key} and a setting inside it")
    balance_preload = settings.get("balance_preload", True)
    if not isinstance(balance_preload, bool):
        raise ValueError("balance_preload must be true or false")

    physical_id = _digest(physical)
    cases = {}
    for track, lap in tracks:
        lap_id = sha256(lap.read_bytes()).hexdigest()
        track_slug = re.sub(r"[^a-z0-9]+", "_", track.lower()).strip("_")[:32].rstrip("_") or "track"
        for scale, *values in product(scales, *overrides.values()):
            choices = dict(zip(overrides, values))
            run = {key: deepcopy(value) for key, value in template.items() if key not in _FILE_KEYS}
            for key, value in choices.items():
                _set_override(run, key, value)
            validate_config(physical, run)

            lap_settings = deepcopy(base_lap)
            lap_settings["aero_coefficient"] = float(base_lap["aero_coefficient"]) * scale
            if balance_preload:
                _balance_preload(lap_settings, physical, run)
            input_id = _digest(lap_id, lap_settings)
            input_path = destination / "inputs" / f"{track_slug}_{input_id}.parquet"
            dynamics = {key: value for key, value in run.items() if key not in _SENSOR_KEYS}
            dynamics_id = _digest(physical_id, input_id, dynamics)
            case_id = f"{track_slug}_{_digest(track, physical_id, input_id, run)}"
            config_file = destination / "configs" / f"{case_id}.yaml"
            output = destination / f"{case_id}.parquet"
            label = [track, lap_settings["axle"].title(),
                     "Baseline aero" if scale == 1.0 else f"{(scale - 1) * 100:+.6g}% aero"]
            label.extend(_override_label(key, value) for key, value in choices.items())
            name = " | ".join(label)
            run.update(
                run_name=name,
                input_parquet=_relative(input_path, config_file.parent),
                output_parquet=_relative(output, config_file.parent),
            )
            cases.setdefault(case_id, Case(name, lap, lap_settings, input_path, run, config_file,
                                           output, dynamics_id, scale, choices))
    return destination, physical_id, physical, list(cases.values())


def _prepare_input(case, frames, *, regenerate=False):
    if not regenerate and case.input_path.is_file():
        return
    if case.lap not in frames:
        read = pd.read_parquet if case.lap.suffix.lower() == ".parquet" else pd.read_csv
        frames[case.lap] = read(case.lap)
    frame = build_model_inputs(frames[case.lap], case.lap_settings)
    frame.attrs["source_lap"] = str(case.lap)
    frame.attrs["lap_settings"] = case.lap_settings
    _write_parquet(case.input_path, frame)


def _simulate(physical, case):
    from .solver import simulate

    frame = load_inputs(case.run, case.config_path)
    start, end = frame.time_s.iloc[[0, -1]]
    started = perf_counter()

    def progress(time_s, events):
        print(f"\r  {time_s - start:.2f}/{end - start:.2f} s, {events} stop events, "
              f"{perf_counter() - started:.1f} s elapsed", end="", flush=True)

    try:
        return simulate(physical, case.run, frame, progress_callback=progress)
    finally:
        print(flush=True)


def execute(destination, physical_id, physical, cases, *, force=False):
    """Run or resume the sweep and save a manifest after each case."""
    physical_path = destination / "configs" / f"physical_{physical_id}.yaml"
    _write_yaml(physical_path, physical)
    manifest = {
        "physical_yaml": _relative(physical_path, destination),
        "runs": [
            {"run_name": case.name, "source_lap": str(case.lap),
             "aero_scale": case.aero_scale, "overrides": case.overrides,
             "input_parquet": _relative(case.input_path, destination),
             "run_yaml": _relative(case.config_path, destination),
             "output_parquet": _relative(case.output_path, destination), "status": "pending"}
            for case in cases
        ],
    }

    def save_manifest():
        _atomic_write(destination / "manifest.json", lambda temp: temp.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"))

    save_manifest()
    frames = {}
    prepared = set()
    donors = {case.dynamics_id: case.output_path for case in cases
              if not force and case.output_path.is_file() and case.input_path.is_file()}
    for index, (case, record) in enumerate(zip(cases, manifest["runs"]), start=1):
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        started = perf_counter()
        try:
            if case.input_path not in prepared:
                _prepare_input(case, frames, regenerate=force)
                prepared.add(case.input_path)
            _write_yaml(case.config_path, case.run)
            if not force and case.output_path.is_file():
                record["status"] = "skipped"
                print("  Result already exists.", flush=True)
                continue

            donor = donors.get(case.dynamics_id)
            if donor is None:
                result = _simulate(physical, case)
            else:
                print(f"  Reusing dynamics from {donor.name}.", flush=True)
                result = add_sensor_measurements(pd.read_parquet(donor), physical, case.run)
                record["reused_dynamics_from"] = _relative(donor, destination)
            metadata = output_run_metadata(case.run, case.config_path, case.input_path, case.output_path)
            metadata.update(aero_scale=case.aero_scale, overrides=case.overrides,
                            physical_yaml=_relative(physical_path, case.output_path.parent))
            if donor is not None:
                metadata["reused_dynamics_from"] = _relative(donor, case.output_path.parent)
            result.attrs["run_metadata"] = metadata
            _write_parquet(case.output_path, result)
            donors.setdefault(case.dynamics_id, case.output_path)
            record["status"] = "complete"
            if "truth_minimum_liquid_absolute_pressure_pa" in result:
                pressure = result["truth_minimum_liquid_absolute_pressure_pa"].min()
                if pressure <= 0:
                    record["warning"] = f"Model requires liquid tension ({pressure / 1e5:.2f} bar absolute)"
                    print(f"  {record['warning']}", flush=True)
            print(f"  Saved {case.output_path.name}", flush=True)
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            raise
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"  Failed: {record['error']}", flush=True)
        finally:
            record["elapsed_s"] = perf_counter() - started
            save_manifest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="batch YAML")
    parser.add_argument("--dry-run", action="store_true", help="list cases without writing or simulating")
    parser.add_argument("--force", action="store_true", help="recompute completed cases in this batch")
    args = parser.parse_args()
    try:
        destination, physical_id, physical, cases = plan(args.config)
        print(f"{len(cases)} results, {len({case.dynamics_id for case in cases})} distinct physical solves.")
        print(f"Output directory: {destination}", flush=True)
        if args.dry_run:
            for case in cases:
                pending = not (case.output_path.is_file() and case.input_path.is_file())
                print(f"  [{'run' if pending or args.force else 'skip'}] {case.name}")
            return 0
        manifest = execute(destination, physical_id, physical, cases, force=args.force)
    except (KeyError, OSError, OverflowError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nBatch interrupted. Run again to resume.", flush=True)
        return 130
    statuses = [entry["status"] for entry in manifest["runs"]]
    print(f"{statuses.count('complete')} saved, {statuses.count('skipped')} skipped, "
          f"{statuses.count('failed')} failed. See {destination / 'manifest.json'}.", flush=True)
    return int("failed" in statuses)


if __name__ == "__main__":
    raise SystemExit(main())
