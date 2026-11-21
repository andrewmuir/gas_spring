"""Convert lap kinematics to the two actuator force inputs for one axle."""

import argparse
from collections.abc import Mapping, Sequence
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_GRAVITY_M_S2 = 9.80665
_AXLES = ("front", "rear")
_SAMPLE_RATE_HZ = 100


def _number(value, name, minimum=0.0, maximum=np.inf):
    try:
        if isinstance(value, bool):
            raise TypeError
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum:g}, {maximum:g}]")
    return result


def _positive(settings, name):
    result = _number(settings.get(name), name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _section(settings, name):
    section = settings.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return section


def _balance(speed, anchors, name):
    if (not isinstance(anchors, (list, tuple)) or len(anchors) != 2
            or any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in anchors)):
        raise ValueError(f"{name} requires two [speed_kph, front_fraction] points")
    (speed_low, fraction_low), (speed_high, fraction_high) = [
        (_number(point[0], f"{name} anchor speed"),
         _number(point[1], f"{name} anchor fraction", maximum=1.0))
        for point in anchors
    ]
    if speed_high <= speed_low:
        raise ValueError(f"{name} anchor speeds must be increasing")
    gradient = (fraction_high - fraction_low) / (speed_high - speed_low)
    fractions = fraction_low + (speed - speed_low) * gradient
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError(f"{name} falls outside [0, 1] at a lap speed")
    return fractions


def _column(lap, name):
    if name not in lap:
        raise ValueError(f"missing lap column: {name}")
    try:
        return lap[name].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _acceleration(lap, time_s, direction, interior_limit_s, edge_limit_s):
    name = f"estimated_{direction}_acceleration_m_s2"
    values = _column(lap, name)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains an infinite value")
    missing = np.isnan(values)
    flag = f"{direction}_estimate_valid"
    if flag in lap:
        if not lap[flag].isin((True, False)).all():
            raise ValueError(f"{flag} must contain booleans")
        missing |= ~lap[flag].to_numpy(dtype=bool)
    if not missing.any():
        return values, missing

    good_times = time_s[~missing]
    if len(good_times) < 2:
        raise ValueError(f"{name} requires at least two valid samples to fill gaps")
    sample_times = time_s[missing]
    for time, next_index in zip(sample_times, np.searchsorted(good_times, sample_times)):
        if 0 < next_index < len(good_times):
            width = good_times[next_index] - good_times[next_index - 1]
            limit = interior_limit_s
            kind = "interior gap"
        else:
            width = abs(time - good_times[0 if next_index == 0 else -1])
            limit = edge_limit_s
            kind = "lap boundary gap"
        if width > limit:
            raise ValueError(f"{name}: {kind} at {time:g} s needs {width:g} s, limit is {limit:g} s")
    filled = values.copy()
    filled[missing] = np.interp(sample_times, good_times, values[~missing])
    return filled, missing


def _resample(profile, preload_duration):
    """Keep the full duration on a uniform grid and carry source flags forward."""
    source_time = profile["time_s"].to_numpy()
    end_tick = np.nextafter(source_time[-1] * _SAMPLE_RATE_HZ, -np.inf)
    last_tick = max(1, int(np.ceil(end_tick)))
    time = np.arange(last_tick + 1) / _SAMPLE_RATE_HZ
    query = np.minimum(time, source_time[-1])
    right = np.searchsorted(source_time, query)
    left = np.maximum(right - 1, 0)

    tolerance = 4 * np.spacing(source_time[-1])
    at_right = np.abs(query - source_time[right]) <= tolerance
    at_left = np.abs(query - source_time[left]) <= tolerance
    query = np.where(at_right, source_time[right], np.where(at_left, source_time[left], query))
    is_preload = query < preload_duration
    columns = {}
    for name, series in profile.items():
        values = series.to_numpy()
        if name == "time_s":
            columns[name] = time
        elif name == "lap_time_s":
            columns[name] = time - preload_duration
        elif name == "is_preload":
            columns[name] = is_preload
        elif values.dtype == bool:
            flagged = np.where(at_right, values[right],
                               np.where(at_left, values[left], values[left] | values[right]))
            columns[name] = flagged & ~is_preload
        else:
            columns[name] = np.interp(query, source_time, values)
    return pd.DataFrame(columns)


def build_model_inputs(lap: pd.DataFrame, settings: Mapping) -> pd.DataFrame:
    """Return model inputs and corner loads at 100 Hz without modifying the lap."""
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a mapping")
    if len(lap) < 2 or not lap.columns.is_unique:
        raise ValueError("lap must have at least two rows and unique column names")
    time = _column(lap, "time_s")
    speed = _column(lap, "vehicle_speed_kph")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("lap time_s must be finite and strictly increasing")
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("vehicle_speed_kph must be finite and non-negative")
    lap_time = time - time[0]
    axle = settings.get("axle")
    if axle not in _AXLES:
        raise ValueError("axle must be 'front' or 'rear'")
    ratio_setting = settings.get("motion_ratio", 1.0)
    ratio_name = "motion_ratio"
    if isinstance(ratio_setting, Mapping):
        ratio_setting = ratio_setting.get(axle)
        ratio_name = f"motion_ratio.{axle}"
    motion_ratio = _number(ratio_setting, ratio_name)
    if motion_ratio == 0:
        raise ValueError(f"{ratio_name} must be positive")
    mass = _positive(settings, "mass_kg")
    wheelbase = _positive(settings, "wheelbase_m")
    track = _positive(settings, "track_width_m")
    height = _number(settings.get("cog_height_m"), "cog_height_m")
    front_weight = _number(settings.get("front_weight_fraction"), "front_weight_fraction", maximum=1.0)
    mech_balance = _balance(speed, settings.get("mechanical_balance"), "mechanical_balance")
    aero_balance = _balance(speed, settings.get("aero_balance"), "aero_balance")
    aero_power = _positive(settings, "aero_power")
    aero_coefficient = _number(settings.get("aero_coefficient"), "aero_coefficient")

    gaps = _section(settings, "acceleration_gaps")
    interior_limit = _number(gaps.get("max_interior_s", 0.0), "acceleration_gaps.max_interior_s")
    edge_limit = _number(gaps.get("max_boundary_s", 0.0), "acceleration_gaps.max_boundary_s")
    longitudinal, longitudinal_filled = _acceleration(lap, lap_time, "longitudinal", interior_limit, edge_limit)
    lateral, lateral_filled = _acceleration(lap, lap_time, "lateral", interior_limit, edge_limit)

    speed_m_s = speed / 3.6
    downforce = aero_coefficient * speed_m_s ** aero_power
    fore_aft_transfer = mass * longitudinal * height / wheelbase
    lateral_transfer = mass * lateral * height / track
    weight = mass * _GRAVITY_M_S2
    front_load = weight * front_weight - fore_aft_transfer + downforce * aero_balance
    rear_load = weight * (1 - front_weight) + fore_aft_transfer + downforce * (1 - aero_balance)
    if not all(np.isfinite(values).all() for values in (downforce, lateral_transfer, front_load, rear_load)):
        raise ValueError("calculated loads are not finite")
    negative_policy = settings.get("negative_wheel_load", "error")
    if negative_policy not in ("error", "limit_transfer"):
        raise ValueError("negative_wheel_load must be 'error' or 'limit_transfer'")
    ambient = _number(settings.get("hot_ambient_c", 120.0), "hot_ambient_c", -273.15)
    if ambient == -273.15:
        raise ValueError("hot_ambient_c must be above absolute zero")

    columns = {
        "time_s": lap_time,
        "lap_time_s": lap_time,
        "vehicle_speed_kph": speed,
        "vehicle_speed_m_s": speed_m_s,
        "longitudinal_acceleration_m_s2": longitudinal,
        "lateral_acceleration_m_s2": lateral,
        "longitudinal_acceleration_filled": longitudinal_filled,
        "lateral_acceleration_filled": lateral_filled,
        "duct_air_speed_m_s": _number(settings.get("duct_air_speed_m_s", 2.0), "duct_air_speed_m_s"),
        "hot_ambient_c": ambient,
        "front_mechanical_balance": mech_balance,
        "front_aero_balance": aero_balance,
        "aero_downforce_n": downforce,
        "rearward_load_transfer_n": fore_aft_transfer,
        "rightward_load_transfer_n": lateral_transfer,
        "is_preload": False,
    }
    for name, axle_load, transfer in (
        ("front", front_load, lateral_transfer * mech_balance),
        ("rear", rear_load, lateral_transfer * (1 - mech_balance)),
    ):
        half_load = axle_load / 2
        limited = np.abs(transfer) > half_load
        infeasible = (half_load < 0) | (limited & (negative_policy == "error"))
        if infeasible.any():
            sample = np.flatnonzero(infeasible)[0]
            raise ValueError(f"calculated negative {name} wheel load at {lap_time[sample]:g} s")
        applied_transfer = np.clip(transfer, -half_load, half_load)
        columns[f"{name}_axle_load_n"] = axle_load
        columns[f"{name}_lateral_transfer_limited"] = limited
        columns[f"{name}_left_load_n"] = half_load - applied_transfer
        columns[f"{name}_right_load_n"] = half_load + applied_transfer
    for actuator, side in (("a", "left"), ("b", "right")):
        with np.errstate(over="ignore"):
            force = columns[f"{axle}_{side}_load_n"] / motion_ratio
        if not np.isfinite(force).all():
            raise ValueError(f"{ratio_name} produces nonfinite actuator forces")
        columns[f"actuator_{actuator}_force_n"] = force
    result = pd.DataFrame(columns)

    preload = _section(settings, "preload")
    duration = 0.0
    if preload:
        duration = _positive(preload, "duration_s")
        force = _number(preload.get("force_per_actuator_n"), "preload.force_per_actuator_n", -np.inf)
        result["time_s"] += duration
        start = {name: np.nan for name in result}
        for name in result.select_dtypes(include=bool):
            start[name] = False
        start.update(time_s=0.0, lap_time_s=-duration, actuator_a_force_n=force,
                     actuator_b_force_n=force, is_preload=True,
                     duct_air_speed_m_s=columns["duct_air_speed_m_s"], hot_ambient_c=ambient)
        result = pd.concat([pd.DataFrame([start]), result], ignore_index=True)
    result = _resample(result, duration)
    result.attrs["axle"] = axle
    result.attrs["motion_ratio"] = motion_ratio
    result.attrs["sample_rate_hz"] = _SAMPLE_RATE_HZ
    result.attrs["lap_duration_s"] = float(lap_time[-1])
    return result


def process_lap_files(inputs: str | Path | Sequence[str | Path], settings: Mapping,
                      output_dir: str | Path) -> list[Path]:
    """Write one model input Parquet per lap and return the paths in input order."""
    paths = [Path(inputs)] if isinstance(inputs, (str, Path)) else [Path(path) for path in inputs]
    if not paths:
        raise ValueError("at least one lap file is required")
    axle = settings.get("axle")
    if axle not in _AXLES:
        raise ValueError("axle must be 'front' or 'rear'")
    readers = {".parquet": pd.read_parquet, ".csv": pd.read_csv}
    directory = Path(output_dir)
    outputs = [directory / f"{path.stem}_{axle}_input.parquet" for path in paths]
    resolved_outputs = {path.resolve() for path in outputs}
    if len(resolved_outputs) != len(outputs):
        raise ValueError("multiple laps would write to the same output file")
    if resolved_outputs.intersection(path.resolve() for path in paths):
        raise ValueError("output would overwrite a lap file")
    directory.mkdir(parents=True, exist_ok=True)
    for source, target in zip(paths, outputs):
        reader = readers.get(source.suffix.lower())
        if reader is None:
            raise ValueError(f"{source}: expected a CSV or Parquet lap file")
        try:
            result = build_model_inputs(reader(source), settings)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{source}: {exc}") from exc
        result.attrs["source_lap"] = source.name
        result.to_parquet(target, index=False)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path, help="vehicle and conversion settings YAML")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for model input Parquets")
    parser.add_argument("inputs", nargs="+", help="lap CSV/Parquet paths or glob patterns")
    args = parser.parse_args()
    try:
        with args.settings.open(encoding="utf-8-sig") as stream:
            settings = yaml.safe_load(stream)
        if not isinstance(settings, dict):
            raise ValueError("settings YAML must contain a mapping")
        inputs = []
        for argument in args.inputs:
            matches = [argument] if Path(argument).is_file() else sorted(glob(argument))
            if not matches:
                raise ValueError(f"no lap files found for {argument}")
            inputs.extend(matches)
        outputs = process_lap_files(inputs, settings, args.output_dir)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
