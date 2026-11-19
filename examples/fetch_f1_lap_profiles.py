"""Download eleven 2023 race lap examples and estimate horizontal accelerations.

python examples/fetch_f1_lap_profiles.py
python examples/fetch_f1_lap_profiles.py --tracks barcelona silverstone
Requires matplotlib for the exported figures. Existing raw downloads are reused.
"""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gas_spring.lap_profiles import estimate_lap_acceleration


DESTINATION = Path(__file__).resolve().parent / "lap_profiles"
# Circuit keys in the API's session records avoid ambiguity between same-country races.
TRACKS = {
    "barcelona": ("Catalunya", "Barcelona", "spanish-grand-prix"),
    "bahrain": ("Sakhir", "Bahrain / Sakhir", "bahrain-grand-prix"),
    "monaco": ("Monte Carlo", "Monaco", "monaco-grand-prix"),
    "montreal": ("Montreal", "Montreal", "canadian-grand-prix"),
    "spielberg": ("Spielberg", "Red Bull Ring", "austrian-grand-prix"),
    "silverstone": ("Silverstone", "Silverstone", "british-grand-prix"),
    "hungaroring": ("Hungaroring", "Hungaroring", "hungarian-grand-prix"),
    "spa": ("Spa-Francorchamps", "Spa-Francorchamps", "belgian-grand-prix"),
    "monza": ("Monza", "Monza", "italian-grand-prix"),
    "suzuka": ("Suzuka", "Suzuka", "japanese-grand-prix"),
    "interlagos": ("Interlagos", "Interlagos", "sao-paulo-grand-prix"),
}
LAST_REQUEST = 0.0


def fetch(endpoint, **filters):
    """Stay below the free API's burst and minute limits; respect Retry-After."""
    global LAST_REQUEST
    url = f"https://api.openf1.org/v1/{endpoint}?{urlencode(filters)}"
    for attempt in range(4):
        time.sleep(max(0.0, 2.2 - (time.monotonic() - LAST_REQUEST)))
        LAST_REQUEST = time.monotonic()
        try:
            with urlopen(url, timeout=30) as response:
                data = json.load(response)
            if not data:
                raise ValueError(f"No data: {url}")
            return data, url
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry = exc.headers.get("Retry-After", "10")
            try:
                delay = float(retry)
            except ValueError:
                delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
            # Break a long server-requested delay into bounded waits.
            remaining = max(2.2, delay)
            while remaining > 0:
                print(f"API retry in {remaining:.0f} s...", flush=True)
                pause = min(30.0, remaining)
                time.sleep(pause)
                remaining -= pause


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download_track(track, sessions, session_url, refresh=False):
    label = TRACKS[track][1]
    existing = list(DESTINATION.glob(f"{track}_2023_verstappen_lap*_raw_openf1.json"))
    if existing and not refresh:
        raw = read_json(existing[0])
        if "location_with_boundary_buffer" in raw:
            print(f"{label}: reusing {existing[0].name}", flush=True)
            return raw, existing[0]
    matched = [s for s in sessions if s["circuit_short_name"] == TRACKS[track][0] and not s.get("is_cancelled")]
    if len(matched) != 1:
        raise ValueError(f"Expected one 2023 race session for {label}")
    session = matched[0]
    laps, laps_url = fetch("laps", session_key=session["session_key"], driver_number=1)
    by_number = {lap["lap_number"]: lap for lap in laps}
    candidates = [lap for lap in laps if (
        lap.get("lap_duration") and lap["lap_number"] > 1 and lap.get("date_start")
        and not lap.get("is_pit_out_lap")
        and not by_number.get(lap["lap_number"] + 1, {}).get("is_pit_out_lap")
        and all(lap.get(f"duration_sector_{n}") is not None for n in (1, 2, 3))
        and abs(sum(lap[f"duration_sector_{n}"] for n in (1, 2, 3)) - lap["lap_duration"]) < 0.03
    )]
    if not candidates:
        raise ValueError(f"No usable timed race laps for {label}")
    lap = min(candidates, key=lambda item: item["lap_duration"])
    print(f"{label}: downloading lap {lap['lap_number']} ({lap['lap_duration']:.3f} s)", flush=True)
    start = pd.Timestamp(lap["date_start"])
    end = start + pd.Timedelta(seconds=lap["lap_duration"])
    filters = {"session_key": session["session_key"], "driver_number": 1,
               "date>": (start - pd.Timedelta(seconds=3)).isoformat(),
               "date<": (end + pd.Timedelta(seconds=3)).isoformat()}
    telemetry, telemetry_url = fetch("car_data", **filters)
    location, location_url = fetch("location", **filters)
    raw = dict(session=session, lap=lap, next_lap=by_number.get(lap["lap_number"] + 1),
               telemetry_with_boundary_buffer=telemetry, location_with_boundary_buffer=location,
               retrieved_utc=datetime.now(timezone.utc).isoformat(),
               sources=dict(sessions=session_url, laps=laps_url, telemetry=telemetry_url,
                            location=location_url))
    path = DESTINATION / f"{track}_2023_verstappen_lap{lap['lap_number']}_raw_openf1.json"
    write_json(path, raw)
    return raw, path


def _frame(records, start):
    frame = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    frame["time_s"] = (pd.to_datetime(frame.date, format="ISO8601") - start).dt.total_seconds()
    if np.any(np.diff(frame.time_s) <= 0):
        raise ValueError("Source timestamps are duplicated or non-increasing")
    return frame


def process_track(track, raw, raw_path):
    lap = raw["lap"]
    start, duration = pd.Timestamp(lap["date_start"]), float(lap["lap_duration"])
    speed = _frame(raw["telemetry_with_boundary_buffer"], start)
    position = _frame(raw["location_with_boundary_buffer"], start)
    if speed.time_s.iloc[0] > 0 or speed.time_s.iloc[-1] < duration:
        raise ValueError("Source speed samples must bracket both lap endpoints")
    native = speed.loc[speed.time_s.between(0, duration)]
    stamps = np.unique(np.r_[0.0, native.time_s, duration])
    values = np.interp(stamps, speed.time_s, speed.speed)
    distance = np.r_[0.0, np.cumsum((values[:-1] + values[1:]) / 7.2 * np.diff(stamps))]
    acceleration = estimate_lap_acceleration(speed.time_s, speed.speed, position.time_s,
                                            position[["x", "y"]], stamps)
    profile = pd.DataFrame(dict(time_s=stamps, vehicle_speed_kph=values,
                                distance_from_speed_m=distance,
                                interpolated_boundary=~np.isin(stamps, native.time_s)))
    profile = profile.join(acceleration.drop(columns=[
        "time_s", "estimated_heading_rate_rad_s", "speed_fit_span_s", "position_fit_span_s",
    ]))
    base = raw_path.with_name(raw_path.name.removesuffix("_raw_openf1.json"))
    old = read_json(base.with_suffix(".json")) if base.with_suffix(".json").exists() else {}
    sources = {**old.get("sources", {}), **raw.get("sources", {}),
               "documentation": "https://openf1.org/docs/",
               "location_documentation": "https://openf1.org/docs/#location",
               "official_timing": "https://www.fia.com/events/fia-formula-one-world-championship/"
                                  f"season-2023/{TRACKS[track][2]}/race-classification"}
    if "location_source_url" in raw:
        sources["location"] = raw["location_source_url"]
    lateral = profile.estimated_lateral_acceleration_g
    longitudinal = profile.estimated_longitudinal_acceleration_g
    if lateral.notna().mean() < 0.75 or longitudinal.notna().mean() < 0.90:
        raise ValueError("Too little reliable coverage for an acceleration example")
    metadata = {**old, **dict(
        kinematics_schema_version=1, track=track, circuit=TRACKS[track][1], year=2023,
        session="Race", driver="Max Verstappen", team="Red Bull Racing", driver_number=1,
        lap_number=lap["lap_number"], session_key=raw["session"]["session_key"],
        lap_duration_s=duration, native_sample_count=len(native),
        native_position_sample_count=int(position.time_s.between(0, duration).sum()),
        boundary_sample_count=int(profile.interpolated_boundary.sum()),
        minimum_speed_kph=float(values.min()), maximum_speed_kph=float(values.max()),
        native_interval_max_s=float(np.diff(native.time_s).max()),
        observed_samples_per_second=len(native) / duration,
        integrated_distance_m=float(distance[-1]),
        speed_unit="km/h", acceleration_units=["m/s^2", "g (9.80665 m/s^2)"],
        start_utc=start.isoformat(), sources=sources,
        selection="Fastest returned Verstappen race lap with complete consistent sector times, excluding first lap and pit entry/exit laps; not necessarily the fastest lap of the entire race.",
        acceleration_source="Derived kinematic estimates, not measured accelerometer channels",
        longitudinal_method="Local cubic derivative of speed/3.6; positive accelerating, negative braking",
        lateral_method="v*(x_dot*y_ddot-y_dot*x_ddot)/(x_dot^2+y_dot^2); positive left in the published XY plane; ignores banking, gradient and sideslip",
        smoothing=dict(method="Gaussian-weighted local cubic on original irregular timestamps",
                       speed_window_s=1.5, position_window_s=2.0, minimum_samples=7,
                       actual_speed_fit_span_range_s=[float(acceleration.speed_fit_span_s.min()),
                                                      float(acceleration.speed_fit_span_s.max())],
                       actual_position_fit_span_range_s=[float(acceleration.position_fit_span_s.min()),
                                                         float(acceleration.position_fit_span_s.max())],
                       max_source_gap_s=0.75, gap_policy="Supports with larger gaps produce NaN acceleration and a false validity flag"),
        lateral_valid_fraction=float(lateral.notna().mean()),
        longitudinal_valid_fraction=float(longitudinal.notna().mean()),
        min_longitudinal_g=float(longitudinal.min()), max_longitudinal_g=float(longitudinal.max()),
        min_lateral_g=float(lateral.min()), max_lateral_g=float(lateral.max()),
        max_position_gap_s=float(np.diff(position.time_s).max()),
        distance_definition="Cumulative trapezoidal integral of speed/3.6, not surveyed track distance",
        time_alignment="OpenF1 lap date_start is approximate; elapsed time runs from that start to the reported lap duration",
        processing="Native in-lap speeds retained; only speed endpoints interpolated. Acceleration fits use buffered samples from neighboring laps. No time/distance rescaling or acceleration clipping.",
        processed_utc=datetime.now(timezone.utc).isoformat(),
    )}
    profile.attrs["speed_profile"] = metadata
    profile.to_csv(base.with_suffix(".csv"), index=False, float_format="%.9f")
    profile.to_parquet(base.with_suffix(".parquet"), index=False)
    write_json(base.with_suffix(".json"), metadata)
    plot_profile(profile, metadata, lap, base.with_name(base.name + "_kinematics.png"))
    print(f"{TRACKS[track][1]}: saved; longitudinal {longitudinal.min():.2f} to {longitudinal.max():.2f} g, "
          f"lateral {lateral.min():.2f} to {lateral.max():.2f} g; lateral coverage {lateral.notna().mean():.1%}", flush=True)
    return metadata


def plot_profile(profile, metadata, lap, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(profile.time_s, profile.vehicle_speed_kph, color="#1465af", linewidth=1.5)
    axes[0].set_ylabel("Speed (km/h)")
    axes[1].plot(profile.time_s, profile.estimated_longitudinal_acceleration_g, color="#bc4b26", linewidth=1.4)
    axes[1].set_ylabel("Estimated longitudinal (g)")
    axes[2].plot(profile.time_s, profile.estimated_lateral_acceleration_g, color="#168072", linewidth=1.4)
    axes[2].set_ylabel("Estimated lateral (g)\n+ left / - right in XY")
    axes[2].set_xlabel("Elapsed lap time (s)")
    edges = [lap["duration_sector_1"], lap["duration_sector_1"] + lap["duration_sector_2"]]
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, metadata["lap_duration_s"])
        for boundary in edges:
            ax.axvline(boundary, color="#8995a3", linestyle="--", linewidth=0.7)
    for ax in axes[1:]:
        ax.axhline(0, color="#8995a3", linewidth=0.8)
    minutes, seconds = divmod(metadata["lap_duration_s"], 60)
    fig.suptitle(f"{metadata['circuit']} 2023 - Max Verstappen, race lap {metadata['lap_number']} "
                 f"({int(minutes)}:{seconds:06.3f})", fontsize=14, fontweight="bold")
    fig.supxlabel("Speed: OpenF1. Accelerations: smoothed kinematic estimates from speed and approximate XY position.\n"
                  "Missing segments indicate source gaps; these are not onboard accelerometer data.", fontsize=8, color="#5b6570")
    save_figure(fig, path, dpi=160)
    plt.close(fig)


def save_figure(fig, path, dpi):
    # A complete replacement also works when a viewer has the previous PNG open.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lap_plot_", suffix=".png", delete=False) as file:
        temporary = Path(file.name)
    try:
        fig.savefig(temporary, dpi=dpi)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_index():
    summaries = []
    for path in DESTINATION.glob("*_2023_verstappen_lap*.json"):
        if path.name.endswith("_raw_openf1.json"):
            continue
        data = read_json(path)
        if "kinematics_schema_version" not in data:
            continue
        row = {key: data[key] for key in (
            "track", "circuit", "lap_number", "lap_duration_s", "minimum_speed_kph", "maximum_speed_kph",
            "min_longitudinal_g", "max_longitudinal_g", "min_lateral_g", "max_lateral_g",
            "longitudinal_valid_fraction", "lateral_valid_fraction", "native_sample_count")}
        row.update(csv=path.with_suffix(".csv").name, parquet=path.with_suffix(".parquet").name,
                   plot=path.stem + "_kinematics.png", provenance=path.name)
        summaries.append(row)
    summaries.sort(key=lambda row: list(TRACKS).index(row["track"]))
    write_json(DESTINATION / "index.json", summaries)
    pd.DataFrame(summaries).to_csv(DESTINATION / "index.csv", index=False, float_format="%.6f")
    catalog = ["# Lap profile catalog", "",
               "2023 Verstappen race laps. Speed is recorded telemetry; acceleration is estimated.", "",
               "[Methods and limitations](METHOD.md) · [All files as ZIP](f1_lap_profiles_2023.zip)", "",
               "| Circuit | Lap | Lap time | Speed (km/h) | Lateral coverage | Data | Plot |",
               "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for item in summaries:
        minutes, seconds = divmod(item["lap_duration_s"], 60)
        catalog.append(
            f"| {item['circuit']} | {item['lap_number']} | {int(minutes)}:{seconds:06.3f} | "
            f"{item['minimum_speed_kph']:.0f}–{item['maximum_speed_kph']:.0f} | "
            f"{100*item['lateral_valid_fraction']:.1f}% | "
            f"[CSV]({item['csv']}) / [Parquet]({item['parquet']}) | [View]({item['plot']}) |"
        )
    (DESTINATION / "CATALOG.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(4, 3, figsize=(15, 10), constrained_layout=True)
    for ax, item in zip(axes.flat, summaries):
        profile = pd.read_parquet(DESTINATION / item["parquet"])
        ax.plot(profile.time_s, profile.vehicle_speed_kph, color="#1465af", linewidth=1.1)
        ax.set_title(f"{item['circuit']} - lap {item['lap_number']} ({item['lap_duration_s']:.3f} s)", fontsize=10)
        ax.set(xlabel="Time (s)", ylabel="km/h")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(summaries):]:
        ax.set_visible(False)
    fig.suptitle("2023 race lap examples - Max Verstappen", fontweight="bold", fontsize=16)
    save_figure(fig, DESTINATION / "collection_overview.png", dpi=150)
    plt.close(fig)
    with tempfile.NamedTemporaryFile(dir=DESTINATION, prefix=".lap_archive_", suffix=".zip", delete=False) as file:
        temporary = Path(file.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for path in sorted(DESTINATION.iterdir()):
                if path.suffix in (".csv", ".parquet", ".json", ".md", ".png"):
                    archive.write(path, arcname=path.name)
        os.replace(temporary, DESTINATION / "f1_lap_profiles_2023.zip")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=list(TRACKS), default=list(TRACKS))
    parser.add_argument("--refresh", action="store_true", help="download raw data again")
    args = parser.parse_args(argv)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    # Cache this small session listing alongside the original per-lap source bundles.
    sessions_path = DESTINATION / "sessions_2023_race.json"
    session_url = "https://api.openf1.org/v1/sessions?year=2023&session_name=Race"
    if sessions_path.exists() and not args.refresh:
        sessions = read_json(sessions_path)
    else:
        sessions, session_url = fetch("sessions", year=2023, session_name="Race")
        write_json(sessions_path, sessions)
    failures = []
    for track in args.tracks:
        try:
            raw, raw_path = download_track(track, sessions, session_url, args.refresh)
            process_track(track, raw, raw_path)
        except (HTTPError, ValueError, TimeoutError) as exc:
            failures.append(dict(track=track, error=str(exc)))
            print(f"{track}: unavailable: {exc}", flush=True)
    write_json(DESTINATION / "download_status.json", dict(requested=args.tracks, failures=failures))
    update_index()
    print(f"Collection index: {DESTINATION / 'index.csv'}; failures: {len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
