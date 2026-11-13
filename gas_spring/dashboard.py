"""Interactive Parquet viewer for gas-spring inputs and simulation results.

Launch from the project root with ``streamlit run gas_spring/dashboard.py``.
Install the optional UI dependencies with ``pip install -e '.[dashboard]'``.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_POINTS_PER_PLOT = 20_000


@st.cache_data(show_spinner=False)
def _read_path(path_text: str, modified_ns: int, size: int) -> pd.DataFrame:
    """Cache a file by its path and filesystem version."""
    del modified_ns, size  # These arguments are part of Streamlit's cache key.
    return pd.read_parquet(path_text)


@st.cache_data(show_spinner=False)
def _read_bytes(contents: bytes, filename: str) -> pd.DataFrame:
    """Cache an uploaded Parquet file by its content."""
    del filename  # Keep the name in the cache key so files remain identifiable.
    return pd.read_parquet(BytesIO(contents))


def _resolve_path(raw_path: str) -> Path | None:
    raw_path = raw_path.strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_source(label: str, default_path: str, key: str) -> tuple[pd.DataFrame | None, str]:
    """Render a file path and uploader for one Parquet source."""
    with st.expander(f"{label} Parquet source", expanded=False):
        uploaded = st.file_uploader(
            f"Upload {label.lower()} Parquet", type=["parquet", "pq"], key=f"{key}_upload"
        )
        path_text = st.text_input(
            "Or read from a local path (relative paths use the project folder)",
            value=default_path,
            key=f"{key}_path",
        )
    try:
        if uploaded is not None:
            frame = _read_bytes(uploaded.getvalue(), uploaded.name)
            source = f"upload: {uploaded.name}"
        else:
            path = _resolve_path(path_text)
            if path is None:
                return None, "No file selected"
            if not path.is_file():
                return None, f"File not found: {path}"
            stat = path.stat()
            frame = _read_path(str(path), stat.st_mtime_ns, stat.st_size)
            source = str(path)
        if "time_s" not in frame:
            return None, f"{source} has no required `time_s` column"
        times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
        if len(times) == 0 or not np.all(np.isfinite(times)):
            return None, f"{source} has an empty or invalid `time_s` column"
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            return None, f"{source} must have strictly increasing `time_s` values"
        frame = frame.copy()
        frame["time_s"] = times
        return frame, source
    except Exception as exc:  # Present common Parquet/path problems in the UI.
        return None, f"Could not read {label.lower()} Parquet: {exc}"


def _pretty(column: str) -> str:
    label = re.sub(r"^(truth|sensor)_", "", column)
    label = label.replace("_", " ")
    label = re.sub(r"\bactuator ([ab])\b", lambda m: f"Actuator {m.group(1).upper()}", label)
    return label[:1].upper() + label[1:]


def _sample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_POINTS_PER_PLOT:
        return frame
    stride = int(np.ceil(len(frame) / MAX_POINTS_PER_PLOT))
    sampled = frame.iloc[::stride]
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]])
    return sampled


def _fig(frame: pd.DataFrame, columns: list[str], title: str, y_title: str,
         multipliers: dict[str, float] | None = None) -> go.Figure:
    plotted = _sample(frame)
    fig = go.Figure()
    multipliers = multipliers or {}
    for column in columns:
        values = pd.to_numeric(plotted[column], errors="coerce")
        multiplier = multipliers.get(column, 1.0)
        fig.add_trace(go.Scatter(
            x=plotted["time_s"], y=values * multiplier,
            mode="lines", name=_pretty(column),
            connectgaps=False,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 55, "r": 20, "t": 75, "b": 50},
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def _select_and_plot(frame: pd.DataFrame, title: str, options: list[str], unit: str,
                     defaults: list[str], key: str,
                     multipliers: dict[str, float] | None = None) -> None:
    options = [column for column in options if column in frame]
    if not options:
        st.info(f"No {title.lower()} channels found in this file.")
        return
    selected = st.multiselect(
        f"{title} channels", options,
        default=[column for column in defaults if column in options],
        format_func=_pretty,
        key=key,
    )
    if selected:
        st.plotly_chart(
            _fig(frame, selected, title, unit, multipliers),
            use_container_width=True,
            key=f"chart_{key}",
        )


def _channel_groups(frame: pd.DataFrame) -> dict[str, tuple[list[str], str, dict[str, float]]]:
    columns = list(frame.columns)
    pressure = [c for c in columns if c.endswith("_pressure_bar")]
    temperature = [c for c in columns if c.endswith("_temperature_c")]
    displacement = [c for c in columns if c.endswith("_displacement_mm")]
    position = [c for c in columns if c.endswith("_position_m")]
    velocity = [c for c in columns if c.endswith("_velocity_m_s")]
    force = [c for c in columns if c.endswith("_force_n") or c.endswith("_friction_limit_n")]
    liq_flow = [c for c in columns if c.startswith("truth_mass_flow_hydraulic_") and c.endswith("_kg_s")]
    gas_flow = [c for c in columns if c.startswith("truth_mass_flow_") and c.endswith("_kg_s") and c not in liq_flow]
    hydraulic_volume = [c for c in columns if c.startswith("truth_hydraulic_pipe_flow_")]
    energy = [c for c in columns if c.endswith("_energy_j") or c.endswith("_residual_j") or c.endswith("_system_energy_j") or c.endswith("_strain_energy_j")]
    mass = [c for c in columns if c.endswith("_mass_kg")]
    power = [c for c in columns if c.endswith("_power_w")]
    return {
        "Pressure": (pressure, "Pressure (bar gauge)", {}),
        "Temperature": (temperature, "Temperature (°C)", {}),
        "Displacement": (displacement + position, "Displacement (mm)", {c: 1000.0 for c in position}),
        "Piston velocity": (velocity, "Velocity (m/s)", {}),
        "Force": (force, "Force (N)", {}),
        "Hydraulic mass flow": (liq_flow, "Mass flow (kg/s)", {}),
        "Hydraulic pipe volume flow": (hydraulic_volume, "Volume flow (m³/s)", {}),
        "Nitrogen mass flow": (gas_flow, "Mass flow (kg/s)", {}),
        "Stored energy and energy audit": (energy, "Energy (J)", {}),
        "Mass inventory": (mass, "Mass (kg)", {}),
        "External power": (power, "Power (W)", {}),
    }


def _time_filter(frame: pd.DataFrame) -> pd.DataFrame:
    low = float(frame["time_s"].iloc[0])
    high = float(frame["time_s"].iloc[-1])
    if high <= low:
        return frame
    step = max((high - low) / 10_000, 1e-9)
    selected = st.slider(
        "Displayed time range (s)", min_value=low, max_value=high,
        value=(low, high), step=step, key="output_time_range",
    )
    window = frame.loc[frame["time_s"].between(selected[0], selected[1])]
    if window.empty:
        midpoint = sum(selected) / 2
        nearest = int(np.argmin(np.abs(frame["time_s"].to_numpy() - midpoint)))
        return frame.iloc[[nearest]]
    return window


def _input_profile(frame: pd.DataFrame | None, source: str) -> None:
    st.subheader("Input profile")
    if frame is None:
        st.info(source)
        return
    st.caption(f"Loaded {source} · {len(frame):,} samples")
    numeric = [c for c in frame.columns if c != "time_s" and pd.api.types.is_numeric_dtype(frame[c])]
    buckets: dict[str, list[str]] = {"Force (N)": [], "Temperature (°C)": [], "Speed (m/s)": [], "Other channels": []}
    for column in numeric:
        lowered = column.lower()
        if "force" in lowered or lowered.endswith("_force_n"):
            buckets["Force (N)"].append(column)
        elif "temp" in lowered:
            buckets["Temperature (°C)"].append(column)
        elif "speed" in lowered or "velocity" in lowered:
            buckets["Speed (m/s)"].append(column)
        else:
            buckets["Other channels"].append(column)
    for index, (unit, options) in enumerate(buckets.items()):
        if options:
            _select_and_plot(frame, f"Input {unit.split(' (')[0]}", options, unit,
                             options[:min(4, len(options))], f"input_group_{index}")


def _sensors(frame: pd.DataFrame) -> None:
    sensor_columns = [c for c in frame if c.startswith("sensor_")]
    pairs = {c: "truth_" + c[len("sensor_"):] for c in sensor_columns
             if "truth_" + c[len("sensor_"):] in frame}
    if not pairs:
        st.info("No sensor channels with matching truth channels were found.")
        return
    selected = st.multiselect(
        "Sensor channels to compare", list(pairs),
        default=[c for c in pairs if any(k in c for k in (
            "actuator_a_fluid_pressure", "low_pressure_gas_pressure",
            "actuator_a_fluid_temperature", "actuator_a_displacement",
        ))],
        format_func=_pretty,
        key="sensor_pairs",
    )
    show_error = st.checkbox("Show sensor error (measurement − truth)", value=False)
    for sensor_column in selected:
        truth_column = pairs[sensor_column]
        unit = sensor_column.rsplit("_", 1)[-1]
        suffix_unit = {"bar": "Pressure (bar gauge)", "c": "Temperature (°C)",
                       "mm": "Displacement (mm)", "n": "Force (N)"}.get(unit, unit)
        plotted = _sample(frame)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plotted.time_s, y=plotted[truth_column],
                                 name="Model truth", mode="lines"))
        fig.add_trace(go.Scatter(x=plotted.time_s, y=plotted[sensor_column],
                                 name="Virtual sensor", mode="lines"))
        fig.update_layout(title=_pretty(sensor_column), xaxis_title="Time (s)",
                          yaxis_title=suffix_unit, hovermode="x unified",
                          margin={"l": 55, "r": 20, "t": 65, "b": 45})
        st.plotly_chart(fig, use_container_width=True, key=f"sensor_{sensor_column}")
        if show_error:
            error_frame = frame.copy()
            error_frame["_sensor_error"] = frame[sensor_column] - frame[truth_column]
            err_fig = _fig(error_frame, ["_sensor_error"],
                           f"{_pretty(sensor_column)} error", f"Error ({unit})")
            st.plotly_chart(err_fig, use_container_width=True, key=f"sensor_error_{sensor_column}")


def _overview(frame: pd.DataFrame) -> None:
    time = frame["time_s"].to_numpy(dtype=float)
    duration = time[-1] - time[0] if len(time) > 1 else 0.0
    interval = float(np.median(np.diff(time))) if len(time) > 1 else np.nan
    liquid = frame.get("truth_liquid_mass_total_kg")
    nitrogen = frame.get("truth_nitrogen_mass_total_kg")
    columns = st.columns(4)
    columns[0].metric("Duration", f"{duration:g} s")
    columns[1].metric("Samples", f"{len(frame):,}")
    columns[2].metric("Median sample interval", f"{interval:g} s" if np.isfinite(interval) else "—")
    columns[3].metric("Output channels", f"{len(frame.columns) - 1:,}")
    drift_columns = st.columns(2)
    for slot, series, title, unit in (
        (drift_columns[0], liquid, "Liquid mass range", "kg"),
        (drift_columns[1], nitrogen, "Nitrogen mass range", "kg"),
    ):
        if series is not None:
            slot.metric(title, f"{series.max() - series.min():.3g} {unit}")

    pressure_default = [
        "truth_actuator_a_fluid_pressure_bar", "truth_actuator_b_fluid_pressure_bar",
        "truth_low_pressure_gas_pressure_bar", "truth_main_gas_pressure_bar",
        "truth_blowoff_gas_pressure_bar",
    ]
    temperature_default = [
        "truth_actuator_a_fluid_temperature_c", "truth_actuator_b_fluid_temperature_c",
        "truth_actuator_a_gas_temperature_c", "truth_actuator_b_gas_temperature_c",
        "truth_main_gas_temperature_c", "truth_blowoff_gas_temperature_c",
        "hot_ambient_c", "cooler_region_temperature_c",
    ]
    motion_default = [
        "truth_actuator_a_displacement_mm", "truth_actuator_b_displacement_mm",
        "truth_blowoff_displacement_mm", "truth_main_accumulator_piston_position_m",
    ]
    groups = _channel_groups(frame)
    pressure_columns, pressure_unit, pressure_scale = groups["Pressure"]
    temperature_columns, temperature_unit, temperature_scale = groups["Temperature"]
    motion_columns, motion_unit, motion_scale = groups["Displacement"]
    _select_and_plot(frame, "Pressure", pressure_columns, pressure_unit,
                     pressure_default, "overview_pressure", pressure_scale)
    _select_and_plot(frame, "Temperature", temperature_columns, temperature_unit,
                     temperature_default, "overview_temperature", temperature_scale)
    _select_and_plot(frame, "Displacement", motion_columns, motion_unit,
                     motion_default, "overview_motion", motion_scale)


def main() -> None:
    st.set_page_config(page_title="Gas Spring Dashboard", page_icon="📈", layout="wide")
    st.title("Gas spring simulation dashboard")
    st.caption("Explore Parquet inputs, coupled model states, virtual sensors, and mass/energy audits.")

    with st.sidebar:
        st.header("Parquet files")
        input_frame, input_source = _load_source("Input", "examples/inputs.parquet", "input")
        output_frame, output_source = _load_source("Simulation output", "examples/output.parquet", "output")

    if output_frame is not None:
        st.caption(f"Simulation result: {output_source}")
        display_frame = _time_filter(output_frame)
    else:
        display_frame = None
        st.warning(f"Simulation output is not loaded: {output_source}")

    overview_tab, channels_tab, sensors_tab, input_tab, audit_tab = st.tabs(
        ["Overview", "All channels", "Sensor checks", "Input profile", "Mass & energy"]
    )

    with overview_tab:
        if display_frame is None:
            st.info("Load a simulation output Parquet to see model plots.")
        else:
            _overview(display_frame)

    with channels_tab:
        if display_frame is None:
            st.info("Load a simulation output Parquet to browse its channels.")
        else:
            groups = _channel_groups(display_frame)
            group_name = st.selectbox("Channel group", list(groups), key="channel_group")
            columns, unit, multipliers = groups[group_name]
            defaults = columns[:min(6, len(columns))]
            _select_and_plot(display_frame, group_name, columns, unit, defaults,
                             "all_channels", multipliers)
            if len(display_frame) > MAX_POINTS_PER_PLOT:
                st.caption(f"Charts are downsampled to at most {MAX_POINTS_PER_PLOT:,} points; files remain unchanged.")

    with sensors_tab:
        if display_frame is None:
            st.info("Load a simulation output Parquet to compare measurements with model truth.")
        else:
            st.write("Each virtual sensor is plotted against its corresponding internal model value.")
            _sensors(display_frame)

    with input_tab:
        _input_profile(input_frame, input_source)

    with audit_tab:
        if display_frame is None:
            st.info("Load a simulation output Parquet to see model balance channels.")
        else:
            groups = _channel_groups(display_frame)
            for group_name in ("Mass inventory", "Stored energy and energy audit", "External power"):
                columns, unit, multipliers = groups[group_name]
                defaults = [c for c in columns if any(term in c for term in (
                    "total_kg", "system_energy", "sampled_energy_balance_residual", "external_power", "shaft_strain"
                ))]
                _select_and_plot(display_frame, group_name, columns, unit, defaults,
                                 f"audit_{group_name}", multipliers)
            st.caption(
                "The energy residual is sampled at the output times and uses trapezoidal integration of external power. "
                "Sparse output can make this diagnostic appear larger."
            )


main()
