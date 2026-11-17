"""Saved solver telemetry must describe integration work, not output sampling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gas_spring.config import load_yaml
from gas_spring.force_input import InputInterpolator
from gas_spring.solver import simulate


@pytest.fixture(scope="module")
def instrumented_run():
    physical = load_yaml("config/physical.yaml")
    run = load_yaml("config/run_5s.yaml")
    run["solver"].update(pipe_cells=1, actuator_wall_cells=2, progress_interval_s=0.006)
    inputs = pd.DataFrame(dict(
        time_s=[0.0, 0.003, 0.006, 0.009, 0.012],
        actuator_a_force_n=[1729.0, 1732.0, 1735.0, 1732.0, 1729.0],
        actuator_b_force_n=[1729.0, 1729.0, 1729.0, 1735.0, 1741.0],
        hot_ambient_c=120.0, duct_air_speed_m_s=2.0,
    ))
    callbacks = []
    output = simulate(physical, run, inputs, diagnostics_callback=callbacks.append)
    return inputs, output, callbacks


def test_metrics_are_internal_steps_and_match_reported_work(instrumented_run, tmp_path):
    inputs, output, callbacks = instrumented_run
    metrics = output.attrs["solve_metrics"]
    summary = metrics["summary"]
    steps = pd.DataFrame(metrics["steps"])
    windows = pd.DataFrame(metrics["windows"])
    assert len(windows) == len(callbacks) == 2
    assert len(steps) == summary["accepted_steps"] > len(inputs)
    assert np.all(steps.step_size_s > 0)
    np.testing.assert_allclose(np.diff(np.r_[inputs.time_s.iloc[0], steps.time_s]), steps.step_size_s)
    assert steps.step_size_s.sum() == pytest.approx(0.012)
    assert steps.step_size_s.min() == summary["min_step_s"]
    assert steps.step_size_s.median() == summary["median_step_s"]
    for index, window in windows.iterrows():
        assert (steps.window_index == index).sum() == window.accepted_steps
        assert steps.loc[steps.window_index == index, "time_s"].iloc[-1] == window.time_s
    for key in ("accepted_steps", "nfev", "njev", "nlu", "rhs_calls", "jacobian_rhs_calls"):
        assert summary[key] == sum(window[key] for window in callbacks)
    assert summary["rhs_calls"] == summary["nfev"] + summary["jacobian_rhs_calls"]
    assert summary["jacobian_rhs_calls"] > 0
    assert 0 < summary["rhs_wall_s"] + summary["jacobian_wall_s"] <= summary["integration_wall_s"]
    assert summary["integration_wall_s"] + summary["sampling_wall_s"] <= summary["simulation_elapsed_wall_s"]
    assert summary["integration_cpu_s"] > 0
    assert summary["simulation_cpu_s"] > 0
    assert summary["integration_wall_s"] == pytest.approx(windows.solve_wall_s.sum())
    assert summary["integration_wall_s"] == pytest.approx(callbacks[-1]["cumulative_solve_wall_s"])
    assert np.all(np.diff(windows.simulation_elapsed_wall_s) > 0)
    assert np.isfinite(output.to_numpy()).all()

    path = tmp_path / "result.parquet"
    output.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(loaded, output)
    assert loaded.attrs == output.attrs  # A single uploaded file retains telemetry.


def test_shaft_velocity_includes_the_derivative_of_elastic_stretch(instrumented_run):
    _, output, _ = instrumented_run
    # Piecewise-linear load slopes: right derivative at the 0.006 s knot,
    # left derivative at the final sample. Shaft stiffness is 20 MN/m.
    for name, expected_slopes in (
        ("a", [1000.0, 1000.0, -1000.0, -1000.0, -1000.0]),
        ("b", [0.0, 0.0, 2000.0, 2000.0, 2000.0]),
    ):
        np.testing.assert_allclose(
            output[f"truth_actuator_{name}_shaft_velocity_m_s"]
            - output[f"truth_actuator_{name}_piston_velocity_m_s"],
            np.array(expected_slopes) / 20e6, rtol=1e-10, atol=1e-14,
        )


def test_force_slopes_handle_nonzero_start_and_reject_extrapolation():
    inputs = pd.DataFrame(dict(time_s=[2.0, 3.0, 5.0], actuator_a_force_n=[0.0, 10.0, -30.0],
                               actuator_b_force_n=5.0, hot_ambient_c=120.0, duct_air_speed_m_s=2.0))
    interpolate = InputInterpolator(inputs)
    for stamp, expected in ((2.0, 10.0), (2.5, 10.0), (3.0, -20.0), (5.0, -20.0)):
        assert interpolate.force_rate_n_s("actuator_a", stamp) == expected
        assert interpolate.force_rate_n_s("actuator_b", stamp) == 0.0
    with pytest.raises(ValueError, match="outside input history"):
        interpolate.force_rate_n_s("actuator_a", 1.9)


def test_dashboard_displays_metrics_and_velocity_from_a_single_parquet(instrumented_run, tmp_path, caplog):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    pytest.importorskip("plotly")
    inputs, output, _ = instrumented_run
    input_path, output_path = tmp_path / "input.parquet", tmp_path / "output.parquet"
    inputs.to_parquet(input_path, index=False)
    output.to_parquet(output_path, index=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "gas_spring/dashboard.py"))
    app.session_state["input_path"] = str(input_path)
    app.session_state["output_path"] = str(output_path)
    app.run(timeout=20)
    assert not app.exception
    assert "Solve metrics" in [tab.label for tab in app.tabs]
    assert "Simulation elapsed" in [metric.label for metric in app.metric]
    assert app.multiselect(key="overview_velocity").value == [
        "truth_actuator_a_shaft_velocity_m_s", "truth_actuator_b_shaft_velocity_m_s",
    ]
    app.selectbox(key="channel_group").select("Piston and shaft velocity")
    app.slider(key="output_time_range").set_range(0.003, 0.006)
    app.checkbox[0].check()  # Exercise the sensor-error plot too.
    app.run(timeout=20)
    assert not app.exception
    duration = next(metric for metric in app.metric if metric.label == "Simulated duration")
    assert duration.value == "0.012 s"  # Complete-run metrics survive output filtering.
    assert "use_container_width" not in caplog.text


def test_dashboard_handles_older_results_without_saved_metrics(instrumented_run, tmp_path):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    pytest.importorskip("plotly")
    _, output, _ = instrumented_run
    legacy = output.copy()
    legacy.attrs.clear()
    legacy = legacy.drop(columns=[column for column in legacy if "_shaft_velocity_m_s" in column])
    path = tmp_path / "legacy.parquet"
    legacy.to_parquet(path, index=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "gas_spring/dashboard.py"))
    app.session_state["output_path"] = str(path)
    app.run(timeout=20)
    assert not app.exception
    assert any("predates saved solve metrics" in message.value for message in app.info)
    assert app.multiselect(key="overview_velocity").value == [
        "truth_actuator_a_piston_velocity_m_s", "truth_actuator_b_piston_velocity_m_s",
    ]
