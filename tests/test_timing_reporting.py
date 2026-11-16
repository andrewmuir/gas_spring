"""Timing must include work outside individual integration windows."""

import sys

import pandas as pd
import pytest
import yaml

from gas_spring import cli
from gas_spring.config import load_yaml


@pytest.mark.parametrize("diagnostics", [False, True])
def test_cli_reports_elapsed_loading_simulation_and_write_time(
    tmp_path, monkeypatch, capsys, diagnostics
):
    physical = load_yaml("config/physical.yaml")
    run = load_yaml("config/run_5s.yaml")
    run.update(input_parquet="input.parquet", output_parquet="output.parquet")
    for name, content in (("physical", physical), ("run", run)):
        (tmp_path / f"{name}.yaml").write_text(
            yaml.safe_dump(content), encoding="utf-8"
        )
    pd.DataFrame(
        dict(
            time_s=[0.0, 0.001],
            actuator_a_force_n=1729.0,
            actuator_b_force_n=1729.0,
            duct_air_speed_m_s=2.0,
        )
    ).to_parquet(tmp_path / "input.parquet", index=False)
    clock = dict(wall=1000.0, cpu=0.0)
    monkeypatch.setattr(cli, "perf_counter", lambda: clock["wall"])
    monkeypatch.setattr(cli, "process_time", lambda: clock["cpu"])
    import gas_spring.config as config

    original_load = config.load_inputs

    def load(*args):
        clock["wall"] += 7.0
        clock["cpu"] += 2.0
        return original_load(*args)

    def simulate(physical, run, inputs, **callbacks):
        clock["wall"] += 30.0
        clock["cpu"] += 25.0
        callbacks["progress_callback"](0.001, 0)
        for start, end, integration_time in ((0.0, 0.0005, 8.0), (0.0005, 0.001, 12.0)):
            callbacks["diagnostics_callback"](
                dict(
                    window_start_s=start,
                    time_s=end,
                    solve_calls=1,
                    accepted_steps=1,
                    nfev=1,
                    njev=1,
                    nlu=1,
                    rhs_calls=1,
                    jacobian_rhs_calls=0,
                    solve_wall_s=integration_time,
                    min_step_s=0.001,
                    max_step_s=0.001,
                    event_counts={},
                    warnings={},
                )
            )
        return pd.DataFrame(dict(time_s=inputs.time_s, truth_example=[3.0, 4.0]))

    original_write = pd.DataFrame.to_parquet

    def write(frame, *args, **kwargs):
        clock["wall"] += 11.0
        clock["cpu"] += 1.0
        return original_write(frame, *args, **kwargs)

    monkeypatch.setattr(config, "load_inputs", load)
    monkeypatch.setattr("gas_spring.solver.simulate", simulate)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", write)
    arguments = [
        "gas-spring",
        "--physical",
        str(tmp_path / "physical.yaml"),
        "--run",
        str(tmp_path / "run.yaml"),
        "--simulate",
    ]
    if diagnostics:
        arguments.append("--diagnostics")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    output = capsys.readouterr().out
    assert "elapsed 37.0 s" in output
    assert "Imports, configuration and preparation: 7.00 s" in output
    assert "Integration, summed over ALL windows: 20.00 s" in output
    assert "Simulation setup, sampling, sensors and progress: 10.00 s" in output
    assert "Output writing: 11.00 s" in output
    assert "Process CPU time: 28.00 s" in output
    assert "Total elapsed wall time (loading through saved output): 48.00 s" in output
    if diagnostics:
        assert "12.0 s integration wall time THIS WINDOW" in output
        assert "Cumulative integration: 20.0 s" in output
