"""Plot verified restriction drops and pressure-sensor noise (Matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    directory = Path(__file__).resolve().parent / "benchmarks"
    result = json.loads((directory / "restriction_calibration/result.json").read_text())
    frame = pd.read_parquet(directory / f"{result['label']}.parquet")
    mask = frame.time_s >= frame.time_s.iloc[-1] - 1.0
    t = frame.loc[mask, "time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
    for branch, color, style in [
        ("actuator_a", "#0f766e", "-"),
        ("actuator_b", "#475569", "--"),
    ]:
        axes[0].plot(
            t,
            frame.loc[mask, f"truth_{branch}_restriction_pressure_drop_bar"],
            color=color,
            ls=style,
            lw=1.5,
            label=branch.replace("_", " ").title(),
        )
    for target in (-1.0, 1.0):
        axes[0].axhline(target, color="#9ca3af", ls=":", lw=1.0)
    axes[0].set(
        ylabel="Actuator restriction Δp [bar]",
        ylim=(-1.4, 1.4),
        title="Terminal restriction drops (+ junction to chamber)",
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].plot(
        t,
        frame.loc[mask, "truth_main_accumulator_restriction_pressure_drop_bar"],
        color="#7c3aed",
        lw=1.5,
    )
    for target in (-4.0, 4.0):
        axes[1].axhline(target, color="#9ca3af", ls=":", lw=1.0)
    axes[1].set(ylabel="Main inlet restriction Δp [bar]", ylim=(-5.2, 5.2))
    residual = (
        frame.sensor_actuator_a_fluid_pressure_bar
        - frame.truth_actuator_a_fluid_pressure_bar
    )
    axes[2].plot(t, residual.loc[mask], color="#2563eb", lw=0.8, marker=".", ms=2)
    axes[2].fill_between(
        t, -0.2, 0.2, color="#94a3b8", alpha=0.2, label="Configured ±1σ = ±0.2 bar"
    )
    axes[2].axhline(0.0, color="#9ca3af", lw=0.7)
    axes[2].set(
        ylabel="Actuator A sensor − truth [bar]",
        xlabel="Time [s]",
        title=f"Pressure noise: full-record residual standard deviation {residual.std(ddof=1):.3f} bar",
    )
    axes[2].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "restriction_calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
