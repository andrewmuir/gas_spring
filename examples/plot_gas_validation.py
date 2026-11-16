"""Optional Matplotlib figure for the saved gas-validation artifacts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    directory = Path(__file__).resolve().parent / "benchmarks"
    reference = pd.read_parquet(directory / "gas_validation_25hz_reference.parquet")
    reduced = pd.read_parquet(directory / "gas_validation_25hz_reduced.parquet")
    np.testing.assert_array_equal(reference.time_s, reduced.time_s)
    mask = reference.time_s >= 0.45
    t = reference.loc[mask, "time_s"].to_numpy() - 0.25
    channel = "truth_actuator_a_piston_position_m"
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), layout="constrained")
    axes[0].plot(
        t,
        1000 * reference.loc[mask, channel],
        color="#475569",
        label="Previous gas formulation",
        lw=2,
    )
    axes[0].plot(
        t,
        1000 * reduced.loc[mask, channel],
        color="#0f766e",
        ls="--",
        label="Reduced gas formulation",
        lw=1.4,
    )
    axes[0].set(
        ylabel="Actuator A position [mm]", title="25 Hz forcing: last five cycles"
    )
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(
        t,
        1e6 * (reduced.loc[mask, channel] - reference.loc[mask, channel]),
        color="#0f766e",
    )
    axes[1].axhline(0, color="#94a3b8", lw=0.7)
    axes[1].set(
        xlabel="Time after cyclic load begins [s]",
        ylabel="Position difference [µm]",
        title="Reduced minus previous response",
    )
    cases = ["profile_5s", "4hz", "25hz", "25hz_asymmetric"]
    labels = [
        "5 s profile",
        "4 Hz\n1.25 s input",
        "25 Hz\n0.65 s input",
        "25 Hz, out of phase\n0.49 s input",
    ]
    cpu = {
        model: [
            json.loads((directory / f"gas_validation_{name}_{model}.json").read_text())[
                "process_cpu_time_s"
            ]
            for name in cases
        ]
        for model in ["reference", "reduced"]
    }
    x = np.arange(len(cases))
    width = 0.35
    axes[2].bar(
        x - width / 2, cpu["reference"], width, label="Previous", color="#64748b"
    )
    axes[2].bar(
        x + width / 2,
        cpu["reduced"],
        width,
        label="Reduced + block Jacobian",
        color="#0f766e",
    )
    axes[2].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Simulation CPU time [s]",
        title="Same input and numerical tolerances within each pair",
    )
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "gas_reduction_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
