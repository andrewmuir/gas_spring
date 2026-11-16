"""Generate the two-actuator ramp and swept-sine input profile."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gas_spring.hydraulics import piston_area

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "inputs.parquet"

RAMP_DURATION_S = 4.0
SWEEP_DURATION_S = 15.0
SAMPLE_INTERVAL_S = 0.005
TARGET_FORCE_N = 5000.0
OSCILLATION_AMPLITUDE_N = 3000.0
START_FREQUENCY_HZ = 1.0
END_FREQUENCY_HZ = 4.0


def zero_position_static_load_n(physical, run):
    """Balance the actuator at zero piston travel using configured pressures."""
    geometry = physical["geometry"]
    low_pressure_pa = run["initial_gas_gauge_pressure_bar"]["low"] * 1e5
    hydraulic_pressure_pa = run["initial_gas_gauge_pressure_bar"]["main"] * 1e5
    full_area = piston_area(geometry["actuator_piston_diameter_m"])
    fluid_area = piston_area(
        geometry["actuator_piston_diameter_m"], geometry["actuator_shaft_diameter_m"]
    )
    # Positive applied load drives extension against high-side liquid pressure.
    # Gas pressure assists extension; the accumulator sets hydraulic pressure.
    return hydraulic_pressure_pa * fluid_area - low_pressure_pa * full_area


def main():
    with (ROOT / "config" / "physical.yaml").open(encoding="utf-8") as stream:
        physical = yaml.safe_load(stream)
    with (ROOT / "config" / "run.yaml").open(encoding="utf-8") as stream:
        run = yaml.safe_load(stream)

    initial_force_n = zero_position_static_load_n(physical, run)
    duration_s = RAMP_DURATION_S + SWEEP_DURATION_S
    sample_count = round(duration_s / SAMPLE_INTERVAL_S)
    time_s = np.linspace(0.0, duration_s, sample_count + 1)

    ramp_force_n = initial_force_n + (TARGET_FORCE_N - initial_force_n) * (
        time_s / RAMP_DURATION_S
    )
    sweep_time_s = np.maximum(time_s - RAMP_DURATION_S, 0.0)
    sweep_rate_hz_s = (END_FREQUENCY_HZ - START_FREQUENCY_HZ) / SWEEP_DURATION_S
    phase_rad = (
        2
        * np.pi
        * (START_FREQUENCY_HZ * sweep_time_s + 0.5 * sweep_rate_hz_s * sweep_time_s**2)
    )
    swept_force_n = TARGET_FORCE_N + OSCILLATION_AMPLITUDE_N * np.sin(phase_rad)
    force_n = np.where(time_s <= RAMP_DURATION_S, ramp_force_n, swept_force_n)

    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "actuator_a_force_n": force_n,
            "actuator_b_force_n": force_n,
            "hot_ambient_c": np.full_like(time_s, 120.0),
            "duct_air_speed_m_s": np.full_like(time_s, 2.0),
        }
    )
    profile.to_parquet(OUTPUT, index=False)
    short_output = OUTPUT.with_name("inputs_5s.parquet")
    short_profile = profile.loc[profile.time_s <= 5.0].copy()
    short_profile.to_parquet(short_output, index=False)
    print(f"Wrote {len(profile)} samples to {OUTPUT}")
    print(f"Wrote {len(short_profile)} samples to {short_output}")
    print(f"Initial zero-position static load: {initial_force_n:.2f} N per actuator")
    print(f"Profile duration: {duration_s:.1f} s")


if __name__ == "__main__":
    main()
