"""Virtual sensors; calibration, drift and white noise are independent per channel."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Sensor:
    noise_std: float
    drift_std_per_sqrt_hour: float = 0.0
    gain: float = 1.0
    offset: float = 0.0
    temperature_coefficient: float = 0.0
    reference_temperature_c: float = 20.0

    def sample(self, truth, time_s, rng, ambient_temperature_c=None):
        values = np.asarray(truth, dtype=float)
        time = np.asarray(time_s, dtype=float)
        if values.ndim != 1 or time.shape != values.shape or not np.all(np.isfinite(values)) or not np.all(np.isfinite(time)):
            raise ValueError("truth and time must be finite, equal-length 1D arrays")
        if self.noise_std < 0 or self.drift_std_per_sqrt_hour < 0 or np.any(np.diff(time) <= 0):
            raise ValueError("noise must be non-negative and time strictly increasing")
        temp_effect = 0.0
        if ambient_temperature_c is not None:
            temp = np.asarray(ambient_temperature_c, dtype=float)
            if temp.shape != values.shape or not np.all(np.isfinite(temp)):
                raise ValueError("ambient temperature must match truth")
            temp_effect = self.temperature_coefficient * (temp-self.reference_temperature_c)
        increments = rng.normal(0, self.drift_std_per_sqrt_hour*np.sqrt(np.diff(time)/3600))
        drift = np.r_[0.0, np.cumsum(increments)] if len(time) else np.array([])
        return self.gain*values + self.offset + temp_effect + drift + rng.normal(0, self.noise_std, len(time))


# (channel name, quantity profile, output unit suffix, temperature environment)
CHANNELS = (
    ("actuator_a_fluid_pressure", "pressure", "bar", None),
    ("actuator_b_fluid_pressure", "pressure", "bar", None),
    ("actuator_a_fluid_temperature", "temperature", "c", None),
    ("actuator_b_fluid_temperature", "temperature", "c", None),
    ("low_pressure_gas_pressure", "pressure", "bar", None),
    ("main_gas_pressure", "pressure", "bar", None),
    ("blowoff_gas_pressure", "pressure", "bar", None),
    ("low_pressure_gas_temperature", "temperature", "c", None),
    ("main_gas_temperature", "temperature", "c", None),
    ("blowoff_gas_temperature", "temperature", "c", None),
    ("actuator_a_displacement", "displacement", "mm", "hot_ambient_c"),
    ("actuator_b_displacement", "displacement", "mm", "hot_ambient_c"),
    ("blowoff_displacement", "displacement", "mm", "cooler_region_temperature_c"),
    ("actuator_a_force", "force", "n", None),
    ("actuator_b_force", "force", "n", None),
)


def add_sensor_measurements(truth, physical, run):
    """Return a copy of a truth frame with reproducible sensor channels added.

    Pressure truth and measurements use gauge bar. Displacements are shaft
    positions for the actuators and piston position for the blow-off, in mm.
    Low-pressure gas truth must be supplied from the reservoir. Channel truth
    columns use `truth_<channel>_<unit>` and measurements use `sensor_...`.
    """
    required = {"time_s"}
    for channel, _, unit, ambient in CHANNELS:
        required.add(f"truth_{channel}_{unit}")
        if ambient:
            required.add(ambient)
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"missing truth columns: {sorted(missing)}")
    calibration = run.get("sensor_calibration", {})
    defaults = calibration.get("default", {})
    allowed = {"gain", "offset", "temperature_coefficient", "reference_temperature_c"}
    if not isinstance(defaults, dict) or set(defaults) - allowed:
        raise ValueError("invalid default sensor calibration")
    result = truth.copy()
    seed_sequence = np.random.SeedSequence(run["seed"])
    for (channel, profile_name, unit, ambient), child_seed in zip(CHANNELS, seed_sequence.spawn(len(CHANNELS))):
        profile = physical["sensor_profiles"][profile_name]
        overrides = calibration.get(channel, {})
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError(f"invalid calibration for {channel}")
        settings = {**defaults, **overrides}
        sensor = Sensor(noise_std=float(profile["noise_std"]),
                        drift_std_per_sqrt_hour=float(profile["drift_std_per_sqrt_hour"]),
                        **settings)
        env = None if ambient is None else result[ambient].to_numpy(dtype=float)
        result[f"sensor_{channel}_{unit}"] = sensor.sample(
            result[f"truth_{channel}_{unit}"].to_numpy(dtype=float),
            result["time_s"].to_numpy(dtype=float), np.random.default_rng(child_seed), env)
    return result
