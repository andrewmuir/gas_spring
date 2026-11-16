"""Applied loads are positive in actuator extension. Units: N, seconds."""

import numpy as np


def transfer_force(force_n: float, time_s: float) -> float:
    """Return the applied force unchanged until a transfer model is supplied."""
    return float(force_n)


class InputInterpolator:
    """Validate and copy a load history once, then interpolate during a solve."""

    channels = (
        "actuator_a_force_n",
        "actuator_b_force_n",
        "hot_ambient_c",
        "duct_air_speed_m_s",
    )

    def __init__(self, frame):
        self.sample_time = np.array(frame["time_s"], dtype=float, copy=True)
        if (
            len(self.sample_time) < 2
            or not np.isfinite(self.sample_time).all()
            or np.any(np.diff(self.sample_time) <= 0)
        ):
            raise ValueError(
                "input time must be finite, strictly increasing and contain at least two points"
            )
        self.data = {}
        for name in self.channels:
            data = np.array(frame[name], dtype=float, copy=True)
            if data.shape != self.sample_time.shape or not np.isfinite(data).all():
                raise ValueError(f"{name} must be a finite input series")
            self.data[name] = data
        if np.any(self.data["duct_air_speed_m_s"] < 0):
            raise ValueError("duct air speed cannot be negative")

    def __call__(self, time_s):
        if (
            not np.isfinite(time_s)
            or not self.sample_time[0] <= time_s <= self.sample_time[-1]
        ):
            raise ValueError("requested time lies outside input history")
        result = {
            name: float(np.interp(time_s, self.sample_time, data))
            for name, data in self.data.items()
        }
        for name in ("actuator_a_force_n", "actuator_b_force_n"):
            result[name] = transfer_force(result[name], float(time_s))
        return result


def interpolate_inputs(frame, time_s):
    """Linearly interpolate validated input channels within the supplied span.

    The force transfer function is applied after interpolation. Extrapolation is
    rejected so a run cannot silently continue beyond its specified load history.
    """
    return InputInterpolator(frame)(time_s)


def shaft_position_m(piston_position_m, force_n, stiffness_n_m):
    """Shaft coordinate; positive applied load stretches the shaft in extension."""
    if (
        not np.isfinite([piston_position_m, force_n, stiffness_n_m]).all()
        or stiffness_n_m <= 0
    ):
        raise ValueError("invalid shaft position, force or stiffness")
    return piston_position_m + force_n / stiffness_n_m
