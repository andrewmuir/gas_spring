"""Replaceable force input transfer function. Units: N, seconds."""
import numpy as np

def transfer_force(force_n: float, time_s: float) -> float:
    """Return the applied force unchanged until a transfer model is supplied."""
    return float(force_n)


def interpolate_inputs(frame, time_s):
    """Linearly interpolate validated input channels within the supplied span.

    The force transfer function is applied after interpolation. Extrapolation is
    rejected so a run cannot silently continue beyond its specified load history.
    """
    sample_time = np.asarray(frame["time_s"], dtype=float)
    if len(sample_time) < 2 or not np.isfinite(sample_time).all() or np.any(np.diff(sample_time) <= 0):
        raise ValueError("input time must be finite, strictly increasing and contain at least two points")
    if not np.isfinite(time_s) or not sample_time[0] <= time_s <= sample_time[-1]:
        raise ValueError("requested time lies outside input history")
    result = {}
    for name in ("actuator_a_force_n", "actuator_b_force_n", "hot_ambient_c", "duct_air_speed_m_s"):
        data = np.asarray(frame[name], dtype=float)
        if data.shape != sample_time.shape or not np.isfinite(data).all():
            raise ValueError(f"{name} must be a finite input series")
        result[name] = float(np.interp(time_s, sample_time, data))
    for name in ("actuator_a_force_n", "actuator_b_force_n"):
        result[name] = transfer_force(result[name], float(time_s))
    if result["duct_air_speed_m_s"] < 0:
        raise ValueError("duct air speed cannot be negative")
    return result


def shaft_position_m(piston_position_m, compression_force_n, stiffness_n_m):
    """Shaft coordinate, with extension positive and compression force positive."""
    if not np.isfinite([piston_position_m, compression_force_n, stiffness_n_m]).all() or stiffness_n_m <= 0:
        raise ValueError("invalid shaft position, force or stiffness")
    return piston_position_m - compression_force_n / stiffness_n_m
