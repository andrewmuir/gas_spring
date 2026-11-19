"""Coarse, derived lap accelerations from public speed and position telemetry.

These are kinematic estimates, not onboard accelerometer measurements. XY
coordinates retain their source units: heading rate is invariant to uniform
coordinate scaling, so no assumption about their length unit is required.
"""

import numpy as np
import pandas as pd

STANDARD_GRAVITY_M_S2 = 9.80665


def _series(time_s, values):
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if (time_s.ndim != 1 or len(time_s) < 7 or values.ndim == 0 or values.shape[0] != len(time_s)
            or not np.isfinite(time_s).all() or not np.isfinite(values).all()
            or np.any(np.diff(time_s) <= 0)):
        raise ValueError("telemetry needs at least seven finite samples and increasing timestamps")
    return time_s, values


def _local_cubic(time_s, values, query_s, window_s, max_gap_s):
    """Gaussian-weighted local polynomial; handles irregular source timestamps."""
    half = window_s / 2
    output = []
    values = values[:, None] if values.ndim == 1 else values
    for query in query_s:
        nearby = np.flatnonzero(abs(time_s - query) <= half)
        if len(nearby) < 7:
            nearby = np.sort(np.argsort(abs(time_s - query))[:7])
        delta = (time_s[nearby] - query) / half
        weights = np.exp(-0.5 * (delta / 0.5) ** 2)
        # Center values to keep derivatives independent of arbitrary XY origin.
        origin = values[nearby[np.argmin(abs(delta))]]
        design = np.polynomial.polynomial.polyvander(delta, 3)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            (values[nearby] - origin) * np.sqrt(weights[:, None]), rcond=None,
        )
        valid = (time_s[nearby[0]] <= query <= time_s[nearby[-1]] and rank == 4
                 and np.diff(time_s[nearby]).max() <= max_gap_s)
        output.append((coefficients[0] + origin, coefficients[1] / half,
                       2 * coefficients[2] / half**2, valid,
                       time_s[nearby[-1]] - time_s[nearby[0]]))
    return output


def estimate_lap_acceleration(speed_time_s, speed_kph, position_time_s, position_xy,
                              query_time_s, speed_window_s=1.5, position_window_s=2.0,
                              max_gap_s=0.75):
    """Estimate forward acceleration and signed horizontal cornering acceleration.

    Local cubic fits use 1.5 s speed and 2.0 s position windows (expanded to
    seven source samples when necessary), with Gaussian weights. Longitudinal = d(v)/dt. Lateral =
    v*(x'*y''-y'*x'')/(x'^2+y'^2), positive to the left in the published XY
    plane. Speed is in km/h; output accelerations are in m/s² and standard g.
    Supports with gaps >0.75 s are marked invalid and yield NaN acceleration.
    Banking, road gradient, gravity projection and tyre slip are not inferred.
    """
    speed_time_s, speed_kph = _series(speed_time_s, speed_kph)
    position_time_s, position_xy = _series(position_time_s, position_xy)
    query = np.asarray(query_time_s, dtype=float)
    if (speed_kph.ndim != 1 or np.any(speed_kph < 0)
            or position_xy.shape != (len(position_time_s), 2)
            or query.ndim != 1 or not len(query) or not np.isfinite(query).all()
            or np.any(np.diff(query) <= 0)
            or not np.isfinite([speed_window_s, position_window_s, max_gap_s]).all()
            or min(speed_window_s, position_window_s, max_gap_s) <= 0):
        raise ValueError("invalid speed, XY positions, query times or smoothing settings")
    speed_fits = _local_cubic(speed_time_s, speed_kph / 3.6, query, speed_window_s, max_gap_s)
    position_fits = _local_cubic(position_time_s, position_xy, query, position_window_s, max_gap_s)
    rows = []
    for time_s, (v, dv, _, speed_valid, speed_span), (_, dxy, ddxy, pos_valid, pos_span) in zip(
            query, speed_fits, position_fits):
        tangent_squared = float(dxy @ dxy)
        longitudinal_valid = bool(speed_valid and v[0] >= 0)
        lateral_valid = bool(longitudinal_valid and pos_valid and tangent_squared > 1e-12)
        longitudinal = float(dv[0]) if longitudinal_valid else np.nan
        heading_rate = float((dxy[0] * ddxy[1] - dxy[1] * ddxy[0]) / tangent_squared) if lateral_valid else np.nan
        lateral = float(v[0] * heading_rate)
        rows.append(dict(
            time_s=float(time_s), estimated_longitudinal_acceleration_m_s2=longitudinal,
            estimated_longitudinal_acceleration_g=longitudinal / STANDARD_GRAVITY_M_S2,
            estimated_lateral_acceleration_m_s2=lateral,
            estimated_lateral_acceleration_g=lateral / STANDARD_GRAVITY_M_S2,
            estimated_heading_rate_rad_s=heading_rate,
            longitudinal_estimate_valid=longitudinal_valid, lateral_estimate_valid=lateral_valid,
            speed_fit_span_s=float(speed_span), position_fit_span_s=float(pos_span),
        ))
    return pd.DataFrame(rows)
