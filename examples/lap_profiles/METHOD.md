# Acceleration estimation

[OpenF1's car and location feeds](https://openf1.org/docs/) provide original
speed and approximate XY position at irregular intervals around 3.7 Hz.
Acceleration is derived from those channels. The resulting estimates provide
coarse lap load envelopes; smoothing attenuates fast transients and the peaks
depend on the smoothing window. They are not onboard accelerometer data.

| Column | Definition |
| --- | --- |
| `time_s` | Seconds since the reported lap start |
| `vehicle_speed_kph` | Original vehicle speed, with interpolated lap endpoints |
| `distance_from_speed_m` | Cumulative trapezoidal integral of speed divided by 3.6 |
| `interpolated_boundary` | True for speed samples added at exact lap endpoints |
| `estimated_longitudinal_acceleration_m_s2` | Smoothed speed derivative; positive accelerating, negative braking |
| `estimated_longitudinal_acceleration_g` | Longitudinal estimate divided by 9.80665 |
| `estimated_lateral_acceleration_m_s2` | Horizontal acceleration normal to travel; positive left in published XY coordinates |
| `estimated_lateral_acceleration_g` | Lateral estimate divided by 9.80665 |
| `longitudinal_estimate_valid` | False where speed support has a large gap or a fit is invalid |
| `lateral_estimate_valid` | False where speed or position support has a large gap or a fit is invalid |

All native in-lap speed values are retained. Lap endpoints are linearly
interpolated from bracketing samples to cover exactly 0 to the reported lap
duration. Distance is an estimate from speed; it is not surveyed track distance
and is not normalized to circuit length.

For speed `v` in m/s, longitudinal acceleration is `dv/dt`. Lateral acceleration
is `v * d(heading)/dt`, with heading rate:

```text
(x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2)
```

This cancels the uniform XY length scale. Positive lateral acceleration means
leftward turning in the published XY plane. Clockwise laps give a negative
net heading change; Interlagos gives positive, and Suzuka's figure-eight has
approximately zero net heading change, consistent with the circuit layouts.

Derivatives use Gaussian-weighted local cubic fits directly on the irregular
timestamps. Nominal windows are 1.5 s for speed and 2.0 s for position, expanded
to at least seven samples when necessary. Metadata records actual support
span ranges. Buffered samples from neighboring laps support boundary fits.
Source gaps larger than 0.75 s anywhere in a fit's support produce a false
validity flag and blank/NaN acceleration. Plots leave those segments blank.

The [position feed is approximate and does not resolve lateral racing-line placement](https://openf1.org/docs/#location).
Lateral acceleration is consequently less certain than the speed-based
longitudinal estimate. Banking, road gradient, gravity projection, vertical
acceleration and tyre sideslip are not reconstructed. Estimated acceleration
peaks should be treated as approximate envelope values.

OpenF1 describes lap-start timestamps as approximate. Each profile records
its original source URLs, selected lap, smoothing settings, acceleration
coverage and processing time in its `.json` file and Parquet metadata. The
`_raw_openf1.json` files preserve original source records. Every profile's
provenance also links to the corresponding FIA classification.

Analytic straight-line and circular-motion tests verify acceleration units,
left/right signs, independence from uniform XY coordinate scale, rotation
and translation, and exclusion of source gaps.
