# 2023 F1 lap profile collection

Eleven circuits: Barcelona plus ten other tracks, all using **Max Verstappen
race laps in 2023**. The downloader selects his fastest returned lap with
consistent, complete sector timing, excluding the first lap and pit entry/exit
laps. These are not necessarily the fastest laps of the entire race.

Each profile includes speed in **km/h** and **estimated longitudinal and lateral
acceleration** in **m/s²** and **g**. Acceleration is derived from public speed
and approximate position, with smoothing and validity flags. These are coarse
kinematic estimates rather than onboard accelerometer channels.

See [the catalog](CATALOG.md) for each lap's CSV, Parquet and three-panel plot,
[the collection overview](collection_overview.png), and
[the methods and column definitions](METHOD.md). The machine-readable index
is available as [CSV](index.csv) and [JSON](index.json).

The Parquet files can be loaded in the dashboard's **Input Parquet source**
panel, with separate axes for speed, acceleration in g, acceleration in m/s²
and distance.

Reproduce all eleven profiles from the project root (requires matplotlib):

```powershell
python examples/fetch_f1_lap_profiles.py
python examples/fetch_f1_lap_profiles.py --tracks barcelona silverstone
```

Raw downloads are reused; add `--refresh` to download again. The downloader
paces requests and respects rate-limit responses. The original Barcelona-only
command below also refreshes its acceleration estimates.

## Barcelona example

The saved example is **Max Verstappen, 2023 Spanish Grand Prix race, lap 61**.
The [FIA race timing](https://www.fia.com/events/fia-formula-one-world-championship/season-2023/spanish-grand-prix/race-classification)
records this as the fastest race lap, **1:16.330**. The speed samples come from
[OpenF1's car-data API](https://openf1.org/docs/#car-data), in **km/h**.

This uses the 2023 circuit without the final chicane, as described in the
[FIA layout announcement](https://www.fia.com/news/fia-homologates-updated-layout-2023-fia-formula-1-spanish-grand-prix).

Files share the stem `barcelona_2023_verstappen_lap61`:

- `.csv` and `.parquet`: elapsed time, speed, integrated distance and endpoint flags.
- `.png`: original speed-only plot; `_kinematics.png` adds estimated longitudinal and lateral acceleration.
- `.json`: sources, processing notes, units and data checks.
- `_raw_openf1.json`: original session/lap records and telemetry, including a
  one-second buffer at each end. The raw telemetry also contains throttle,
  brake, gear, engine speed and DRS channels.

The raw source bundle also contains approximate XY position. Acceleration
columns, processing and source-gap handling are described in [METHOD.md](METHOD.md).

The profile contains 284 native samples and two interpolated lap endpoints,
covering 0 to 76.330 s. Speeds range from **101 to 308 km/h**. Sampling is
irregular, averaging about 3.72 samples/s; the median interval is 0.24 s and
the largest interval is 0.52 s. Original in-lap speed samples are preserved.

Columns:

| Column | Meaning |
| --- | --- |
| `time_s` | Seconds since the reported lap start |
| `vehicle_speed_kph` | Vehicle speed in km/h |
| `distance_from_speed_m` | Cumulative trapezoidal integral of speed divided by 3.6 |
| `interpolated_boundary` | True for the two interpolated lap endpoints |

Distance is an estimate derived from the speed samples; it is not surveyed
track distance and is not normalized to the circuit length. Its final value is
4,628.46 m. OpenF1 describes the lap start timestamp as approximate. The next
lap's reported start differs from this start plus the official duration by
0.013 s.

The speed trace is ready for defining the vehicle-speed-to-actuator-load
relationship in a subsequent input profile. The Parquet can also be loaded in
the dashboard's **Input Parquet source** panel for viewing the speed in km/h.

To reproduce the download and plot from the project root (requires matplotlib):

```powershell
python examples/fetch_barcelona_speed.py
```
