# Gas spring reference model: experimental

This project contains a provisional coupled hydraulic, pneumatic and thermal
simulation. It is **not yet a validated reference model**. Important physical
parameters are estimates, and the numerical solver requires validation under
representative loads and convergence checks. Do not draw correlation
conclusions from its output until those checks are complete.

## Run

```powershell
python -m pip install -e ".[test]"
python examples/make_input.py
python -m gas_spring.cli --physical config/physical.yaml --run config/run.yaml --initialize
python -m gas_spring.cli --physical config/physical.yaml --run config/run.yaml --thermal-estimate
python -m gas_spring.cli --physical config/physical.yaml --run config/run.yaml --simulate --diagnostics
python -m pytest -q
```

For timing tests, `config/run_5s.yaml` uses `examples/inputs_5s.parquet`: the
unchanged first 5 seconds of the full input, including the 4-second ramp and
first second of cycling (1001 samples). Its output is `examples/output_5s.parquet`.
Its initial blow-off pressure matches the current full run at 70 bar.

```powershell
python -m gas_spring.cli --physical config/physical.yaml --run config/run_5s.yaml --simulate --diagnostics
python examples/benchmark.py --label timing
```

The benchmark saves wall/CPU times, solver counters, parameters and source hash
under `examples/benchmarks/`. Add `--profile` for a Python call profile.
Temporary parameter experiments use, for example,
`--label friction_test --set run.solver.friction_velocity_scale_m_s=0.02`;
they do not edit the YAML files. See
[the timing investigation](examples/benchmarks/investigation.md) for measured
speed improvements, accuracy checks and parameter sensitivity results.
The [solver options comparison](examples/benchmarks/solver_options.md) tests
BDF, LSODA, restart spacing and tolerances against the current 70-bar blow-off
setting, and ranks the next optimization work.
The implemented [gas reduction and block Jacobian](examples/benchmarks/gas_reduction.md)
includes a complete 19-second timing run and symmetric/asymmetric comparisons
through 25 Hz.
The subsequent [restriction calibration](examples/benchmarks/restriction_calibration.md)
sets pressure-sensor white-noise standard deviation to 0.2 bar and fits the
terminal valve losses to approximately 1 bar per actuator and 4 bar at the main
accumulator inlet. These are operating-point targets for the current 19-second
input, not fixed losses at every flow rate.

The [lap profile collection](examples/lap_profiles/README.md) contains Barcelona
and ten other 2023 circuits, using Verstappen race laps. Each includes speed in
km/h and estimated longitudinal/lateral acceleration in m/s² and g, with
CSV/Parquet data, plots, validity flags and original source records for
developing vehicle-based input profiles. Acceleration is derived from speed
and approximate position, with smoothing and source gaps documented.

## Convert laps to actuator inputs

`config/lap_input.yaml` contains the example 750 kg car settings. Select
`axle: front` or `axle: rear`. Actuator A uses the left wheel load and B the
right wheel load from that axle. Each output retains all four wheel loads,
both axle totals, and the acceleration and load transfer channels.

Set `motion_ratio` as actuator travel divided by wheel travel. Supply one
positive number for both axles or separate `front` and `rear` values. The
converter divides wheel load by this ratio to get actuator force: for example,
8 mm of actuator travel per 10 mm of wheel travel gives a ratio of 0.8 and
converts 4,000 N at the wheel to 5,000 N at the actuator. The sample values
are 1.0 placeholders. The ratio does not change the wheel load channels or
the optional preload, which is already specified at the actuator.

```powershell
python -m gas_spring.lap_input --settings config/lap_input_baseline.yaml --output-dir examples/model_inputs examples/lap_profiles/barcelona_2023_verstappen_lap61.parquet
```

Accepts CSV or Parquet. Pass multiple paths after the options or use the quoted
pattern `"examples/lap_profiles/*.parquet"` to convert the whole collection.
The converter calculates loads at the source timestamps, then linearly
interpolates the resulting profile onto a 100 Hz grid (one row every 10 ms).
If the lap ends between grid points, its final values are held to the next
grid point. The original lap duration is saved in the Parquet metadata.
Each output is named `<lap>_<axle>_input.parquet`. To simulate the Barcelona
front axle after generating its input, run:

```powershell
python -m gas_spring.cli --physical config/physical.yaml --run config/run_barcelona_front.yaml --simulate
```

For another lap or axle, set `input_parquet` in a copy of `config/run.yaml`
to the corresponding `../examples/model_inputs/...` path and give
`output_parquet` a separate name.

The static axle loads are `mass × g × front_weight_fraction` at the front and
`mass × g × (1 − front_weight_fraction)` at the rear. Longitudinal transfer
`mass × ax × cog_height / wheelbase` shifts load to the rear under acceleration
and to the front under braking. Lateral transfer
`mass × ay × cog_height / track_width` shifts load to the right for a left turn;
mechanical balance sets the fraction of that transfer applied at the front.
Aero adds `aero_coefficient × (speed_kph / 3.6) ** aero_power` newtons, divided
between the axles by aero balance and equally between wheels on each axle.
Both balances are front fractions on straight lines through the two km/h
anchors, including outside the anchors as long as the fractions stay in [0, 1].
At 300 km/h, `config/lap_input_baseline.yaml` produces 15,000 N of
downforce and `config/lap_input.yaml` produces 16,500 N (+10%). To regenerate
the higher downforce inputs and run Barcelona:

```powershell
python -m gas_spring.lap_input --settings config/lap_input.yaml --output-dir examples/model_inputs/aero_plus_10pct "examples/lap_profiles/*.parquet"
python -m gas_spring.cli --physical config/physical.yaml --run config/run_barcelona_front_aero_plus_10pct.yaml --simulate
```

To compare blow-off precharge, run the same higher downforce inputs at
65 bar gauge instead of 70 bar:

```powershell
python -m gas_spring.cli --physical config/physical.yaml --run config/run_barcelona_front_aero_plus_10pct_blowoff_65bar.yaml --simulate
```

The two results are saved separately in `examples/runs` and share the same
input profile.

- The example interpolates interior acceleration gaps up to 2.3 s between
  valid samples and holds the nearest valid value for up to 0.3 s at a lap
  boundary. Filled samples have `*_acceleration_filled` set. An interpolated
  row carries the flags of the source samples used to calculate it. Omit
  `acceleration_gaps` to reject missing estimates.
- `negative_wheel_load: limit_transfer` limits the shift within an axle so the
  inside wheel does not go below zero and the axle total is preserved. Rows
  with `*_lateral_transfer_limited` set no longer achieve the requested lateral
  transfer on that axle. This is an approximation for model inputs; it does not
  predict wheel lift. Set `negative_wheel_load: error` to reject those rows.
- The optional `preload` introduces a separate linear ramp from the model's
  zero-displacement force to the first lap loads. In the example, the real lap
  starts at model `time_s = 4` and `lap_time_s = 0`. All rows before the lap
  starts have `is_preload` set. Omit `preload` to start directly at the lap loads,
  or update its force if the bench geometry or initial pressures change.
- Duct air speed and hot ambient are constant settings, independent of the lap
  telemetry. `config/run.yaml` can also specify its own hot ambient temperature.

For use from Python, `build_model_inputs(lap_dataframe, settings_dictionary)`
returns an independent DataFrame. Its required input columns are `time_s`,
`vehicle_speed_kph`, `estimated_longitudinal_acceleration_m_s2`, and
`estimated_lateral_acceleration_m_s2`. It also honors the lap files' optional
`longitudinal_estimate_valid` and `lateral_estimate_valid` columns.
`process_lap_files(path_or_paths, settings_dictionary, output_dir)` writes the
Parquets and returns their paths. Both functions are in `gas_spring.lap_input`.

## Batch runs

Edit [config/batch_example.yaml](config/batch_example.yaml) to choose a run
template, base lap settings, tracks, and parameter values. Preview the matrix
before launching the batch:

```powershell
python -m gas_spring.run_harness config/batch_example.yaml --dry-run
python -m gas_spring.run_harness config/batch_example.yaml
```

The example applies all combinations of −10%, baseline, and +10% aero,
65 and 70 bar blow-off precharge, and 0 and +0.5 bar actuator A pressure sensor
offset to Barcelona and Silverstone. That gives **24 results from 12 physical
solves**. Cases that differ only in virtual sensor calibration or random seed
share the same solve and get separate sensor readings and Parquet results.
For all tracks, set `tracks: ["../examples/lap_profiles/*.parquet"]`.

Each key in `run_overrides` is a dot-separated path in the run YAML. For
example, `initial_gas_gauge_pressure_bar.main: [33, 35, 37]` sweeps main
precharge. Aero scales multiply the coefficient from `lap_settings`.
By default, the harness recomputes the startup preload from the physical
geometry and each case's initial pressures. Set `balance_preload: false` to
use the fixed preload in the lap settings.

The batch saves inputs, resolved YAMLs, and named results under
`output_dir`. Point the dashboard's **Run directory** at that folder to load
them. Check `manifest.json` there for case settings, status, elapsed time,
and errors. A failed case does not stop the rest, and the command exits with
a failure status if any case fails. Run the same command again to resume
pending or failed cases, or add `--force` to recompute completed cases.

## Dashboard

Install the optional dashboard dependencies and start the local viewer from the
project root:

```powershell
python -m pip install -e ".[dashboard]"
python -m streamlit run gas_spring/dashboard.py
```

Set **Run directory** in the sidebar and pick a run by name. The default
directory is `config`. The dashboard scans subdirectories for run YAMLs and
saved simulation results. A run YAML can set `run_name` and names both
`input_parquet` and `output_parquet`, relative to the YAML. New output
Parquets also record the run name and the path to the input used, relative to
the output. This lets the dashboard match the input from either directory
without having to select it again. Click **Refresh runs** after saving a new
result.

The Barcelona result is `examples/runs/barcelona_2023_lap61_front_100hz.parquet`.
Choose **Barcelona 2023 | Verstappen lap 61 | Front axle** from either
`config` or `examples/runs`. For older results without a run YAML or saved
link, the sidebar still accepts separate local files or uploads.

The dashboard provides an adjustable time window, plots grouped by unit,
sensor comparisons, the input profile, and mass and energy audits. Large
traces are downsampled for display; the source Parquet files are not altered.

The overview includes actuator velocity. Each new result contains
`truth_actuator_a_shaft_velocity_m_s` and `truth_actuator_b_shaft_velocity_m_s`,
alongside the existing `truth_<piston>_piston_velocity_m_s` channels. Shaft
velocity is piston velocity plus the rate of elastic shaft stretch (`dF/dt / k`),
so it uses the same coordinate as actuator displacement. Positive is extension.
Force inputs are piecewise linear: their slope is right-sided at sample knots
and left-sided at the final sample. Velocities are model truth, with no added
sensor noise.

The **Solve metrics** tab shows accepted internal step sizes, integration time
per checkpoint window, simulation elapsed and CPU time, RHS/Jacobian costs,
evaluation counts, LU factorizations, stop events, and solver settings. Window
metrics and individual accepted steps can be downloaded as CSV. These metrics
cover the complete run even when the output plots use a shorter time window.
Step sizes are the actual accepted integrator steps, independent of the output
sampling interval. SciPy does not expose rejected step sizes; timing and work
counters include that work. No per-step solve time is inferred from window time.

Metrics are saved automatically, with or without `--diagnostics`, in the output
Parquet's pandas metadata. For Python use, read them with
`pd.read_parquet(path).attrs["solve_metrics"]` (keys: `summary`, `windows`,
`steps`, `events`, `settings`). They stay with a local file or uploaded result;
no companion files are needed. Older outputs still open, but need a new run to
gain saved solve metrics. Stored simulation timing excludes imports, input
loading, and file writing; the CLI separately reports complete command timing.

Without `--simulate`, the CLI validates inputs only. `--initialize` also checks
that the first forces can be held at rest by the specified initial gas pressures,
stops and provisional seal friction. The example
forces and other data are illustrative, not user-supplied load histories.
The initializer checks whether static seal friction could hold the initial
loads. With smooth friction, any remaining unbalanced force at zero speed
produces motion, because the smooth law has no exact static sticking.
The simulator uses incompressible liquid with thermal expansion. Liquid pressure
is solved algebraically from piston force balance, pipe inertia, restrictions
and volume continuity. The shared pipe and short blow-off connection retain
fluid inertia, including kinetic energy, without adding independent flow states.
There is no liquid pressure state or compression spring.
The main accumulator position and velocity are eliminated from the integration
states: its motion follows the actuators, blow-off and liquid thermal expansion.

The run YAML sets `roll_spring_stiffness_n_m`. The spring reacts to relative
piston travel `x_a - x_b` with forces `-k * (x_a - x_b)` on A and
`+k * (x_a - x_b)` on B. Equal piston travel leaves the spring unloaded.
The example rate of 212,500 N/m gives 4,250 N of reaction on each side
at 20 mm separation, balancing an 8,500 N applied force difference at
±10 mm about the mean piston position in static equilibrium. Set the rate to
zero to disable the spring; older run configs without the setting default to
zero. The outputs include each actuator's roll spring force, relative travel,
and stored spring energy. The system energy audit includes that stored energy.

Every mechanical travel stop is a unilateral elastic contact with
`end_stops.stiffness_n_m: 10000000` (10 kN/mm). The piston may move slightly
beyond the first-contact coordinate, with reaction `F = k * deflection`.
Contact and separation do not reset position or velocity. The spring stores
`0.5*k*deflection^2`, included in the system energy audit; no extra stop damping
or impact heat is assumed. The blow-off lower stop is initially preloaded by
its gas/fluid force imbalance, preserving the loaded zero-position reference.
Its contact surface is offset by preload/stiffness, so that preload recovers
before separation. Stop force, lower/upper deflections and energy are emitted
for each piston as `truth_<piston>_end_stop_force_n`,
`truth_<piston>_lower_stop_deflection_m`,
`truth_<piston>_upper_stop_deflection_m` and
`truth_<piston>_end_stop_energy_j`.

The main accumulator uses `Vgas = Vgas0 + piston_area*x` and uncapped ideal-gas
pressure. Compression is negative travel; pressure diverges as its closed gas
volume approaches zero. The configured +7 mm stop is on the expansion side,
where 50 mL initial gas volume has increased to about 61.1 mL. It is reachable
and carries load through its elastic reaction. No liquid compliance is added.

The connected actuator/reservoir gas uses one shared pressure and total energy.
Each chamber and pipe cell retains its mass and temperature, including chamber
boundary layers. Node energies follow `U_i = P*V_i/(gamma-1)`, and conservative
enthalpy flows follow from the local energy balances. This removes the tiny
pressure-relaxation states; the two closed accumulator gases remain separate
ideal-gas springs. Connected-gas pressure loss and acoustic transients are
neglected. The output estimates omitted Darcy losses and gas Mach number in
`truth_gas_network_estimated_max_pressure_drop_pa`,
`truth_gas_network_estimated_pressure_drop_fraction` and
`truth_gas_network_estimated_max_mach_number`. `--diagnostics` prints the peak
pressure-loss estimate so this approximation can be assessed for each load case.

Heat-transport donor selection is smoothed only around flow reversal
(1e-11 m3/s for liquid, 1e-8 kg/s for gas), along with the liquid restriction's
zero-flow transition. Mass and energy fluxes remain equal and opposite across
each link. The liquid mixture density is evaluated consistently with its
transported temperature, preserving the volume constraint. These small
numerical transition scales avoid ambiguous derivatives at zero flow.

The CLI reports progress at 0.5-second simulated-time checkpoints, with a
wall-time heartbeat during longer solves. Add `--diagnostics` to see BDF
integration steps, function/Jacobian evaluations, factorizations and stop events.
If a trial state gives a chamber an invalid volume, the solver retries that
window from the last checkpoint with a smaller maximum step. The retry count,
chamber name and reduced step limit are saved in the solve metrics.
Total RHS calls include the evaluations needed to construct numerical Jacobians;
SciPy's `nfev` excludes those and is reported only for attempts that returned a
solution object. BDF replaces Radau after a five-second
comparison and tolerance check. Thermal expansion-source accelerations are
differentiated analytically. The Newton Jacobian reuses the gas/thermal block
and freezes weak thermal-volume acceleration coefficients during matrix
construction. The physical RHS still solves the full differentiated liquid
constraint; this approximation affects Newton iteration, not its target equations.
Each diagnostic integration time refers to **that simulation window only**.
Progress also shows total elapsed wall time, and completion reports the sum of
all integration windows plus imports/loading, simulation setup and sampled
outputs, and file writing. The final `Total elapsed wall time` includes loading
through saving the output and is shown with or without `--diagnostics`.
The benchmark reports `total_elapsed_wall_s` separately from
`simulation_wall_time_s`; its historical `wall_time_s` field covers only
`simulate()` and has an explicit scope label in new reports.
`solver.progress_interval_s` changes the checkpoint spacing. Smooth seal
friction uses `tanh(velocity / friction_velocity_scale_m_s)`; the default scale
is 0.005 m/s. Contact events only report elastic contact and separation.

Run and physical parameters are configured separately in the YAML files. Input
and output paths are resolved relative to the run YAML. Required Parquet input
columns are `time_s`, `actuator_a_force_n`, `actuator_b_force_n` and
`duct_air_speed_m_s`. With `hot_ambient.source: parquet`, the configured hot
temperature column is required as well. With `source: constant`, the run YAML
provides the temperature; its initial value is 120 °C. Input channels are
linearly interpolated, with no extrapolation beyond the recorded times. Output
is sampled at input times.

Positive applied load drives **actuator extension**. Extension reduces the
actuator high-side liquid volume, expels liquid into the accumulator and
compresses the main accumulator gas. High-side pressure opposes extension;
low-side gas pressure assists it. At the initial 35 bar hydraulic and 15 bar
low-side gas pressures, the balancing applied load is approximately **+1729 N**
per actuator, subject to the assumed friction. Arbitrary initial forces can make the requested static
initial state infeasible; the initializer reports that instead of changing the
specified gas pressures or silently starting out of balance. Running
`python examples/make_input.py` generates the current 19-second two-actuator
profile at 5 ms intervals. It starts at the zero-position balancing load
calculated from the configured low-side and main pressures (about +1729.45 N
per actuator with the supplied settings), ramps linearly to +5000 N over
4 seconds, then applies a ±3000 N sinusoid about +5000 N. Its frequency sweeps
linearly from 1 to 4 Hz during the following 15 seconds.
The generator also writes the first-five-second input used by the timing config.
`config/physical.yaml` currently treats the roughly
30 kg wheel assembly as shared equally between the two actuator inputs (15 kg
each) under symmetric motion. This is an effective-mass split; the model does
not constrain the two actuator displacements to match, and it does not track
the wheel's separate thermal capacity.

The output includes state columns prefixed `truth_` and measurement columns
prefixed `sensor_`. Pressure channels are gauge bar; temperature is °C;
displacement sensors are mm; force is N. Actuator displacement truth is the
shaft coordinate, including input-shaft compliance, while piston positions and
velocities are also recorded. Low-side gas pressure and temperature sensors use
the reservoir. Sensor gain, offset, noise, drift and displacement temperature
sensitivity use the profiles and seed in the YAML files.
White-noise standard deviations are 0.2 bar for pressure, 0.3 mm for all
displacement channels, and 25 N for both actuator force channels. These are
measurement-noise levels applied to the recorded sensor outputs.

Dedicated valve diagnostics are `truth_<name>_restriction_pressure_drop_bar`,
`truth_<name>_restriction_flow_m3_s` and
`truth_<name>_restriction_dissipation_w`, for `actuator_a`, `actuator_b` and
`main_accumulator`. Positive drop/flow is from the junction into the chamber.
These isolate each restriction from pipe loss/inertia and piston forces.

The truth output also includes total liquid and nitrogen mass, total stored
energy and external power. `truth_sampled_energy_balance_residual_j` compares
stored-energy change with external power integrated by the trapezoid rule at
the **output sample times**. Sparse output can therefore make the residual
appear large even when the integration itself balances energy well. The audit
uses applied load power F*velocity at the piston; elastic energy in the massless input
shafts is reported separately as `truth_input_shaft_strain_energy_j`.

## Model and limitations

See the [current model reference](docs/current_model.md) for the circuit,
governing equations, thermal states, initialization, solver and output units.

See the [four reduced model proposals](docs/model_fidelity_proposals.md) for
alternatives aimed at faster lap sweeps and simpler parameter studies.

- The hydraulic circuit has separate actuator restrictions, a finite-volume
  pipe with inertance and loss, the blow-off branch, and a separate main inlet
  restriction. The original fitted restriction asymptote has a provisional
  tangent high-flow continuation so its inverse remains defined.
- Nitrogen is represented as an ideal gas with constant heat capacities.
  Chambers use core and boundary temperatures; gas pipe cells use one
  temperature each. Connected low-side gas has algebraic common pressure and
  conservative enthalpy transport; Darcy loss is estimated for diagnostics.
  Choking, gas inertia and real-gas corrections are not represented. These
  approximations need particular scrutiny at blow-off pressures.
- Liquid specific volume increases linearly with temperature:
  `v = (1 + alpha*(T - 20))/rho20`. Sensible energy uses constant configured
  heat capacity. Mass and sensible energy are advected conservatively using
  upstream core temperature. Restriction and pipe losses heat the liquid;
  seal friction heats the piston faces. Stop strain energy is stored elastically.
  The thermal approximation neglects the small pressure-work correction to
  liquid heat capacity. Its mechanical expansion work is exposed as
  `truth_thermal_expansion_work_w` and remains visible in the energy audit.
  Viscosity is interpolated logarithmically between 20 and 100 C.
  Cavitation is not represented. Algebraic pressures are reported without
  clamping: a nonpositive absolute pressure means that the imposed loads
  require liquid tension and the single-phase physical assumption has failed.
- Actuator walls and both pipe walls retain axial temperature history.
  Wetted wall areas follow piston position. Accumulator wetted areas assume
  the full chamber volumes lie within their bores; a remote gas volume would
  require different geometry. Chamber liquid and gas each use core and
  boundary thermal zones. Liquid pressure is common to both zones, and
  advection uses the upstream core temperature and density. The chamber temperature channels
  report the core; boundary temperatures are included in truth output. Axial
  conduction, gas and liquid
  advection, piston heat flow and the finite cooled region are represented.
- Moving pistons use pressure-dependent smooth friction by default:
  `F = F_limit * tanh(v / friction_velocity_scale_m_s)`. The 0.005 m/s
  transition scale is provisional; it allows slow creep instead of exact
  static sticking. Travel stops use the elastic contact model described above;
  their reaction is transmitted through the incompressible fluid constraint.
- The 100 °C after three minutes static actuator warm-up is an estimate used
  to set an initial hot-side coefficient, **not yet a validated outcome** of
  the spatial thermal model. `--thermal-estimate` runs a separate six-node,
  fixed-mass estimate with no flow or pipe exchange; it predicts roughly 102 °C
  for the housing, 93 °C for fluid, 89 °C for the gas core and 68 °C for the
  piston faces after 180 seconds. These are provisional and depend strongly on
  the assumed thermal masses and coefficients. Coupled
  trial results from previous implementations are not validation of this
  solver. The short junctions and some chamber volumes are guessed.
- Initial pipe temperatures interpolate between connected chamber temperatures.
  If two connected chambers differ, their arithmetic mean is used as the
  provisional junction temperature. Static initialization requires the
  blow-off at its zero stop and non-negative common hydraulic gauge pressure.

Perform sensitivity and convergence checks for pipe and wall cell counts,
solver tolerances and time steps. Validate friction, high-flow loss, thermal
properties and material data against measurements before treating the results
as a physical reference.
