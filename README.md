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

## Dashboard

Install the optional dashboard dependencies and start the local viewer from the
project root:

```powershell
python -m pip install -e ".[dashboard]"
python -m streamlit run gas_spring/dashboard.py
```

The dashboard defaults to `examples/inputs.parquet` and
`examples/output.parquet`. In the sidebar, change either local path or upload a
Parquet file. It provides an adjustable time window, unit-separated channel
plots, virtual sensor versus truth comparisons, input force and environment
profiles, and mass/energy audit plots. Large traces are downsampled for display;
the source Parquet files are not altered.

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

Heat-transport donor selection is smoothed only around flow reversal
(1e-11 m3/s for liquid, 1e-8 kg/s for gas), along with the liquid restriction's
zero-flow transition. Mass and energy fluxes remain equal and opposite across
each link. The liquid mixture density is evaluated consistently with its
transported temperature, preserving the volume constraint. These small
numerical transition scales avoid ambiguous derivatives at zero flow.

The CLI reports progress at 0.5-second simulated-time checkpoints, with a
wall-time heartbeat during longer solves. Add `--diagnostics` to see BDF
integration steps, function/Jacobian evaluations, factorizations and stop events.
Total RHS calls include the evaluations needed to construct numerical Jacobians;
SciPy's `nfev` counter excludes those. BDF replaces Radau after a five-second
comparison and tolerance check. Thermal expansion-source accelerations are
differentiated analytically, and the gas-flow calculation retains small pressure
differences directly rather than subtracting reconstructed absolute pressures.
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

The truth output also includes total liquid and nitrogen mass, total stored
energy and external power. `truth_sampled_energy_balance_residual_j` compares
stored-energy change with external power integrated by the trapezoid rule at
the **output sample times**. Sparse output can therefore make the residual
appear large even when the integration itself balances energy well. The audit
uses applied load power F*velocity at the piston; elastic energy in the massless input
shafts is reported separately as `truth_input_shaft_strain_energy_j`.

## Model and limitations

- The hydraulic circuit has separate actuator restrictions, a finite-volume
  pipe with inertance and loss, the blow-off branch, and a separate main inlet
  restriction. The original fitted restriction asymptote has a provisional
  tangent high-flow continuation so its inverse remains defined.
- Nitrogen is represented as an ideal gas with constant heat capacities.
  Chambers use core and boundary temperatures; gas pipe cells use one
  temperature each. Gas-pipe flow uses quasi-steady mean-density Darcy loss;
  choking, gas inertia and real-gas corrections are not represented. These
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
