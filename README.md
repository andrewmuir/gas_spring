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
python -m gas_spring.cli --physical config/physical.yaml --run config/run.yaml --simulate
python -m pytest -q
```

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
stops and provisional seal friction. `--simulate` performs that check and writes
the requested Parquet output only after the calculation succeeds. The example
forces and other data are illustrative, not user-supplied load histories.

Run and physical parameters are configured separately in the YAML files. Input
and output paths are resolved relative to the run YAML. Required Parquet input
columns are `time_s`, `actuator_a_force_n`, `actuator_b_force_n` and
`duct_air_speed_m_s`. With `hot_ambient.source: parquet`, the configured hot
temperature column is required as well. With `source: constant`, the run YAML
provides the temperature; its initial value is 120 °C. Input channels are
linearly interpolated, with no extrapolation beyond the recorded times. Output
is sampled at input times.

Compression force is **positive**; actuator extension displacement is
**positive** and expels fluid. Consequently, at the specified initial 35 bar
hydraulic and 15 bar low-side gas pressures, a roughly 1.73 kN *extension*
force is needed per actuator, represented as approximately **-1729 N**, subject
to the assumed friction. Arbitrary initial forces can make the requested static
initial state infeasible; the initializer reports that instead of changing the
specified gas pressures or silently starting out of balance. Running
`python examples/make_input.py` generates the current 19-second two-actuator
profile at 5 ms intervals. It starts at the zero-position balancing load
calculated from the configured low-side and main pressures (about -1729.45 N
per actuator with the supplied settings), ramps linearly to +5000 N over
4 seconds, then applies a ±3000 N sinusoid about +5000 N. Its frequency sweeps
linearly from 1 to 4 Hz during the following 15 seconds. Some load histories
cause many short stick–slip events and can be much slower; a ten-second
synthetic history exhibited this behaviour.

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
uses applied force power at the piston; elastic energy in the massless input
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
- The DOT 4 equation of state uses provisional constant bulk modulus and
  expansion, with a thermodynamically consistent constant-cv energy model.
  Viscosity is a provisional log-linear interpolation and is held constant
  outside 20–100 °C. The cited material properties and guesses are documented
  in `config/physical.yaml`. Cavitation, dissolved gas and fluid phase changes
  are not modeled; non-positive absolute liquid pressure is rejected.
- Actuator walls and both pipe walls retain axial temperature history.
  Wetted wall areas follow piston position. Accumulator wetted areas assume
  the full chamber volumes lie within their bores; a remote gas volume would
  require different geometry. Chamber liquid and gas each use core and
  boundary thermal zones; the liquid zones share an approximate mean density
  for pressure and enthalpy calculations. The chamber temperature channels
  report the core; boundary temperatures are included in truth output. Axial
  conduction, gas and liquid
  advection, piston heat flow and the finite cooled region are represented.
- Moving pistons use pressure-dependent Coulomb friction and event-driven
  stick, release and stops. Static and sliding friction are provisionally equal,
  with a configurable 0.000001 N numerical breakaway margin; a lower sliding
  ratio can be configured when data is available. Impacts are assumed perfectly
  inelastic, with
  impact and seal-friction energy added to the piston. This behavior needs
  dedicated convergence and impact verification under representative loads.
- The 100 °C after three minutes static actuator warm-up is an estimate used
  to set an initial hot-side coefficient, **not yet a validated outcome** of
  the spatial thermal model. `--thermal-estimate` runs a separate six-node,
  fixed-mass estimate with no flow or pipe exchange; it predicts roughly 102 °C
  for the housing, 93 °C for fluid, 89 °C for the gas core and 68 °C for the
  piston faces after 180 seconds. These are provisional and depend strongly on
  the assumed thermal masses and coefficients. Two constant-force coupled
  trial runs reached approximately 101.5 °C (one pipe cell, two wall cells)
  and 101.2 °C (three pipe cells, six wall cells) at one actuator wall location.
  The finer run gave about 84.5 °C for fluid core, 91.9 °C for fluid near the
  wall and 90.6 °C for gas core. This is a limited check, not calibration
  against measured hardware. The short junctions and some chamber volumes are
  guessed.
- Initial pipe temperatures interpolate between connected chamber temperatures.
  If two connected chambers differ, their arithmetic mean is used as the
  provisional junction temperature. Static initialization requires the
  blow-off at its zero stop and non-negative common hydraulic gauge pressure.

Perform sensitivity and convergence checks for pipe and wall cell counts,
solver tolerances and time steps. Validate friction, high-flow loss, thermal
properties and material data against measurements before treating the results
as a physical reference.
