# Five-second solve investigation

Measured on this computer, 15 November 2026. The short input contains the
**exact first 1001 rows, 0–5 s inclusive**, of `examples/inputs.parquet`.
It includes the complete four-second ramp and the first second of cycling.
The full input and run configuration are preserved.

```powershell
python -m gas_spring.cli --physical config/physical.yaml --run config/run_5s.yaml --simulate --diagnostics
python examples/benchmark.py --label timing
```

The first command writes `examples/output_5s.parquet`. The benchmark writes
separate result, JSON timing and optional profile files under this directory.
Wall time covers initialization, integration, sampled outputs and sensors;
it excludes imports, configuration/input loading and writing the result file.
CPU time is recorded separately to expose pauses or suspension. Runs were
sequential, without concurrent simulation benchmarks.

### Timing-report correction, 16 November 2026

The original CLI diagnostic `X s solver wall time` was the integration time
for **one window**, not cumulative runtime. For example, an externally timed
five-second run took 61.37 s; its last two windows reported 27.8 and 22.8 s,
all integration windows summed to about 54.5 s, and imports/loading, sample
processing and writing occupied the remainder. Reading the final window as
the total substantially understates elapsed time.

The CLI now labels each window, displays cumulative integration and total
elapsed time, and prints a final breakdown including loading through the
saved output. In a separate verification run, that final total was **17.80 s**
against **17.95 s** measured externally around the Python process. The 0.15 s
difference is interpreter startup/shutdown and reporting precision. These are
different runs under different machine conditions; this change corrects time
reporting and does not change the integration method or physical equations.
The simulation result matched the previous saved output exactly.

The benchmark now also reports `total_elapsed_wall_s`, covering imports,
preparation, simulation, result writing and report preparation. Its historical
`wall_time_s` field retains its original simulation-only meaning and is explicitly
labeled; `simulation_wall_time_s` is the clearer equivalent. Writing the JSON
timing report itself and interpreter shutdown occur after its elapsed snapshot.
Regression checks cover deliberately delayed loading, simulation and output
writing, with and without `--diagnostics`. All 74 tests pass.

## Result and implementation

| Five-second run | Wall time | CPU time | Accepted steps | Jacobian requests |
| --- | ---: | ---: | ---: | ---: |
| Original implementation (`baseline.json`) | 43.71 s | 42.94 s | 658 | 385 |
| Analytic expansion-source derivatives, original Jacobian cache | 38.21 s | 37.34 s | 609 | 351 |
| Fresh Jacobians, stable gas pressure differences, Radau | 54.93 s | 53.77 s | 421 | 155 |
| BDF, before removing repeated copies/input validation | 21.11 s | 20.91 s | 490 | 78 |
| Final implementation (`final.json`) | **18.99 s** | **18.80 s** | **490** | **78** |

The final implementation is **2.30 times faster** than the original short run,
with unchanged physical parameters, resolution and solver tolerances. The CLI
command above was also run successfully and produced all 1001 output rows.
This does not establish the new runtime for the full 19-second sweep: its later
cycles are faster and more demanding, so five-second timing should not be
extrapolated linearly. No full-duration timing run was performed in this work.

Changes retained:

- BDF replaces Radau. On this problem it requires substantially fewer Newton
  iterations and Jacobian refreshes. The custom finite-difference Jacobian is
  refreshed whenever the integrator requests it; the old cache could return
  the same stale matrix after Newton convergence problems.
- Thermal expansion-source accelerations are differentiated from heat-transfer
  and dissipation equations. This replaces nested finite differences of future
  circuit states and repeated thermal/gas evaluations. Wetted-area changes,
  two-zone temperature rates, viscosity changes and friction heat are included.
- The low-pressure gas network retains its small pressure offsets directly in
  the Darcy flow calculation, avoiding loss of precision when subtracting two
  absolute pressures near 1.6 MPa.
- Input channels are copied and validated once per simulation. Each RHS call
  interpolates these arrays. State reconstruction copies only mutable containers
  instead of recursively copying the entire state hierarchy.
- Diagnostics now count **all** RHS evaluations, including those inside numerical
  Jacobians. SciPy's `nfev` alone omits that cost.

The liquid remains incompressible, the main piston remains constrained by liquid
volume, wheel inertia is retained, and end-stop stiffness remains 10 kN/mm.
No physical parameter was changed in either committed run configuration.

## What costs the time

The original call profile made 223 million Python calls. Repeated thermal-state
assembly occupied 56.6 of 111.1 profiled seconds, numerical-Jacobian construction
31.0 s, and recursive copying 15.1 s. These are overlapping cumulative times,
not additive categories. Instrumentation more than doubled runtime; the timing
table uses unprofiled measurements.

The final solve reports 1447 ordinary RHS evaluations but **7843 actual RHS
evaluations**, of which **6396 (82%) construct Jacobians**. There are 81
integration variables. The remaining primary computational target is therefore
Jacobian construction and convergence, rather than output-file writing.
Most integration time is in the 4–5 s cycling portion (15.3 solver seconds);
the initial ramp is cheap.

The gas representation also introduces a severe scale separation. With the
provisional 50 mm × 4 mm actuator gas connection, 1 mL junction and initial
pressure, the laminar conductance is about 7.06e-6 m³/(Pa·s). A two-node,
adiabatic linearization gives

`tau = 1 / (gamma * Pabs * conductance * (1/Vactuator + 1/Vjunction))`

or approximately **61 ns**. The sound transit time through that same connection
is approximately **0.14 ms**. The quasi-steady gas links omit inertia, so their
fast relaxation is a consequence of the representation, not evidence of a
real mechanical vibration at that timescale. This is a stiffness candidate,
not a claim that it alone accounts for all runtime. The diagnostic startup
prints this estimate for the configured dimensions.

At the tested operating conditions, the maximum actuator-to-reservoir gas
pressure difference is only about **189 Pa (0.00189 bar)**, against roughly
15 bar gauge. The model is numerically resolving very small pressure
differences alongside much slower mechanical and thermal motion.

## Parameter and resolution experiments

Each experiment changes just the listed setting relative to the final physical
model. Timing noise of a few percent is normal. Changes below were diagnostic
only and were **not adopted**. JSON files record every tested configuration.

| Experiment | Wall / CPU time | Interpretation |
| --- | ---: | --- |
| Friction transition 0.005 → 0.020 m/s | 20.47 / 20.22 s | No speed gain; changes piston travel by up to 0.432 mm and pressure by 2.49 bar. |
| Gas junction 1 → 10 mL | 20.37 / 19.33 s | No useful speed gain; changes gas storage and temperatures. |
| Blow-off piston mass 0.1 → 1 kg | 19.39 / 19.00 s | No useful speed gain. This also changes its thermal capacity. |
| End-stop stiffness 10 → 1 kN/mm | 22.96 / 22.70 s | Slower; changes travel by up to 5.87 mm. The specified stiffness should be retained. |
| Pipe cells 3 → 1 | 14.47 / 14.27 s | Faster, but changes chamber/junction pressures by up to 0.0251 bar, actuator gas temperature by 1.79 °C and junction gas temperature by 6.40 °C. |
| SciPy automatic Jacobian with BDF | 18.97 / 18.80 s | Comparable runtime; no reason to discard the physical perturbation scales. |

Pipe-cell temperatures in different meshes represent different spatial locations
and are not directly compared in the table. The one-cell model is useful for
exploration, but its thermal resolution needs a convergence study before using
it in place of the three-cell default. Increasing the guessed junction volume
by tenfold did not fix runtime, despite reducing its fastest relaxation rate.
These tests do not support arbitrary physical-parameter tuning for speed.

## Accuracy and checks

- All **72 tests pass**, including analytic heat/loss derivative comparisons
  against directional finite differences, state-copy isolation and pressure
  differences smaller than absolute-pressure roundoff.
- Compared with the original five-second output, the final maximum differences
  across piston positions, pressure nodes and temperature nodes are respectively
  **0.49 µm**, **0.00269 bar** and **0.00620 °C**.
- A separate five-second run reduced relative tolerance from 1e-5 to 1e-6,
  reduced every absolute tolerance tenfold and halved the maximum step to
  0.025 s. Differences from the default final run were at most **0.54 µm**,
  **0.00289 bar** and **0.00636 °C**. It took 48.66 s wall / 47.91 s CPU.
- All final output values are finite. Maximum liquid-volume constraint residual
  is 3.12e-21 m³; liquid mass drift is −2.78e-17 kg and nitrogen mass drift is
  zero at reported precision. No stop contact/separation events occur in this
  five-second profile; the preloaded blow-off stop remains modeled.
- The sampled energy-audit residual is at most 0.134 J. This audit uses 5 ms
  trapezoidal integration and retains the existing thermal-expansion work
  approximation, so it is not an exact integration-error measure.

The Radau automatic-Jacobian experiment (`auto_jacobian.json`) recorded 7031 s
wall but only 66.75 s CPU, indicating a long pause/suspension. Its wall time is
excluded from comparisons. The uninterrupted baseline and final measurements
have closely agreeing CPU and wall times.

## Further work with the strongest justification

1. Supply an analytic or hybrid Jacobian, starting with gas-network fluxes and
   constrained mechanics. Numerical-Jacobian evaluations currently account for
   82% of RHS calls. Preserve the coupled thermal derivatives and verify against
   the existing finite-difference Jacobian before changing the default.
2. Investigate eliminating the very fast gas pressure-difference modes through
   an algebraic gas-network formulation. Retain gas transport, separate thermal
   states, conservation and any pressure losses needed by measurements. Adding
   gas connection inertia is another physical representation to assess if
   acoustic transients matter; neither approach adds liquid compliance.
3. Measure the short gas-connection bore/length and junction volume, together
   with friction near zero speed, blow-off connection dimensions and restriction
   curves. These affect the equations strongly. The tests above provide no
   justification for changing their values just to make the solver faster.
4. Compare three and finer pipe meshes if temperature prediction is important.
   The one-cell experiment shows that a smaller mesh changes thermal results
   enough to require deliberate validation.
