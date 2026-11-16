# Solver options — 16 November 2026

This is a historical comparison from before the
[gas reduction](gas_reduction.md) and subsequent
[restriction/noise calibration](restriction_calibration.md). Its timings and
parameter references describe that earlier model configuration.

The production solver is already **BDF**, following the earlier comparison
with Radau. The user's approximately 700-second full-run measurement is not
replaced by a new full-duration measurement here. These are short experiments
to select promising work without repeating the full 19-second sweep.

## Controlled short comparisons

All tests use the first five seconds of the input and the same physical
parameters, three pipe cells, six actuator wall cells and 0.05 s maximum step.
`config/run.yaml` now specifies **70 bar blow-off gas pressure**, while the
short-run configuration still specifies 80 bar. Each test therefore explicitly
overrides the short-run value to **70 bar**. Neither YAML file was edited.
The later gas-reduction work synchronized the short configuration to 70 bar;
the paragraph above records the configuration at the time of these comparisons.

The installed numerical library is SciPy 1.15.3. Experiments were sequential;
each process had a 70-second external timeout. Source hashes and settings are
recorded in each benchmark JSON. Process outcomes and externally measured
durations are in `solver_comparison_runs.json`.

| Test | Simulation + sampled output | CPU for simulation | Whole command, external timer |
| --- | ---: | ---: | ---: |
| BDF, current tolerances | 19.54 s | 19.14 s | 28.75 s |
| LSODA, physical-scale Jacobian | Not completed | Not recorded at termination | Stopped at 70.04 s |
| LSODA, automatic Jacobian | Not completed | Not recorded at termination | Stopped at 70.03 s |
| BDF, one continuous solve across 5 s | 20.48 s | 20.19 s | 21.87 s |
| BDF, tolerances relaxed tenfold | 14.52 s | 14.36 s | 16.05 s |

**Timing scope matters:** the first numeric column excludes imports, loading
and file writing. The BDF baseline command spent about 9.1 s outside
`simulate()`, versus about 1.3–1.4 s in the other completed experiments.
The simulation CPU and wall times are used to compare numerical work. The
smaller whole-command time for the continuous solve is not an integration
speed improvement.

LSODA spent 50.28 s integrating 4.0–4.5 s with the supplied Jacobian, and
47.95 s on the same window with its automatic Jacobian. BDF spent 8.65 s on
that window. Both LSODA experiments were terminated during the final window.
LSODA is an Adams/BDF solver that automatically detects stiffness; it is not
a faster replacement for this particular system in these tests.
[SciPy LSODA documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.LSODA.html)

The continuous solve reduced accepted steps from 614 to 516 but made the same
90 Jacobian requests. It did not improve numerical runtime. Removing progress
checkpoints alone is therefore not supported as a speed fix by this experiment.
A persistent integrator that preserves history while reporting progress could
still be assessed later, but there is no measured gain to claim now.

## Tolerance tradeoff

The relaxed run used relative tolerance 1e-4 instead of 1e-5 and multiplied
every absolute tolerance by ten. It was approximately **1.35 times faster**
within `simulate()`. Relative to the current-tolerance BDF run, maximum
differences across all piston/node channels were:

- Position: **2.00 µm**.
- Pressure: **0.114 bar**.
- Temperature: **0.0187 °C**.

These are differences from the baseline, not proven absolute errors. A result
accuracy budget and a tighter reference run are needed before adopting the
relaxed settings. Later 4 Hz cycling and blow-off events may behave differently.
The default settings were retained. All completed experiments preserved liquid
volume and total mass to numerical precision and produced finite output.

## Recommended order of work

1. **Analytic or hybrid Jacobian.** The current 70-bar baseline used 9172 RHS
   calls, including 7380 calls to build numerical Jacobians (about 80%). The
   81-state Jacobian takes 82 complete model evaluations per refresh. Supplying
   derivatives of the coupled mechanics, gas transport and thermal constraint
   should directly reduce this cost and may improve Newton convergence. Sparsity
   should be derived from the actual dependencies before applying coloring;
   global pressure/thermal constraints couple otherwise local nodes.
2. **Reduce the very fast gas-pressure modes.** The earlier linearization gives
   roughly 61 ns actuator/junction pressure equalisation with the guessed short
   gas connection and junction volume. This is much faster than the modeled
   mechanical forcing. Assess a quasi-static gas pressure/flow network that
   retains separate gas thermal states, gas transport and necessary pressure
   losses. Validate its approximation against physical dimensions and measured
   pressure differences. If gas acoustic transients matter, assess gas-line
   inertia instead. The liquid remains incompressible and the main piston
   remains governed by the liquid-volume constraint.
3. **Compile the expensive RHS calculations.** Converting repeatedly evaluated
   dictionary-based thermal, flow and mechanics routines into an array-based
   compiled implementation is another route to speed without deliberately
   changing the equations. Profile after the Jacobian work to select the parts
   that still dominate. This requires implementation and validation; no speed
   factor has been measured for it.
4. **Assess native CVODE or IDA if the formulation warrants it.** CVODE supplies
   native stiff integration; IDA is relevant to a differential/algebraic
   reformulation. These require a new integration interface and careful
   treatment of events, tolerances and consistent initial conditions. Merely
   switching libraries leaves expensive Python RHS evaluations and artificial
   fast modes in place, so a large gain cannot be promised.
   [SUNDIALS CVODE introduction](https://sundials.readthedocs.io/en/latest/cvode/Introduction_link.html)
   [SUNDIALS IDA introduction](https://sundials.readthedocs.io/en/latest/ida/Introduction_link.html)

Explicit RK45/DOP853 is a poor fit while the extreme fast gas modes remain.
SciPy recommends implicit methods for stiff systems; an explicit solver's
stability restriction can demand tiny steps even when its accuracy tolerance
would allow larger ones.
[SciPy solver selection documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

The earlier Radau comparison at the old 80-bar initial setting took about
54.9 s versus about 19.0 s for optimized BDF. It is supporting historical
evidence, not a new 70-bar side-by-side measurement. Neither the 19-second
runtime nor a gain for any unimplemented option is inferred by multiplying
these five-second results: the remaining sweep reaches higher frequencies.

To reproduce a completed short comparison without editing the configuration:

```powershell
python examples/benchmark.py --label solver_test --method BDF --set run.initial_gas_gauge_pressure_bar.blowoff=70
```

`--method LSODA` is now accepted by the benchmark for controlled experiments;
the production solver and physical parameters are unchanged.
