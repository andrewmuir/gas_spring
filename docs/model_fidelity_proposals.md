# Four reduced models for the gas spring system

The [current solver](../README.md#model-and-limitations) is the reference for
comparison. The four proposals below are ordered from **highest fidelity (1)**
to **lowest fidelity (4)**. Each is a distinct model. The simplest predicts
equilibrium loads and travel; the others retain enough dynamics for different
types of time history. The purpose is to remove effects that do not matter for
the question being asked and measure the resulting speed and accuracy.

| Level | Thermal model | Mechanical model | Best use |
| --- | --- | --- | --- |
| **1. Lumped thermal dynamics** | Three bulk gas temperatures, one liquid temperature and two representative wall temperatures | All piston and hydraulic inertia; nonlinear restrictions and seal friction | Warm-up, repeated laps and precharge drift |
| **2. Polytropic dynamics** | Three pressure–volume laws; fixed liquid temperature and properties | Same mechanical and hydraulic dynamics as Level 1 | First choice for lap sweeps, pressure peaks and dynamic response |
| **3. Heave and roll dynamics** | Same gas laws and fixed liquid properties as Level 2 | Actuator inertia retained; overdamped blow-off; no pipe or accumulator piston inertia | Faster lap comparisons when pipe and accumulator transients are small |
| **4. Quasi-static equilibrium** | Same gas laws and fixed liquid properties as Level 3 | Static piston force balance; no flow losses or inertia | Screening aero, precharge, roll stiffness and available stroke |

## Physics shared by all four

- **Incompressible liquid.** Solve positions with one exact volume constraint.
  With positive actuator travel defined as extension, positive main piston travel
  expanding its gas, and positive blow-off travel compressing its gas:

  ```text
  -A_act * (x_A + x_B) - A_main * x_main + A_bo_liq * x_bo = ΔV_thermal
  ```

  Here `A_act` and `A_bo_liq` are the annular liquid areas, and
  `ΔV_thermal` is the change in liquid volume relative to initialization due
  to temperature. Level 1 calculates it from the bulk liquid temperature;
  Levels 2–4 hold liquid temperature fixed, so it is zero. In the dynamic
  models, reconstruct the main piston from this constraint. Its motion follows
  the actuators, the blow-off piston and, in Level 1, thermal expansion.

- **Heave and roll.** The mean actuator load drives the common travel (heave).
  Unequal loads also drive a relative travel (roll). Use the existing roll
  spring with reactions `-k_roll * (x_A - x_B)` on A and the opposite on B.
  The configured **212,500 N/m** balances an 8,500 N input difference at
  20 mm relative travel (±10 mm about the mean) in static equilibrium.
- **Three separate gas volumes.** Treat the connected low-side gas as one
  pressure and use its combined gas volume. The main and blow-off nitrogen
  charges have separate pressures and fixed gas mass. Gas laws use absolute
  pressure. The main gas pressure rises without a cap as the compression
  coordinate approaches zero gas volume. Retain the configured main stop on
  the expansion side.
- **Blow-off stroke and pressure.** Keep its separate gas and liquid piston
  areas, finite stroke, lower-stop reaction and gas volume reduction
  `V_bo = V_bo0 - A_bo_gas * x_bo`. In static equilibrium off the stop and
  without seal friction, the local liquid gauge pressure balances
  `p_bo_gas,gauge * A_bo_gas / A_bo_liq`. With the current geometry and initial
  gas pressure, this is about **75.4 bar liquid at 70 bar gas**, or **70.0 bar
  liquid at 65 bar gas**. Actual first lift in the dynamic models also depends
  on local losses, piston motion, seal friction and the changing gas pressure.
  The gas pressure rises as fluid displaces the piston and compresses the
  nitrogen; the actuator pressure is measured elsewhere in the circuit.
- **Travel limits.** Keep the existing **10 kN/mm** unilateral end stops,
  including the preloaded lower blow-off stop in the common zero-position
  convention. Reject nonpositive chamber volumes. Flag any operating point
  requiring nonpositive absolute liquid pressure as outside the single-phase
  model's domain.
- **Comparable inputs and outputs.** Use the existing force and environment
  inputs and the same initialization, including a preload ramp matched to the
  selected starting pressures. Keep the cheap input-shaft deflection
  correction `F / k_shaft` for sensor comparisons. Apply virtual sensor
  noise and calibration after the physics solve. Label outputs by fidelity and
  omit channels that a level does not predict.

## 1. Lumped thermal dynamics

**Retain.** Keep the actuator and attached wheel masses, both accumulator
piston masses, incompressible pipe inertance, the individual actuator and main
inlet restrictions, and pressure-dependent smooth seal friction. Retain
compression work and heat exchange for each of the three gas volumes, including
the shared low side.

**Simplify.** Replace the axial pipe and wall cells, chamber boundary layers
and local transport with six bulk thermal states: three gas temperatures, one
liquid temperature, a hot-side wall temperature and a cooler-side wall
temperature. Initialize the bulk temperatures from the corresponding total
energies. Use effective conductances to exchange heat between nodes and with the
environment. Put the restriction and pipe losses into the liquid and the seal
losses into the corresponding wall. Use one instantaneous liquid temperature
for expansion and viscosity; retain the distinct hydraulic branch laws with
their respective flows. Use a lumped hydraulic inertance on each pipe where the
reference retains it. This removes the local mass and enthalpy transport
calculations and most of the thermal Jacobian.

**Solve and use.** Integrate three independent piston positions and velocities
plus six thermal states (12 states total), with the main piston reconstructed
algebraically.
This is the candidate for successive laps or temperature-driven changes in
stroke and precharge. It can predict bulk temperature drift; it cannot resolve
sensor locations that depend on local wall or liquid gradients. Fit the bulk
heat-transfer coefficients to warm-up and cool-down data where available.

## 2. Polytropic dynamics

**Retain.** Keep Level 1's mechanical model, pipe inertance, calibrated
directional restriction curves, pressure-dependent seal friction and end stops.

**Simplify.** Fix liquid properties at a representative temperature. Drop
the heat balance and thermal expansion. Replace each gas's temperature
equation with `p_abs = p0_abs * (V0 / V)^n`, using the combined volume of the
low side and separate laws for the main and blow-off chambers. Fit an effective
exponent for each pressure range and forcing regime. For ideal nitrogen,
`n = 1` (isothermal) and approximately `n = 1.4` (adiabatic) are useful
starting cases for sensitivity checks.
[HYDAC accumulator sizing guide](https://betterfit-us.hydac.com/wp-content/uploads/AOSS-%20accumulators.pdf)

**Solve and use.** Integrate six mechanical states. Solve liquid pressure and
main piston acceleration under the volume constraint. Most of the reference
solver's thermal states and coupled iterations are gone. This is the first
candidate for short bench tests and full-lap aero/precharge sweeps with stable
temperature. It can report pressure, travel, phase and hydraulic dissipation,
but cannot predict warm-up or thermal drift. Check the equilibrium and dynamic
gas stiffness against the reference before relying on a single exponent over
the full frequency range.

## 3. Heave and roll dynamics

**Retain.** Keep the two actuator masses, gas springs, roll spring, stops
and separate actuator and main inlet restrictions. Those restrictions still
allow the fluid pressures at the actuators and accumulators to differ.

**Simplify.** Drop the inertia of the liquid columns and the main and blow-off
pistons. Use fixed-temperature liquid and polytropic gas as in Level 2. Keep
positive, direction-dependent flow resistance on each important path. A simple
monotone loss curve can replace the detailed curve if it reproduces the
calibrated drops over the relevant flows. Approximate seal friction as a
smooth, constant-limit force fitted per piston; calibrate the blow-off seal
separately.

**Solve and use.** Use the mean and half-difference of actuator displacements
and their velocities (four states), plus blow-off travel (one state). For a
trial blow-off rate, the volume constraint gives main piston rate and the
branch laws give the local pressures. Solve the massless main and blow-off
force balances for junction pressure and blow-off rate. The remaining branch
resistance and seal friction govern its opening and closing speed. This model
can capture heave and roll lags and pressure drops without resolving hydraulic
inertial peaks or piston bounce in either accumulator. Use it for lap work
only after comparing it with Level 2 through the fastest relevant inputs.

## 4. Quasi-static equilibrium

**Retain.** Keep the three nonlinear gas springs, the exact liquid volume
constraint, roll stiffness and static stop reactions. Keep unequal areas on the
blow-off piston, so a change in precharge still changes the opening threshold.

**Simplify.** Drop all inertia, flow pressure drops and velocity-dependent
friction. Use one equilibrium liquid pressure for the circuit. The gas pressures
and spring forces still depend on position and the inputs retain independent
left and right loads. Opening is determined by static piston force balance;
once off the stop, extra fluid displaces the piston and compresses its gas.

**Solve and use.** Solve piston force and liquid-volume residuals for each
load sample, seeded by the prior sample. Bound the solve to positive chamber
volumes. If a stop is engaged, allow its elastic deflection and check that
the reaction acts in the correct direction.
With neither actuator at a stop, `x_A - x_B = (F_A - F_B) / (2*k_roll)`.
There are no time integration states. Use this for quick screening of travel,
balanced pressure, and whether slow loads are sufficient to open the blow-off.
Dynamic peaks, phase, restriction losses and frictional hysteresis require a
higher level. Contact time inferred from a static sweep is only an estimate.

## How to choose a level

Use the current solver as a common reference and change only the fidelity
level between comparisons. Repeat with hardware measurements when available.

1. **Check the common balances.** Test zero load increment, equal actuator
   loads and an 8,500 N differential load. Check equilibrium, liquid-volume
   conservation, force signs, roll stiffness and stop reactions. For dynamic
   models check that restrictions dissipate energy in both flow directions.
2. **Check bandwidth and damping.** Run the existing 1–4 Hz bench sweep and
   controlled heave and roll inputs through 25 Hz. Compare amplitudes, phases,
   travel and peak actuator pressures. Use the [restriction calibration
   targets](../examples/benchmarks/restriction_calibration.md) of about 1 bar
   per actuator and 4 bar at the main inlet **at the corresponding flows**.
   These are not fixed pressure drops. A lap interpolated to 100 Hz does not
   supply original 25 Hz force detail.
3. **Check blow-off behaviour.** Use the same Barcelona +10% aero input with
   65 and 70 bar precharge. Compare local blow-off liquid pressure, first
   lift, peak travel, and gas pressure rise above precharge. The reference
   65 bar run opened earlier and moved farther; the 70 bar case had only
   about 0.24 bar rise above precharge, so compare noise-free outputs first.
   Recheck any case where a reduction changes whether the piston reaches a
   stop or whether it opens at all.
4. **Check thermal scope and speed.** Compare warm-up and repeated-lap drift
   before using Level 1 for thermal results. Record end-to-end wall time and
   integration time for each level on the same inputs. Set acceptance
   limits for pressure, travel and event timing based on the engineering
   decision. Sensor noise is useful context, but it does not set the acceptable
   error at a blow-off threshold.

**Recommended order:** implement Level 2 first for the batch harness. It makes
the largest clear reduction in thermal work while keeping the mechanics needed
to interpret pressure and displacement. Use Level 4 to screen a parameter
matrix quickly, then rerun near thresholds with Level 2. Add Level 1 when
temperature history matters and Level 3 if batch timings still justify
reducing the hydraulic dynamics.
