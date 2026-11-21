# Current gas spring model

This document describes the implemented model, using [physical.yaml](../config/physical.yaml)
and [run.yaml](../config/run.yaml) as the reference settings. Other run YAMLs
can change the preload pressures, temperatures, roll stiffness, inputs and
solver controls. The [reduced models](model_fidelity_proposals.md) are proposals.

- [System layout and reference values](#1-what-the-model-represents)
- [Coordinates and piston forces](#2-coordinates-chamber-volumes-and-forces)
- [Liquid continuity and hydraulic losses](#3-incompressible-liquid-and-hydraulic-losses)
- [Nitrogen and heat transfer](#4-nitrogen-and-heat-transfer)
- [Initialization, states and integration](#5-initialization-and-numerical-solve)
- [Inputs, outputs and run command](#6-inputs-outputs-and-how-to-run)
- [Validity and calibration priorities](#7-validity-and-calibration-priorities)

## 1. What the model represents

Two load-driven hydraulic actuators share a high-pressure liquid circuit. The
main gas accumulator reacts to their combined volume displacement. A second,
higher-precharge accumulator accepts liquid when the pressure at its branch
is sufficient to lift its piston from its lower stop. A mechanical roll spring
reacts to the difference in travel between the two actuators.

The two actuator gas chambers share a separate low-pressure nitrogen circuit
with a remote reservoir. The main and blow-off gas chambers are sealed
individually. Liquid is incompressible, with temperature-dependent specific
volume. Gas can compress and expand. Heat transfer, flow loss, pipe inertia,
seal friction and elastic contact are included.

### Circuit layout

The lines below show connections. Flow can reverse.

```text
Liquid
Actuator A ─ restriction ─┐
                          ├─ inlet ── main pipe ── outlet ─┬─ restriction ─ Main piston
Actuator B ─ restriction ─┘                                └─ short pipe ── Blow-off piston

Nitrogen
Actuator A gas ─┐
                ├─ gas junction ── gas pipe ── Low-side reservoir
Actuator B gas ─┘

Main gas chamber    : sealed, acts on main piston
Blow-off gas chamber: sealed, acts on blow-off piston
```

Typical geometry and operating settings:

| Parameter | Reference value |
| --- | --- |
| Actuator piston / shaft diameter | 38 / 14 mm |
| Actuator liquid / gas chamber length at zero travel | 55 / 20 mm |
| Actuator first contact in compression / extension | −15 / +50 mm |
| Main piston diameter; gas volume at zero travel | 45 mm; 50 mL |
| Main first contact on expansion side | +7 mm |
| Blow-off piston / shaft diameter; gas volume at loaded zero | 45 / 12 mm; 1,000 mL |
| Blow-off upper stop | +50 mm |
| Hydraulic and low-side gas pipes | Each 1 m long, 4 mm bore |
| Short blow-off hydraulic connection | 50 mm long, 4 mm bore |
| Low-side gas reservoir | 150 mL |
| Initial low-side / main / blow-off nitrogen pressure | 15 / 35 / 70 bar gauge |
| Effective actuator moving mass | 15.1 kg each: 0.1 kg piston + 15 kg attached mass |
| Main / blow-off piston mass | 0.07 / 0.10 kg |

The 30 kg total attached mass represents the estimated wheel assembly inertia
split between the actuator coordinates. It contributes to kinetic energy and
piston dynamics. It does not add a thermal mass or a separate wheel state.

## 2. Coordinates, chamber volumes and forces

Lengths are in metres and forces in newtons in the solver. Internal gas and
liquid pressures are absolute in pascals. Piston force equations below use
gauge pressure, `p_gauge = p_abs − 101325 Pa`. Pressure outputs use bar gauge
unless the column explicitly says absolute pressure.

Let `A(D, d) = π(D² − d²)/4`. The gas area is the full piston area
`A(D, 0)`. Actuator liquid area `A_a` and blow-off liquid area `A_b` are
annular. The main piston has the same area `A_m` on both sides. Denote
the actuator and blow-off full gas areas by `G_a` and `G_b`.

| Positive coordinate | Change in liquid volume | Change in gas volume |
| --- | --- | --- |
| Actuator `x_A` or `x_B`: extension | `−A_a x` | `+G_a x` |
| Main `x_M`: gas expansion | `−A_m x_M` | `+A_m x_M` |
| Blow-off `x_O`: piston opening | `+A_b x_O` | `−G_b x_O` |

Chamber volume is its value at zero plus the listed change. The liquid pipe
cells, junctions and gas reservoir have fixed geometric volumes. Increasing
the applied actuator load drives extension and expels liquid from the actuator
chamber. Without compensating blow-off travel or thermal expansion, this
compresses the main gas.

### Piston force balance

In each positive coordinate, acceleration follows the signed force balance:

```text
m_a a_A = F_A + G_a p_low − A_a p_A − k_roll (x_A − x_B) + F_stop,A − F_seal,A
m_a a_B = F_B + G_a p_low − A_a p_B + k_roll (x_A − x_B) + F_stop,B − F_seal,B
m_M a_M = A_m (p_main_gas − p_main_liquid) + F_stop,M − F_seal,M
m_O a_O = A_b p_blowoff_liquid − G_b p_blowoff_gas + F_stop,O − F_seal,O
```

Here `p_A` and `p_B` are the actuator liquid chamber pressures.
`F_seal` is signed in the direction of velocity and is subtracted in the force
balance. Piston forces use the pressure at each chamber, which can differ from
the adjacent junction pressure when liquid is flowing.

- **Roll spring.** `k_roll = 212500 N/m` by default in the supplied run YAMLs.
  Equal travel produces no reaction. At 20 mm relative travel, the spring
  carries 4,250 N on each side in opposite directions. This balances an
  8,500 N left-right load difference with ±10 mm about the mean for a static
  free-travel case. Stored energy is `½ k_roll (x_A − x_B)²`.
- **Seal friction.** The static magnitude scale is
  `F_limit = F0 + c_seal [max(p_liquid,gauge, 0) + max(p_gas,gauge, 0)]`.
  The configured base forces for actuator, main and blow-off seals are 40,
  60 and 80 N respectively; `c_seal = 2e-5 N/Pa` for all three.
  The dynamic model uses `F_seal = r F_limit tanh(v/v_s)`; the standard run
  uses `v_s = 0.005 m/s` and the physical settings use `r = 1`. The force is
  zero at rest in the dynamic law, so a residual force can cause slow creep.
- **Elastic stops.** A stop reacts only on contact. Its force is
  `k_stop (δ_lower − δ_upper)` and its stored energy is
  `½ k_stop (δ_lower² + δ_upper²)`, with `k_stop = 10 kN/mm`. Small travel
  beyond first contact is elastic deflection. There is no instant velocity
  reset, added impact damping or heat pulse. The main has an expansion stop;
  on its compression side the ideal-gas pressure grows without a cap as the
  gas volume tends to zero. Nonpositive chamber volumes are rejected.

### What opens the blow-off

The blow-off opens as the lower stop unloads and the piston moves in the
positive direction. Its coordinate starts at the loaded reference.
Initialization calculates a positive lower-stop preload from the gas force
minus the initial liquid force, when the gas force is larger. The unloaded
contact surface is then `preload / k_stop` above zero. At the standard
35 bar hydraulic and 70 bar gas settings, that surface is about 0.596 mm
above the loaded reference.

For a free piston at rest, neglecting friction, the pressure-force balance
is `p_blowoff_liquid = (G_b/A_b) p_blowoff_gas` in gauge units. The area
ratio is about 1.077. A gas precharge of 70 bar corresponds to roughly
75.4 bar on the liquid side, and 65 bar corresponds to roughly 70.0 bar.
These are approximate force-balance reference levels. The actual opening
depends on local flow losses, piston acceleration, friction and gas
temperature. Once moving, each millimetre of piston travel compresses
about 1.59 mL of gas. The gas pressure therefore depends on the volume and
temperature change during the event, rather than tracking an upstream
actuator pressure directly.

## 3. Incompressible liquid and hydraulic losses

### Volume and mass conservation

The sole liquid storage relation is geometric volume plus temperature-driven
expansion. The configured pressure-independent density law is
`ρ(T) = ρ20 / [1 + α (T − 20 °C)]`, with `ρ20 = 1062 kg/m³` and
`α = 0.0007/K`. Liquid sensible energy is `U_i = m_i c_p (T_mean,i − 20 °C)`
with constant `c_p = 1500 J/(kg K)`. The linear specific-volume law makes
the total change in liquid volume proportional to the change in total liquid
sensible energy:

```text
β = α / (ρ20 c_p)
ΔV_thermal = β (Σ U_liquid − Σ U_liquid,initial)
−A_a (x_A + x_B) − A_m x_M + A_b x_O = ΔV_thermal
x_M = [−A_a (x_A + x_B) + A_b x_O − ΔV_thermal] / A_m
```

The solver reconstructs `x_M` and its velocity from this relation. It still
includes main piston inertia in the force and acceleration equations.

Each liquid node expands at local rate `S_i = β (Q_i + D_i)`, where
`Q_i` is its net thermal exchange and `D_i` its share of dissipated
hydraulic power. Continuity gives branch flow into a chamber as
`q_i = dV_geometric,i/dt − S_i`. For example, extension of A at
constant temperature gives negative branch flow, out of A's chamber.
Pipe flow is positive from inlet to outlet and varies across cells as
the liquid expands. Mass flow uses the upstream core density; the same
transported temperature gives the advected sensible energy. Each edge
removes exactly what it adds downstream. Near flow reversal the donor
properties are smoothly blended.

### Pressure drops and pipe inertia

- **Actuator and main inlet restrictions.** Each branch has separate curves
  for flow into and out of its chamber. The fitted magnitude is
  `|Δp| = P_limit |q|/(q_scale + |q|)`. The original force-speed points
  use the actuator annular area as the equivalent reference area for both
  valve types. The solver smoothly blends the two directional curves near
  zero flow and uses a tangent linear continuation beyond 95% of each
  fitted asymptote. The [calibration](../examples/benchmarks/restriction_calibration.md)
  targeted about 1 bar per actuator and 4 bar at the main inlet at the
  terminal test flows. Other flows give other losses.
- **Shared hydraulic pipe and short blow-off branch.** Pressure difference
  includes a signed Darcy loss plus `I dq/dt`, with
  `I = ρ20 L/A_pipe`. The Darcy friction factor blends laminar and
  turbulent estimates through the transition region. The pipe loss uses
  viscosity based on the average of adjacent core temperatures. The
  viscosity law is log linear between 0.010 Pa·s at 20 °C and
  0.0021 Pa·s at 100 °C and holds those values outside that range.
  Pipe inertia stores energy; the Darcy term dissipates it.

The solver obtains a common reference pressure at the outlet junction and
computes all chamber pressures from the path differences. Differentiating
continuity gives flow accelerations in terms of piston accelerations and
changing thermal sources. The result is an algebraic solve for four piston
accelerations and the outlet pressure. There are no liquid pressure or flow
integration states.

## 4. Nitrogen and heat transfer

Nitrogen uses ideal-gas properties with `R = 296.8` and
`c_p = 1040 J/(kg K)`, so `c_v = c_p − R = 743.2 J/(kg K)`. For a nitrogen
node of total internal energy `U`, mass `m` and volume `V`:

```text
p_abs = (R/c_v) U/V
T_mean,K = U/(m c_v)
```

Each sealed accumulator evolves its total gas energy by
`dU/dt = Q_gas − p_abs dV/dt`. Compression adds energy to the gas.
The connected low side instead uses one total energy and one common pressure,
with `U_i = p_abs V_i/(R/c_v)` at each connected node. The sum of node
heat rates and compression work determines the total energy rate. Each
node's energy balance then gives its required net enthalpy inflow, and the
tree of connections gives the individual edge fluxes. Mass flux is recovered
using the donor's core enthalpy, with smooth blending near reversal. Total
low-side gas mass is fixed; reservoir mass is reconstructed from what remains
after accounting for the other low-side nodes. The nodes can have different
temperatures even at the same pressure. Estimated gas Darcy loss and Mach
number are diagnostic only.

### Thermal resolution and heat paths

| Part | Implemented thermal representation |
| --- | --- |
| Four liquid chambers and five gas chambers, including the reservoir | Core and near-wall zones. Each chamber has one pressure, a configured 20% boundary mass fraction by default, and a separate boundary energy. Bulk transport uses the core, with internal exchange to maintain the boundary fraction. |
| Main hydraulic and gas pipes | One medium temperature and one wall temperature per axial cell. Includes axial conduction in the medium and wall, transport in the medium, and heat exchange between pipe wall and its surroundings. |
| Actuator housings | Axial metal cells. The portion of each cell wetted by liquid or gas varies with piston position. Each housing receives heat from the hot environment. |
| Four pistons | Two thermal faces with a through-piston conductance. Each face exchanges heat with its adjacent boundary layer. Seal friction power is split between faces. |
| Accumulator and reservoir housings | Lumped metal temperatures with heat exchange to the cooler region. Accumulator wetted wall areas follow chamber volumes using the bore geometry. |
| Cooler region | Finite thermal capacity. Conductance to the supply air is `UA = UA_base + k_air · duct_air_speed_m_s`. The supply air and the pipe surroundings are separate prescribed temperatures. |

Internal heat links use conductance times temperature difference and act with
equal and opposite signs on the two nodes. Restriction and Darcy dissipation
`q Δp_loss` are added to the liquid at the two ends of each link. Dissipation
feeds thermal expansion; seal heating changes the piston face temperatures.
The reference run starts the actuator chambers and metal at 50 °C, the
remote chambers and metal at 25 °C, and uses 120 °C hot ambient. Pipe cell
temperatures start on linear profiles between the connected end temperatures;
junctions use the configured chamber temperatures or their means.

## 5. Initialization and numerical solve

1. **Initialize at zero travel.** All piston positions and velocities start at
   zero. The static initializer calculates the range of common liquid gauge
   pressures that can support the first input forces using the seal limits
   and the blow-off lower stop. It takes the main gas precharge if
   feasible, otherwise the closest feasible pressure. If the interval is
   empty, initialization fails. The dynamic smooth seal law can still move
   away from a state held only by the static feasibility allowance.
2. **Set mass and energy.** Compute initial volumes from geometry, liquid
   mass from temperature-dependent density, and nitrogen mass from the
   prescribed pressure and temperature. Initialize the boundary energy and
   the solid temperatures. The nominal force per actuator that balances the
   standard initial pressures with zero seal force is
   `A_a p_main − G_a p_low ≈ 1729.45 N`.
3. **Advance the independent states.** Each derivative evaluation reconstructs
   the main coordinate and the common low-side gas state. A local iteration
   couples expansion, flow and dissipation. The solver differentiates the
   thermal expansion sources within the acceleration balance and iterates the
   coupling with seal heat.
4. **Integrate and sample.** The reference solve uses adaptive implicit BDF
   with a block numerical Jacobian, relative tolerance `1e-5` and maximum
   internal step `0.05 s`. Absolute tolerances are set by state type:
   `1e-9 m` for travel, `1e-7 m/s` for velocity, `1e-12 kg` for mass,
   `1e-5 J` for energy, and `1e-5 °C` for solid temperature. The Newton
   matrix reuses parts of the gas and
   thermal calculation and holds weak coefficient variations fixed during
   numerical differentiation; the physical rate evaluation still uses the
   coupled equations. The CLI advances in 0.5 s reporting windows by default.
   Contact and release events are located within the solve. If a trial
   evaluates an invalid chamber volume, the window is retried from its
   checkpoint with a smaller step limit; repeated failure stops the solve.
   Dense output samples the solution at the input timestamps.

With three cells per pipe and six wall cells per actuator, the solver has
**72 integrated states**:

| State group | Count | Contents |
| --- | ---: | --- |
| Mechanics | 6 | A, B and blow-off positions and velocities |
| Liquid | 22 | Mass and total sensible energy at nine nodes; four chamber boundary energies |
| Nitrogen | 14 | Six independent low-side node masses; one low-side total energy; two sealed gas energies; five boundary energies |
| Solids and cooler | 30 | Twelve actuator wall temperatures, six pipe wall temperatures, eight piston face temperatures, three remote housing temperatures and one cooler temperature |

## 6. Inputs, outputs and how to run

```powershell
python -m gas_spring.cli --physical config/physical.yaml --run config/run.yaml --simulate --diagnostics
```

[The run YAML](../config/run.yaml) selects the input and output Parquets,
initial pressures and temperatures, stop travel, roll stiffness, solver
controls and sensor settings. Its file paths are relative to the YAML.
`--initialize` checks startup feasibility without integrating unless
`--simulate` is also supplied. The minimum input columns are:

| Column | Meaning |
| --- | --- |
| `time_s` | Finite, strictly increasing times, with at least two samples |
| `actuator_a_force_n`, `actuator_b_force_n` | Applied actuator forces, positive in extension |
| `duct_air_speed_m_s` | Nonnegative duct air speed used by the cooler model |
| `hot_ambient_c` | Hot-side ambient; required as an input series when the run does not select a constant. The run can name a different input column. |

Loads and environmental inputs are linearly interpolated between samples.
The bench input ramps from static balance to 5,000 N per actuator over 4 s,
then sweeps 1–4 Hz for 15 s at ±3,000 N. The [lap converter](../gas_spring/lap_input.py)
is a separate input generator: it converts speed and acceleration into wheel
loads using static axle load, longitudinal and lateral transfer, aero
downforce, and speed-dependent front balance. The selected axle's left and
right wheel loads are divided by the configured actuator-travel/wheel-travel
ratios to get the forces for A and B. It can prepend a preload ramp and
interpolates the resulting profiles to 100 Hz. Neither the lap converter nor
the prescribed load input responds to the solved piston motion.

### Result channels and interpretation

- **Model truth (`truth_*`).** Includes local gas and liquid pressures,
  core and boundary temperatures, volumes, masses, piston positions and
  velocities, input forces, stop and roll reactions, flow losses and energy.
  Actuator displacement in mm is the input shaft coordinate
  `x_piston + F_applied/k_shaft`, with `k_shaft = 20 MN/m`. Shaft velocity
  adds `(dF_applied/dt)/k_shaft` to piston velocity. The force slope is
  right-sided at interior sample knots and left-sided at the final sample.
  Shaft stretch is an algebraic correction; it is not another moving state.
- **Virtual sensors (`sensor_*`).** Apply gain, offset and applicable
  temperature compensation to truth, then add seeded drift and white noise
  per channel. Default noise standard deviations are 0.2 bar pressure,
  0.05 °C temperature, 0.3 mm displacement and 25 N force. The low-side
  pressure and temperature sensors refer to the reservoir. Calibration and
  noise are applied after integration and cannot change the physics.
- **Conservation diagnostics.** Liquid-volume and instantaneous continuity
  residuals track constraint enforcement. The mass inventory sums liquid and
  nitrogen separately. The energy audit includes internal, solid, piston and
  hydraulic kinetic energy and elastic stop and roll energy. External power
  includes heat exchange, applied force times piston velocity, and atmospheric
  work on the model's net moving volume. Shaft strain energy is reported
  separately. The sampled balance residual uses trapezoidal integration of
  external power at the saved timestamps, so sparse sampling can inflate it.
  The liquid thermal model omits the pressure-work correction to sensible heat;
  `truth_thermal_expansion_work_w` exposes the associated mechanical power
  `Σ p_abs,i S_i` for interpreting the residual.
- **Solve metadata.** The Parquet records accepted steps, solve counters,
  window timings, step retries, settings and stop events. The CLI also saves
  the run name and input path. The dashboard uses those links to open both
  files from a named run. The [batch harness](../config/batch_example.yaml)
  can vary inputs and run settings and reuse physical results for sensor-only
  variants.

## 7. Validity and calibration priorities

- **Liquid pressure:** the solver does not add liquid compressibility, a
  bulk modulus, pressure-wave propagation or cavitation. Unclamped algebraic
  pressure can expose an invalid single-phase result. A nonpositive minimum
  absolute pressure means this model requires liquid tension.
- **Gas pressure:** the common low-side pressure approximation removes gas
  inertia, acoustic modes and pressure losses from the dynamics. The closed
  gas equations include compression and heat exchange but use constant
  ideal-gas properties. The pressure estimates warrant calibration against
  measurements, especially for the high-precharge blow-off.
- **Geometry and damping:** the actual connected blow-off gas volume, seal
  friction, restriction curves and short connecting-pipe sizes strongly affect
  how far the piston lifts and how quickly its gas pressure rises. The
  configured values and the high-flow restriction continuation are provisional.
  The wetted area estimate assumes the configured volume is in the bore; a
  remote gas volume would require a different area model.
- **Thermal coefficients and forcing:** heat-transfer coefficients and some
  material data are engineering estimates. The CLI's isolated thermal
  estimate is separate from the coupled simulation. Lap loads are estimated
  from telemetry; interpolation to 100 Hz adds samples, not measured
  high-frequency content. Use dedicated bench inputs to evaluate fast dynamics.
- **Numerical comparison:** compare pressure peaks, travel, stop timing and
  thermal drift under finer solver tolerances and spatial resolution. The
  [restriction calibration](../examples/benchmarks/restriction_calibration.md)
  and [gas reduction comparison](../examples/benchmarks/gas_reduction.md)
  document engineering checks against prior configurations.

Core implementation: [mechanics](../gas_spring/mechanics.py),
[hydraulic balances](../gas_spring/incompressible.py),
[coupled rates](../gas_spring/coupling.py),
[gas network](../gas_spring/gas_network.py),
[thermal network](../gas_spring/thermal_coupling.py),
[initialization](../gas_spring/initialization.py) and
[time integration](../gas_spring/solver.py).
