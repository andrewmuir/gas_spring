"""Incompressible thermal transport and constrained piston dynamics."""

import numpy as np
from .gas_network import GasNetwork
from .numerics import VOLUME_FLOW_TRANSITION_M3_S, MASS_FLOW_TRANSITION_KG_S
from .hydraulics import (
    piston_area,
    DigressiveCurve,
    regularized_restriction_pressure_drop,
    pipe_pressure_drop,
    restriction_power_slope,
    pipe_power_derivatives,
)
from .materials import liquid_temperatures, liquid_density, liquid_viscosity
from .incompressible import (
    liquid_piston_areas,
    hydraulic_flows_m3_s,
    constrained_accelerations,
)
from .thermal_coupling import thermal_exchange, liquid_heat_rate_derivative
from .mechanics import end_stop_response, roll_spring_stiffness_n_m
from .materials import ATMOSPHERIC_PRESSURE_PA

PISTONS = ("actuator_a", "actuator_b", "main_accumulator", "blowoff")
LIQUID_PISTONS = dict(
    zip(
        PISTONS, ("actuator_a_fluid", "actuator_b_fluid", "main_fluid", "blowoff_fluid")
    )
)
GAS_PISTONS = dict(
    zip(PISTONS, ("actuator_a_gas", "actuator_b_gas", "main_gas", "blowoff_gas"))
)


class IncompressibleSystem:
    def __init__(
        self, physical, run, cells, initial_liquid, initial_liquid_pressure_pa=None
    ):
        self.physical, self.run, self.cells = physical, run, cells
        self.gas_network = GasNetwork(cells, physical)
        self.g, self.fluid = physical["geometry"], physical["fluid"]
        af, am, ab = liquid_piston_areas(self.g)
        self.j = dict(zip(PISTONS, (-af, -af, -am, ab)))
        self.beta = self.fluid["volumetric_expansion_per_k"] / (
            self.fluid["density_kg_m3_at_20c"] * self.fluid["specific_heat_j_kg_k"]
        )
        self.stop_stiffness = float(
            physical.get("end_stops", {}).get("stiffness_n_m", 1e7)
        )
        self.roll_spring_stiffness = roll_spring_stiffness_n_m(run)
        self.stop_bounds = {
            "actuator_a": (
                -self.g["actuator_compression_travel_m"],
                self.g["actuator_extension_travel_m"],
            ),
            "actuator_b": (
                -self.g["actuator_compression_travel_m"],
                self.g["actuator_extension_travel_m"],
            ),
            "main_accumulator": (None, self.g["main_accumulator_expansion_stop_m"]),
            "blowoff": (0.0, run["blowoff_max_travel_m"]),
        }
        # The zero blow-off position is the initially loaded reference. Keep
        # its gas preload in equilibrium instead of launching a startup impact.
        ph = (
            initial_liquid_pressure_pa - ATMOSPHERIC_PRESSURE_PA
            if initial_liquid_pressure_pa is not None
            else run["initial_gas_gauge_pressure_bar"]["main"] * 1e5
        )
        pg = run["initial_gas_gauge_pressure_bar"]["blowoff"] * 1e5
        self.lower_preloads = {n: 0.0 for n in PISTONS}
        self.lower_preloads["blowoff"] = max(
            0.0, pg * piston_area(self.g["blowoff_piston_diameter_m"]) - ph * ab
        )
        self.initial_energy = sum(
            n["internal_energy_j"] for n in initial_liquid.values()
        )
        self.curves = {}
        for label in ("actuator", "accumulator_inlet"):
            curves = []
            for direction in ("compression", "extension"):
                points = physical["restrictions"][label][direction]
                curves.append(
                    DigressiveCurve.from_points(
                        af,
                        *[
                            points[key][i]
                            for i in range(2)
                            for key in ("speed_m_s", "force_n")
                        ],
                    )
                )
            self.curves[label] = curves
        self.path = [
            "hydraulic_inlet_junction",
            *[f"hydraulic_pipe_{i}" for i in range(cells)],
            "hydraulic_outlet_junction",
        ]

    def source_rate_affine(self, s, state):
        """Linear expansion-source acceleration in piston acceleration.

        Differentiate S=beta*(Q+loss(q,T)) and q=B*v+C*S, solving the
        resulting small linear system instead of differentiating the RHS.
        """
        names = tuple(s["liquid"])
        index = {name: i for i, name in enumerate(names)}
        links = state["volume_links"]
        b = np.zeros((len(links), 4))
        c = np.zeros((len(links), len(names)))
        h = np.zeros((len(names), len(links)))
        temperature_power = np.zeros(len(names))
        # Temperature rates do not depend on piston acceleration.
        qdot, temperature_rates = liquid_heat_rate_derivative(
            self.physical,
            s["liquid"],
            s["solid"],
            s["position"],
            s["velocity"],
            state["liquid"]["node_rates"],
            state["thermal"]["solid_temperature_rate_c_s"],
        )
        temperatures = state["liquid"]["temperatures_c"]
        af = -self.j["actuator_a"]
        viscosity_log_slope = (
            np.log(
                self.fluid["dynamic_viscosity_pa_s_at_100c"]
                / self.fluid["dynamic_viscosity_pa_s"]
            )
            / 80
        )
        for edge, (a, destination, flow) in enumerate(links):
            if edge < 4:
                piston = PISTONS[edge]
                b[edge, edge] = self.j[piston]
                c[edge, index[destination]] = -1.0
            else:
                pipe_index = edge - 4
                b[edge, :2] = af
                for node in (
                    "actuator_a_fluid",
                    "actuator_b_fluid",
                    "hydraulic_inlet_junction",
                ):
                    c[edge, index[node]] = 1.0
                for i in range(pipe_index):
                    c[edge, index[f"hydraulic_pipe_{i}"]] = 1.0
            if edge < 3:
                label = "actuator" if edge < 2 else "accumulator_inlet"
                power_q = restriction_power_slope(
                    flow,
                    *self.curves[label],
                    self.physical["restrictions"]["high_flow_tail_fraction"],
                    VOLUME_FLOW_TRANSITION_M3_S,
                )
                power_t = 0.0
            else:
                if edge == 3:
                    length, diameter = (
                        self.g["blowoff_connection_length_m"],
                        self.g["blowoff_connection_diameter_m"],
                    )
                else:
                    i = edge - 4
                    length = (
                        self.g["hydraulic_pipe_length_m"]
                        / self.cells
                        * (0.5 if i in (0, self.cells) else 1.0)
                    )
                    diameter = self.g["hydraulic_pipe_diameter_m"]
                temp = (temperatures[a] + temperatures[destination]) / 2
                dtemp = (temperature_rates[a] + temperature_rates[destination]) / 2
                viscosity = float(liquid_viscosity(temp, self.fluid))
                power_q, power_mu = pipe_power_derivatives(
                    flow,
                    self.fluid["density_kg_m3_at_20c"],
                    viscosity,
                    length,
                    diameter,
                    self.fluid["pipe_roughness_m"],
                )
                inside = (
                    20 < temp < 100
                    or (temp == 20 and dtemp > 0)
                    or (temp == 100 and dtemp < 0)
                )
                power_t = (
                    power_mu * viscosity * viscosity_log_slope * dtemp
                    if inside
                    else 0.0
                )
            for node in (a, destination):
                h[index[node], edge] = power_q / 2
                temperature_power[index[node]] += power_t / 2
        matrix = np.eye(len(names)) - self.beta * h @ c
        rhs = np.column_stack(
            [
                self.beta * (np.array([qdot[n] for n in names]) + temperature_power),
                self.beta * h @ b,
            ]
        )
        solved = np.linalg.solve(matrix, rhs)
        return lambda acceleration: dict(
            zip(names, solved[:, 0] + solved[:, 1:] @ acceleration)
        )

    def stop_responses(self, positions):
        return {
            n: end_stop_response(
                positions[n],
                *self.stop_bounds[n],
                self.stop_stiffness,
                self.lower_preloads[n],
            )
            for n in PISTONS
        }

    def expansion(self, liquid):
        return self.beta * (
            sum(n["internal_energy_j"] for n in liquid.values()) - self.initial_energy
        )

    def volumes(self, s):
        from .mechanics import chamber_volumes

        pos = dict(s["position"])
        volumes = chamber_volumes(self.g, *[pos[n] for n in PISTONS])
        for group in ("liquid", "gas"):
            for name, node in s[group].items():
                if name in volumes:
                    node["volume_m3"] = volumes[name]
        ag = piston_area(self.g["actuator_piston_diameter_m"])
        bg = piston_area(self.g["blowoff_piston_diameter_m"])
        gas_vdot = dict(zip(PISTONS, (ag, ag, -self.j["main_accumulator"], -bg)))
        for name in PISTONS:
            s["gas"][GAS_PISTONS[name]]["volume_rate_m3_s"] = (
                gas_vdot[name] * s["velocity"][name]
            )
        return pos

    def loss_heat(self, velocities, sources, temperatures):
        flows = hydraulic_flows_m3_s(self.g, velocities, sources, self.cells)
        heat = {n: 0.0 for n in temperatures}
        links = []
        tail = self.physical["restrictions"]["high_flow_tail_fraction"]
        for piston in PISTONS:
            chamber = LIQUID_PISTONS[piston]
            junction = self.path[0] if piston.startswith("actuator") else self.path[-1]
            q = flows[piston]
            if piston == "blowoff":
                dp = pipe_pressure_drop(
                    q,
                    self.fluid["density_kg_m3_at_20c"],
                    float(
                        liquid_viscosity(
                            (temperatures[chamber] + temperatures[junction]) / 2,
                            self.fluid,
                        )
                    ),
                    self.g["blowoff_connection_length_m"],
                    self.g["blowoff_connection_diameter_m"],
                    self.fluid["pipe_roughness_m"],
                )
            else:
                label = (
                    "actuator" if piston.startswith("actuator") else "accumulator_inlet"
                )
                dp = regularized_restriction_pressure_drop(
                    q,
                    *self.curves[label],
                    tail,
                    zero_flow_transition_m3_s=VOLUME_FLOW_TRANSITION_M3_S,
                )
            power = q * dp
            heat[chamber] += power / 2
            heat[junction] += power / 2
            links.append((junction, chamber, q))
        for i, q in enumerate(flows["pipe"]):
            a, b = self.path[i : i + 2]
            length = (
                self.g["hydraulic_pipe_length_m"]
                / self.cells
                * (0.5 if i in (0, self.cells) else 1)
            )
            dp = pipe_pressure_drop(
                q,
                self.fluid["density_kg_m3_at_20c"],
                float(
                    liquid_viscosity(
                        (temperatures[a] + temperatures[b]) / 2, self.fluid
                    )
                ),
                length,
                self.g["hydraulic_pipe_diameter_m"],
                self.fluid["pipe_roughness_m"],
            )
            heat[a] += q * dp / 2
            heat[b] += q * dp / 2
            links.append((a, b, q))
        return flows, heat, links

    def thermal_state(self, s, u, dependent=None):
        pos = self.volumes(s)
        self.gas_network.project(s["gas"], s.get("gas_network_internal_energy_j"))
        thermal = thermal_exchange(
            self.physical,
            self.run,
            s["liquid"],
            s["gas"],
            s["solid"],
            pos,
            u["hot_ambient_c"],
            u["duct_air_speed_m_s"],
        )
        temperatures = {
            n: liquid_temperatures(
                node["mass_kg"],
                node["internal_energy_j"],
                self.fluid,
                node.get("boundary_energy_j"),
                node.get("boundary_mass_fraction"),
            )[1]
            for n, node in s["liquid"].items()
        }
        sources = {n: self.beta * q for n, q in thermal["liquid_heat_w"].items()}
        for iteration in range(20):
            if dependent is not None:
                s["velocity"][dependent] = (
                    sum(sources.values())
                    - sum(
                        self.j[n] * s["velocity"][n] for n in PISTONS if n != dependent
                    )
                ) / self.j[dependent]
            flows, losses, links = self.loss_heat(s["velocity"], sources, temperatures)
            updated = {
                n: self.beta * (thermal["liquid_heat_w"][n] + losses[n])
                for n in sources
            }
            error = max(abs(updated[n] - sources[n]) for n in sources)
            sources = updated
            if error < 1e-15:
                break
        else:
            raise RuntimeError("thermal volume/flow iteration did not converge")
        if dependent is not None:
            s["velocity"][dependent] = (
                sum(sources.values())
                - sum(self.j[n] * s["velocity"][n] for n in PISTONS if n != dependent)
            ) / self.j[dependent]
        flows, losses, links = self.loss_heat(s["velocity"], sources, temperatures)
        rates = {
            n: {
                "mass_rate_kg_s": 0.0,
                "energy_rate_w": thermal["liquid_heat_w"][n] + losses[n],
            }
            for n in s["liquid"]
        }
        mass_links = []
        for a, b, q in links:
            weight = 0.5 * (1 + np.tanh(q / VOLUME_FLOW_TRANSITION_M3_S))
            transported_temperature = (
                weight * temperatures[a] + (1 - weight) * temperatures[b]
            )
            mass_flow = q * float(liquid_density(transported_temperature, self.fluid))
            energy_flow = (
                mass_flow
                * self.fluid["specific_heat_j_kg_k"]
                * (transported_temperature - 20)
            )
            rates[a]["mass_rate_kg_s"] -= mass_flow
            rates[b]["mass_rate_kg_s"] += mass_flow
            rates[a]["energy_rate_w"] -= energy_flow
            rates[b]["energy_rate_w"] += energy_flow
            mass_links.append((a, b, mass_flow))
        for n, node in s["liquid"].items():
            if "boundary_mass_fraction" in node:
                f = node["boundary_mass_fraction"]
                exchange = f * rates[n]["mass_rate_kg_s"]
                boundary_t = liquid_temperatures(
                    node["mass_kg"],
                    node["internal_energy_j"],
                    self.fluid,
                    node["boundary_energy_j"],
                    f,
                )[2]
                energy = self.fluid["specific_heat_j_kg_k"] * (
                    (
                        0.5
                        * (1 + np.tanh(exchange / MASS_FLOW_TRANSITION_KG_S))
                        * temperatures[n]
                        + 0.5
                        * (1 - np.tanh(exchange / MASS_FLOW_TRANSITION_KG_S))
                        * boundary_t
                    )
                    - 20
                )
                rates[n]["boundary_energy_rate_w"] = (
                    exchange * energy
                    + thermal["liquid_boundary_heat_w"][n]
                    + f * losses[n]
                )
        gas = self.gas_rates(s, thermal)
        return {
            "sources": sources,
            "volume_links": links,
            "flows": flows,
            "losses": losses,
            "thermal": thermal,
            "liquid": {
                "node_rates": rates,
                "mass_flows_kg_s": mass_links,
                "temperatures_c": temperatures,
            },
            "gas": gas,
        }

    def gas_rates(self, s, thermal):
        # Gas chamber volume rates must use the reconstructed velocity.
        self.volumes(s)
        for n, q in thermal["gas_heat_w"].items():
            s["gas"][n]["heat_rate_w"] = q
            if n in thermal["gas_boundary_heat_w"]:
                s["gas"][n]["boundary_heat_rate_w"] = thermal["gas_boundary_heat_w"][n]
        return self.gas_network.evaluate(s["gas"])

    def gas_thermal_jacobian_rate(self, s, u, base, dependent):
        """Update the gas/solid block for gas-only Newton perturbations.

        With thermal-volume acceleration coefficients frozen, these fields
        leave liquid heat input, flows, pressure and mechanics unchanged.
        """
        s["velocity"][dependent] = base["position_rate_m_s"][dependent]
        self.volumes(s)
        thermal = thermal_exchange(
            self.physical,
            self.run,
            s["liquid"],
            s["gas"],
            s["solid"],
            s["position"],
            u["hot_ambient_c"],
            u["duct_air_speed_m_s"],
        )
        for name in PISTONS:
            power = base["mechanics"]["friction_force_n"][name] * s["velocity"][name]
            prefix = "main" if name == "main_accumulator" else name
            capacity = (
                self.physical["moving_masses_kg"][name]
                * self.physical["thermal"]["metal_specific_heat_j_kg_k"]
                / 2
            )
            for side in ("fluid", "gas"):
                key = f"{prefix}_piston_{side}"
                thermal["solid_heat_w"][key] += power / 2
                thermal["solid_temperature_rate_c_s"][key] += power / (2 * capacity)
        return {**base, "thermal": thermal, "gas": self.gas_rates(s, thermal)}

    def evaluate(
        self,
        s,
        u,
        dependent="main_accumulator",
        derivatives=True,
        frozen_source_rate_function=None,
    ):
        tstate = self.thermal_state(s, u, dependent)
        sources = tstate["sources"]
        gp = {n: tstate["gas"]["pressures_pa"][GAS_PISTONS[n]] for n in PISTONS}
        forces = {n: u[n + "_force_n"] for n in ("actuator_a", "actuator_b")}
        args = (self.physical, s["velocity"], gp, forces, sources, {}, self.cells)
        stops = self.stop_responses(s["position"])
        roll_deflection = s["position"]["actuator_a"] - s["position"]["actuator_b"]
        roll_force = self.roll_spring_stiffness * roll_deflection
        roll_forces = {"actuator_a": -roll_force, "actuator_b": roll_force}
        kwargs = dict(
            external_piston_forces_n={
                n: response["force_n"] + roll_forces.get(n, 0.0)
                for n, response in stops.items()
            },
            temperatures_c=tstate["liquid"]["temperatures_c"],
            friction_velocity_scale_m_s=self.run.get("solver", {}).get(
                "friction_velocity_scale_m_s", 0.005
            ),
            allow_nonpositive_pressure=True,
        )
        mechanics = constrained_accelerations(
            *args, source_rate_function=frozen_source_rate_function, **kwargs
        )
        base_heat = dict(tstate["thermal"]["solid_heat_w"])
        base_rates = dict(tstate["thermal"]["solid_temperature_rate_c_s"])

        def friction_heat(result):
            powers = {
                name: result["friction_force_n"][name] * s["velocity"][name]
                for name in PISTONS
            }
            for name, power in powers.items():
                prefix = "main" if name == "main_accumulator" else name
                capacity = (
                    self.physical["moving_masses_kg"][name]
                    * self.physical["thermal"]["metal_specific_heat_j_kg_k"]
                    / 2
                )
                for side in ("fluid", "gas"):
                    key = f"{prefix}_piston_{side}"
                    tstate["thermal"]["solid_heat_w"][key] = base_heat[key] + power / 2
                    tstate["thermal"]["solid_temperature_rate_c_s"][key] = base_rates[
                        key
                    ] + power / (2 * capacity)
            return np.array(list(powers.values()))

        old_power = friction_heat(mechanics)
        source_rates = frozen_source_rate_function
        if derivatives and self.beta and frozen_source_rate_function is None:
            for _ in range(8):
                source_rates = self.source_rate_affine(s, tstate)
                mechanics = constrained_accelerations(
                    *args, source_rate_function=source_rates, **kwargs
                )
                new_power = friction_heat(mechanics)
                if np.max(abs(new_power - old_power)) <= 1e-10 * (
                    1 + np.max(abs(new_power))
                ):
                    break
                old_power = new_power
            else:
                raise RuntimeError(
                    "thermal/friction constraint iteration did not converge"
                )
        return {
            **tstate,
            "mechanics": mechanics,
            "end_stops": stops,
            "roll_spring_force_n": roll_forces,
            "roll_spring_energy_j": 0.5 * roll_force * roll_deflection,
            "position_rate_m_s": s["velocity"],
            "velocity_rate_m_s2": mechanics["acceleration_m_s2"],
            "source_rate_function": source_rates,
        }
