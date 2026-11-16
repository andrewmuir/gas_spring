"""Conservative gas transport with algebraic pressure equalisation.

Connected low-side volumes share one pressure and total internal energy.
Their masses and chamber boundary-layer energies retain thermal histories.
Main and blow-off gas volumes remain separate, closed ideal-gas springs.
"""

import math

import numpy as np

from .balances import ideal_gas_state
from .hydraulics import pipe_pressure_drop
from .numerics import MASS_FLOW_TRANSITION_KG_S


class GasNetwork:
    def __init__(self, pipe_cells, physical):
        if (
            isinstance(pipe_cells, bool)
            or not isinstance(pipe_cells, int)
            or pipe_cells < 1
        ):
            raise ValueError("pipe_cell_count must be positive")
        self.physical = physical
        self.gas, g = physical["gas"], physical["geometry"]
        self.names = (
            "actuator_a_gas",
            "actuator_b_gas",
            "actuator_gas_junction",
            *(f"gas_pipe_{i}" for i in range(pipe_cells)),
            "low_pressure_reservoir_gas",
        )
        self.closed = ("main_gas", "blowoff_gas")
        self.cp = self.gas["specific_heat_cp_j_kg_k"]
        self.cv = self.cp - self.gas["specific_gas_constant_j_kg_k"]
        self.k = self.gas["specific_gas_constant_j_kg_k"] / self.cv
        self.edges = [
            (0, 2),
            (1, 2),
            *((i, i + 1) for i in range(2, len(self.names) - 1)),
        ]
        dx = g["gas_pipe_length_m"] / pipe_cells
        self.lengths = [g["actuator_gas_connection_length_m"]] * 2 + [
            dx * (0.5 if i in (0, pipe_cells) else 1.0) for i in range(pipe_cells + 1)
        ]
        self.diameters = [g["actuator_gas_connection_diameter_m"]] * 2 + [
            g["gas_pipe_diameter_m"]
        ] * (pipe_cells + 1)

    def project(self, nodes, total_energy_j=None):
        """Reconstruct node energies from their volumes and shared total energy."""
        volumes = [nodes[n]["volume_m3"] for n in self.names]
        total = (
            sum(nodes[n]["internal_energy_j"] for n in self.names)
            if total_energy_j is None
            else float(total_energy_j)
        )
        if (
            not math.isfinite(total)
            or total <= 0
            or any(not math.isfinite(v) or v <= 0 for v in volumes)
        ):
            raise ValueError("invalid shared gas energy or volume")
        energy_density = total / sum(volumes)
        for name, volume in zip(self.names, volumes):
            nodes[name]["internal_energy_j"] = energy_density * volume
        return self.k * energy_density

    def evaluate(self, nodes):
        """Solve transport enthalpy fluxes from the differentiated gas EOS.

        For each volume U_i=P*V_i/(gamma-1). Its energy balance determines
        net incoming enthalpy. The tree topology then determines every edge
        enthalpy flux without a pressure relaxation ODE or nonlinear solve.
        """
        if set(nodes) != set(self.names + self.closed):
            raise ValueError("gas nodes do not match network topology")
        pressure = self.project(nodes)
        pressures, temperatures, boundaries, h = {}, {}, {}, {}
        for name, node in nodes.items():
            p, t = ideal_gas_state(
                node["mass_kg"], node["internal_energy_j"], node["volume_m3"], self.gas
            )
            pressures[name] = pressure if name in self.names else p
            if "boundary_mass_fraction" in node:
                f, m, ub, u = (
                    node["boundary_mass_fraction"],
                    node["mass_kg"],
                    node["boundary_energy_j"],
                    node["internal_energy_j"],
                )
                if not 0 < f < 1 or not 0 < ub < u:
                    raise ValueError(f"invalid gas boundary state for {name}")
                boundaries[name] = ub / (f * m * self.cv) - 273.15
                t = (u - ub) / ((1 - f) * m * self.cv) - 273.15
            temperatures[name] = t
            h[name] = self.cp * (t + 273.15)
        v = np.array([nodes[n]["volume_m3"] for n in self.names])
        dv = np.array([nodes[n].get("volume_rate_m3_s", 0.0) for n in self.names])
        heat = np.array([nodes[n].get("heat_rate_w", 0.0) for n in self.names])
        if not np.isfinite(dv).all() or not np.isfinite(heat).all():
            raise ValueError("invalid gas heat or volume rate")
        volume, volume_rate = v.sum(), dv.sum()
        total_rate = heat.sum() - pressure * volume_rate
        pressure_rate = (self.k * total_rate - pressure * volume_rate) / volume
        du = (v * pressure_rate + pressure * dv) / self.k
        # Enforce the aggregate first law to roundoff, including cancellation.
        du[-1] = total_rate - sum(du[:-1])
        incoming = du - heat + pressure * dv
        flux = [-incoming[0], -incoming[1]]
        running = flux[0] + flux[1]
        for i in range(2, len(self.names) - 1):
            running -= incoming[i]
            flux.append(running)
        dm = np.zeros(len(self.names))
        links, energy_links = [], []
        for (a, b), energy_flow in zip(self.edges, flux):
            na, nb = self.names[a], self.names[b]
            ha, hb = h[na], h[nb]
            # Select the enthalpy donor smoothly in energy-flux coordinates.
            # This regularises reversal without solving a mass-flow equation.
            weight = 0.5 * (
                1
                + math.tanh(energy_flow / (MASS_FLOW_TRANSITION_KG_S * 0.5 * (ha + hb)))
            )
            flow = energy_flow / (weight * ha + (1 - weight) * hb)
            dm[a] -= flow
            dm[b] += flow
            links.append((na, nb, flow))
            energy_links.append((na, nb, energy_flow))
        rates = {
            n: dict(mass_rate_kg_s=float(dm[i]), energy_rate_w=float(du[i]))
            for i, n in enumerate(self.names)
        }
        for name in self.closed:
            node = nodes[name]
            rates[name] = dict(
                mass_rate_kg_s=0.0,
                energy_rate_w=(
                    node.get("heat_rate_w", 0.0)
                    - pressures[name] * node.get("volume_rate_m3_s", 0.0)
                ),
            )
        for name, boundary_t in boundaries.items():
            node = nodes[name]
            fraction = node["boundary_mass_fraction"]
            exchange = fraction * rates[name]["mass_rate_kg_s"]
            weight = 0.5 * (1 + math.tanh(exchange / MASS_FLOW_TRANSITION_KG_S))
            source_h = weight * h[name] + (1 - weight) * self.cp * (boundary_t + 273.15)
            rates[name]["boundary_energy_rate_w"] = (
                exchange * source_h
                - fraction * pressures[name] * node.get("volume_rate_m3_s", 0.0)
                + node.get("boundary_heat_rate_w", node.get("heat_rate_w", 0.0))
                + node.get("core_to_boundary_heat_w", 0.0)
            )
        return dict(
            node_rates=rates,
            pressures_pa=pressures,
            temperatures_c=temperatures,
            boundary_temperatures_c=boundaries,
            mass_flows_kg_s=links,
            enthalpy_flows_w=energy_links,
            network_energy_rate_w=float(total_rate),
            network_pressure_rate_pa_s=float(pressure_rate),
            network_pressure_pa=pressure,
        )

    def loss_estimate(self, rates):
        """Estimate omitted Darcy pressure differences and Mach numbers for QA.

        These are diagnostics, not pressure forces or added heat. The reduced
        model assumes negligible pressure differences in the connected gas.
        """
        pressure = rates["network_pressure_pa"]
        drops, mach = [], []
        for (a, b, flow), length, diameter in zip(
            rates["mass_flows_kg_s"], self.lengths, self.diameters
        ):
            temperature = (
                rates["temperatures_c"][a] + rates["temperatures_c"][b]
            ) / 2 + 273.15
            density = pressure / (
                self.gas["specific_gas_constant_j_kg_k"] * temperature
            )
            q = flow / density
            drops.append(
                pipe_pressure_drop(
                    q,
                    density,
                    self.gas["dynamic_viscosity_pa_s_at_25c"],
                    length,
                    diameter,
                )
            )
            speed = abs(q) / (math.pi * diameter**2 / 4)
            mach.append(
                speed
                / math.sqrt(
                    self.cp
                    / self.cv
                    * self.gas["specific_gas_constant_j_kg_k"]
                    * temperature
                )
            )
        offsets = np.zeros(len(self.names))
        for (a, b), dp in reversed(list(zip(self.edges, drops))):
            offsets[a] = offsets[b] + dp
        maximum = float(max(abs(offsets)))
        return dict(
            max_pressure_drop_pa=maximum,
            pressure_drop_fraction=maximum / pressure,
            max_mach_number=max(mach, default=0.0),
        )
