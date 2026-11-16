import copy

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gas_spring.circuit import circuit_nodes_from_initial_state
from gas_spring.config import load_yaml
from gas_spring.gas_network import GasNetwork
from gas_spring.initialization import initialize_system


def network_state():
    physical, run = load_yaml("config/physical.yaml"), load_yaml("config/run.yaml")
    initialized = initialize_system(physical, run, 1729.0, 1729.0, pipe_cells=3)
    _, nodes = circuit_nodes_from_initial_state(initialized)
    return GasNetwork(3, physical), nodes


def test_local_and_global_first_law_with_different_temperatures_and_moving_pistons():
    network, nodes = network_state()
    nodes["actuator_a_gas"]["volume_rate_m3_s"] = 2e-5
    nodes["actuator_b_gas"]["volume_rate_m3_s"] = -7e-6
    for i, name in enumerate(network.names + network.closed):
        nodes[name]["heat_rate_w"] = i * 0.3 - 0.7
    rates = network.evaluate(nodes)
    net = dict.fromkeys(nodes, 0.0)
    for a, b, power in rates["enthalpy_flows_w"]:
        net[a] -= power
        net[b] += power
    for name, node in nodes.items():
        expected = (
            net[name]
            + node.get("heat_rate_w", 0.0)
            - rates["pressures_pa"][name] * node.get("volume_rate_m3_s", 0.0)
        )
        assert rates["node_rates"][name]["energy_rate_w"] == pytest.approx(
            expected, abs=2e-13
        )
    assert sum(
        r["mass_rate_kg_s"] for r in rates["node_rates"].values()
    ) == pytest.approx(0.0, abs=1e-17)
    total = sum(
        nodes[n].get("heat_rate_w", 0.0)
        - rates["pressures_pa"][n] * nodes[n].get("volume_rate_m3_s", 0.0)
        for n in nodes
    )
    assert sum(
        r["energy_rate_w"] for r in rates["node_rates"].values()
    ) == pytest.approx(total, abs=1e-12)
    # Every local EOS has the same pressure derivative despite unequal gas temperatures.
    for name in network.names:
        n = nodes[name]
        derivative = (
            network.k * rates["node_rates"][name]["energy_rate_w"]
            - rates["pressures_pa"][name] * n.get("volume_rate_m3_s", 0.0)
        ) / n["volume_m3"]
        assert derivative == pytest.approx(
            rates["network_pressure_rate_pa_s"], rel=1e-12
        )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_adiabatic_network_compression_and_expansion_conserve_mass(direction):
    network, nodes = network_state()
    for node in nodes.values():
        node.pop("boundary_mass_fraction", None)
        node.pop("boundary_energy_j", None)
    volumes = {n: nodes[n]["volume_m3"] for n in network.names}
    initial_energy = sum(nodes[n]["internal_energy_j"] for n in network.names)
    initial_volume = sum(volumes.values())
    initial_mass = sum(nodes[n]["mass_kg"] for n in network.names)
    speed = direction * volumes["actuator_a_gas"] * 0.2

    def rhs(t, y):
        local = copy.deepcopy(nodes)
        local["actuator_a_gas"]["volume_m3"] += speed * t
        local["actuator_a_gas"]["volume_rate_m3_s"] = speed
        for n, m in zip(network.names, y[1:]):
            local[n]["mass_kg"] = m
        network.project(local, y[0])
        r = network.evaluate(local)
        return [
            r["network_energy_rate_w"],
            *[r["node_rates"][n]["mass_rate_kg_s"] for n in network.names],
        ]

    solved = solve_ivp(
        rhs,
        (0.0, 1.0),
        [initial_energy, *[nodes[n]["mass_kg"] for n in network.names]],
        method="DOP853",
        rtol=1e-10,
        atol=1e-13,
        max_step=0.05,
    )
    assert solved.success
    final_volume = initial_volume + speed
    expected = initial_energy * (initial_volume / final_volume) ** network.k
    assert solved.y[0, -1] == pytest.approx(expected, rel=1e-10)
    assert sum(solved.y[1:, -1]) == pytest.approx(initial_mass, abs=1e-17)
    assert (solved.y[1:] > 0).all()


def test_reversal_uses_the_upstream_gas_enthalpy_and_retains_temperature_gradients():
    network, nodes = network_state()
    for direction in (-1.0, 1.0):
        local = copy.deepcopy(nodes)
        local["actuator_a_gas"]["volume_rate_m3_s"] = direction * 1e-5
        r = network.evaluate(local)
        for (a, b, flow), (_, _, power) in zip(
            r["mass_flows_kg_s"], r["enthalpy_flows_w"]
        ):
            donor = a if flow >= 0 else b
            enthalpy = network.cp * (r["temperatures_c"][donor] + 273.15)
            if abs(flow) > 1e-7:
                assert flow * enthalpy == pytest.approx(power, rel=1e-12)
        assert (
            r["temperatures_c"]["actuator_a_gas"]
            > r["temperatures_c"]["low_pressure_reservoir_gas"]
        )
        assert len(set(r["pressures_pa"][n] for n in network.names)) == 1


def test_pressure_loss_diagnostic_does_not_inject_energy():
    network, nodes = network_state()
    nodes["actuator_a_gas"]["volume_rate_m3_s"] = 1e-4
    rates = network.evaluate(nodes)
    before = copy.deepcopy(rates)
    estimate = network.loss_estimate(rates)
    assert estimate["max_pressure_drop_pa"] > 0
    assert estimate["max_mach_number"] > 0
    assert rates == before
