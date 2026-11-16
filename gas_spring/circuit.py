"""Finite-volume hydraulic and nitrogen circuit balance assembly.

These functions evaluate instantaneous rates, not a time simulation. Every
node contains mass_kg, internal_energy_j, volume_m3 and optionally
volume_rate_m3_s and heat_rate_w. Pressures are absolute. The time integrator
must supply current piston-dependent volumes and handle stops and friction.
"""

from .gas_network import GasNetwork


def circuit_nodes_from_initial_state(state):
    """Convert initialized chamber and pipe states into mutable circuit nodes."""
    liquid = {name: dict(node) for name, node in state["liquid_chambers"].items()}
    gas = {name: dict(node) for name, node in state["gas_chambers"].items()}
    for name, destination in (("hydraulic", liquid), ("gas", gas)):
        pipe = state["pipes"][name]
        for i, (mass, energy) in enumerate(
            zip(pipe["mass_kg"], pipe["internal_energy_j"])
        ):
            destination[f"{name}_pipe_{i}"] = {
                "mass_kg": float(mass),
                "internal_energy_j": float(energy),
                "volume_m3": float(pipe["cell_volume_m3"]),
            }
    return liquid, gas


def nitrogen_rates(nodes, pipe_cell_count, physical):
    """Conservative common-pressure gas network and closed gas-spring rates."""
    return GasNetwork(pipe_cell_count, physical).evaluate(nodes)
