"""Instantaneous coupled circuit and piston rates, without time integration."""
from .circuit import hydraulic_rates, nitrogen_rates
from .hydraulics import piston_area
from .materials import ATMOSPHERIC_PRESSURE_PA
from .mechanics import chamber_volumes, piston_acceleration, seal_friction_limit
from .thermal_coupling import thermal_exchange


def coupled_rates(physical, run, liquid_nodes, gas_nodes, pipe_flows_m3_s,
                  positions_m, velocities_m_s, input_forces_n,
                  solid_temperatures_c=None, hot_ambient_c=None, duct_air_speed_m_s=None,
                  contact_modes=None):
    """Assemble hydropneumatic and mechanical rates at one instant.

    With solid temperatures and environmental inputs supplied, computes
    thermal exchange and wall rates too. The caller integrates the result and
    resolves impacts and friction zero crossings.
    Actuator compression forces are positive, while actuator position is
    positive in extension. At the instant of an outward impact, this function
    rejects the state so that an event handler can resolve the impact.
    """
    g = physical["geometry"]
    names = ("actuator_a", "actuator_b", "main_accumulator", "blowoff")
    pos = {name: float(positions_m[name]) for name in names}
    vel = {name: float(velocities_m_s[name]) for name in names}
    if contact_modes is not None:
        # Implicit solvers may evaluate just outside an event surface while
        # locating it. Evaluate volumes at the surface and let the event
        # handler project the accepted state and resolve the impact.
        pos["actuator_a"] = min(max(pos["actuator_a"], -g["actuator_compression_travel_m"]), g["actuator_extension_travel_m"])
        pos["actuator_b"] = min(max(pos["actuator_b"], -g["actuator_compression_travel_m"]), g["actuator_extension_travel_m"])
        pos["blowoff"] = min(max(pos["blowoff"], 0), run["blowoff_max_travel_m"])
        pos["main_accumulator"] = min(pos["main_accumulator"], g["main_accumulator_expansion_stop_m"])
        for name in names:
            if contact_modes[name] == 0:
                vel[name] = 0.0
    volumes = chamber_volumes(g, pos["actuator_a"], pos["actuator_b"],
                              pos["main_accumulator"], pos["blowoff"])
    af = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    ag = piston_area(g["actuator_piston_diameter_m"])
    am = piston_area(g["main_accumulator_piston_diameter_m"])
    bf = piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"])
    bg = piston_area(g["blowoff_piston_diameter_m"])
    liquid = {name: dict(node) for name, node in liquid_nodes.items()}
    gas = {name: dict(node) for name, node in gas_nodes.items()}
    for name in ("actuator_a", "actuator_b"):
        liquid[name+"_fluid"].update(volume_m3=volumes[name+"_fluid"], volume_rate_m3_s=-af*vel[name])
        gas[name+"_gas"].update(volume_m3=volumes[name+"_gas"], volume_rate_m3_s=ag*vel[name])
    liquid["main_fluid"].update(volume_m3=volumes["main_fluid"], volume_rate_m3_s=-am*vel["main_accumulator"])
    gas["main_gas"].update(volume_m3=volumes["main_gas"], volume_rate_m3_s=am*vel["main_accumulator"])
    liquid["blowoff_fluid"].update(volume_m3=volumes["blowoff_fluid"], volume_rate_m3_s=bf*vel["blowoff"])
    gas["blowoff_gas"].update(volume_m3=volumes["blowoff_gas"], volume_rate_m3_s=-bg*vel["blowoff"])
    thermal = None
    if solid_temperatures_c is not None:
        if hot_ambient_c is None or duct_air_speed_m_s is None:
            raise ValueError("thermal exchange requires hot ambient and duct air speed")
        thermal = thermal_exchange(physical, run, liquid, gas, solid_temperatures_c,
                                   pos, hot_ambient_c, duct_air_speed_m_s)
        for name, heat in thermal["liquid_heat_w"].items():
            liquid[name]["heat_rate_w"] = float(liquid[name].get("heat_rate_w", 0)) + heat
            if name in thermal["liquid_boundary_heat_w"]:
                liquid[name]["boundary_heat_rate_w"] = thermal["liquid_boundary_heat_w"][name]
        for name, heat in thermal["gas_heat_w"].items():
            gas[name]["heat_rate_w"] = float(gas[name].get("heat_rate_w", 0)) + heat
            if name in thermal["gas_boundary_heat_w"]:
                gas[name]["boundary_heat_rate_w"] = thermal["gas_boundary_heat_w"][name]
    liquid_result = hydraulic_rates(liquid, pipe_flows_m3_s, physical)
    gas_result = nitrogen_rates(gas, len(pipe_flows_m3_s)-1, physical)
    hp, gp = liquid_result["pressures_pa"], gas_result["pressures_pa"]
    ambient = ATMOSPHERIC_PRESSURE_PA
    forces = {
        "actuator_a": (gp["actuator_a_gas"]-ambient)*ag - (hp["actuator_a_fluid"]-ambient)*af - float(input_forces_n["actuator_a"]),
        "actuator_b": (gp["actuator_b_gas"]-ambient)*ag - (hp["actuator_b_fluid"]-ambient)*af - float(input_forces_n["actuator_b"]),
        "main_accumulator": ((gp["main_gas"]-ambient)-(hp["main_fluid"]-ambient))*am,
        "blowoff": (hp["blowoff_fluid"]-ambient)*bf - (gp["blowoff_gas"]-ambient)*bg,
    }
    limits = {
        "actuator_a": seal_friction_limit(hp["actuator_a_fluid"]-ambient, gp["actuator_a_gas"]-ambient, physical["seal_friction"]["actuator"]),
        "actuator_b": seal_friction_limit(hp["actuator_b_fluid"]-ambient, gp["actuator_b_gas"]-ambient, physical["seal_friction"]["actuator"]),
        "main_accumulator": seal_friction_limit(hp["main_fluid"]-ambient, gp["main_gas"]-ambient, physical["seal_friction"]["main_accumulator"]),
        "blowoff": seal_friction_limit(hp["blowoff_fluid"]-ambient, gp["blowoff_gas"]-ambient, physical["seal_friction"]["blowoff"]),
    }
    stops = {
        "actuator_a": (-g["actuator_compression_travel_m"], g["actuator_extension_travel_m"]),
        "actuator_b": (-g["actuator_compression_travel_m"], g["actuator_extension_travel_m"]),
        "main_accumulator": (None, g["main_accumulator_expansion_stop_m"]),
        "blowoff": (0.0, run["blowoff_max_travel_m"]),
    }
    kinetic_ratio = float(physical["seal_friction"]["kinetic_to_static_ratio"])
    if not 0 <= kinetic_ratio <= 1:
        raise ValueError("kinetic_to_static_ratio must be between zero and one")
    if contact_modes is None:
        acceleration = {name: piston_acceleration(pos[name], vel[name], forces[name], limits[name],
                                                   physical["moving_masses_kg"][name], *stops[name],
                                                   kinetic_friction_limit_n=limits[name]*kinetic_ratio)
                        for name in names}
        friction_heat = {name: limits[name] * kinetic_ratio * abs(vel[name]) for name in names}
    else:
        if any(contact_modes[name] not in (-1, 0, 1) for name in names):
            raise ValueError("contact modes must be -1, 0 or 1")
        acceleration = {name: (forces[name] - contact_modes[name]*kinetic_ratio*limits[name]) / physical["moving_masses_kg"][name]
                        if contact_modes[name] else 0.0 for name in names}
        friction_heat = {name: limits[name] * kinetic_ratio * max(0.0, contact_modes[name]*vel[name]) for name in names}
    if thermal is not None:
        for name, prefix in (("actuator_a", "actuator_a"), ("actuator_b", "actuator_b"),
                             ("main_accumulator", "main"), ("blowoff", "blowoff")):
            for side in ("fluid", "gas"):
                key = f"{prefix}_piston_{side}"
                power = friction_heat[name]/2
                thermal["solid_heat_w"][key] += power
                heat_capacity = physical["moving_masses_kg"][name] * physical["thermal"]["metal_specific_heat_j_kg_k"]/2
                thermal["solid_temperature_rate_c_s"][key] += power/heat_capacity
    return {
        "liquid": liquid_result, "gas": gas_result,
        "position_rate_m_s": vel, "velocity_rate_m_s2": acceleration,
        "piston_drive_force_n": forces, "piston_friction_limit_n": limits,
        "piston_friction_heat_w": friction_heat, "thermal": thermal,
    }
