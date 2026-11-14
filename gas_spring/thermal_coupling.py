"""Position-dependent thermal exchange rates for the coupled circuit.

All heat exchanges are assembled as equal and opposite internal links. Wall
temperatures are separate states; this module does not integrate them in time.
"""
import math
import numpy as np

from .balances import ideal_gas_state
from .hydraulics import piston_area
from .materials import liquid_temperatures
from .thermal import actuator_wall_cell_areas


def thermal_exchange(physical, run, liquid_nodes, gas_nodes, solid_temperatures_c,
                     positions_m, hot_ambient_c, duct_air_speed_m_s):
    """Compute fluid/gas heat rates, solid temperature rates and ambient power.

    Chamber gas is split into a core and a near-wall zone with an explicit
    energy state. Gas in the pipe is one zone per axial cell. Actuator walls
    and both pipe walls use fixed axial cells; accumulator housings are lumped.
    Film areas on the accumulator housings are provisional, as are their
    effective external areas and the cooler parameters.
    """
    g, th, fluid, gas = physical["geometry"], physical["thermal"], physical["fluid"], physical["gas"]
    if not np.isfinite([hot_ambient_c, duct_air_speed_m_s]).all() or hot_ambient_c <= -273.15 or duct_air_speed_m_s < 0:
        raise ValueError("invalid hot ambient or duct air speed")
    temps, heat, capacity = {}, {}, {}
    def add_temp(key, value):
        if not np.isfinite(value) or value <= -273.15:
            raise ValueError(f"invalid temperature for {key}")
        temps[key] = float(value)
        heat[key] = 0.0
    for name, node in liquid_nodes.items():
        _, temp, boundary = liquid_temperatures(node["mass_kg"], node["internal_energy_j"], fluid,
            node.get("boundary_energy_j"), node.get("boundary_mass_fraction"))
        if "boundary_mass_fraction" in node:
            add_temp("lb:"+name, boundary)
        add_temp("l:"+name, temp)
    cv = gas["specific_heat_cp_j_kg_k"] - gas["specific_gas_constant_j_kg_k"]
    for name, node in gas_nodes.items():
        _, mean_t = ideal_gas_state(node["mass_kg"], node["internal_energy_j"], node["volume_m3"], gas)
        if "boundary_mass_fraction" in node:
            fraction, mass = node["boundary_mass_fraction"], node["mass_kg"]
            ub = node["boundary_energy_j"]
            if not 0 < fraction < 1 or not 0 < ub < node["internal_energy_j"]:
                raise ValueError(f"invalid boundary energy for {name}")
            core = (node["internal_energy_j"]-ub)/((1-fraction)*mass*cv)-273.15
            boundary = ub/(fraction*mass*cv)-273.15
            add_temp("g:"+name, core)
            add_temp("b:"+name, boundary)
        else:
            add_temp("g:"+name, mean_t)
    cp = th["metal_specific_heat_j_kg_k"]
    def solid(name, value, heat_capacity):
        if not np.isfinite(heat_capacity) or heat_capacity <= 0:
            raise ValueError(f"invalid thermal capacity for {name}")
        key = "s:"+name
        add_temp(key, value)
        capacity[key] = heat_capacity
    def exchange(a, b, conductance):
        if not np.isfinite(conductance) or conductance < 0:
            raise ValueError("invalid thermal conductance")
        power = conductance * (temps[a]-temps[b])
        heat[a] -= power
        heat[b] += power
    for label in ("actuator_a", "actuator_b"):
        cells = np.asarray(solid_temperatures_c[label+"_wall"], dtype=float)
        if cells.ndim != 1 or len(cells) == 0:
            raise ValueError("actuator wall must have at least one axial cell")
        for i, temp in enumerate(cells):
            solid(f"{label}_wall_{i}", temp, th["actuator_housing_mass_kg_each"]*cp/len(cells))
    for name, mass in (("actuator_a", physical["moving_masses_kg"]["actuator_a"]),
                       ("actuator_b", physical["moving_masses_kg"]["actuator_b"]),
                       ("main", physical["moving_masses_kg"]["main_accumulator"]),
                       ("blowoff", physical["moving_masses_kg"]["blowoff"])):
        for side in ("fluid", "gas"):
            key = f"{name}_piston_{side}"
            solid(key, solid_temperatures_c[key], mass*cp/2)
    for name, mass in (("main_wall", th["main_accumulator_housing_mass_kg"]),
                       ("blowoff_wall", th["blowoff_housing_mass_kg"]),
                       ("reservoir_wall", th["low_pressure_reservoir_housing_mass_kg"])):
        solid(name, solid_temperatures_c[name], mass*cp)
    solid("cooler_region", solid_temperatures_c["cooler_region"], th["cooler_region_heat_capacity_j_k"])
    for name, diameter, length in (("hydraulic", g["hydraulic_pipe_diameter_m"], g["hydraulic_pipe_length_m"]),
                                    ("gas", g["gas_pipe_diameter_m"], g["gas_pipe_length_m"])):
        cells = np.asarray(solid_temperatures_c[name+"_pipe_wall"], dtype=float)
        if cells.ndim != 1 or len(cells) == 0 or th["pipe_outer_diameter_m"] <= diameter:
            raise ValueError("invalid pipe wall geometry or temperatures")
        steel_area = math.pi/4*(th["pipe_outer_diameter_m"]**2-diameter**2)
        for i, temp in enumerate(cells):
            solid(f"{name}_pipe_wall_{i}", temp,
                  steel_area*length/len(cells)*th["metal_density_kg_m3"]*cp)
    for name, temp in (("hot", hot_ambient_c),
                       ("pipe", run["initial_temperatures_c"]["pipe_surroundings"]),
                       ("supply", th["cooler_supply_air_temperature_c"])):
        add_temp("ambient:"+name, temp)

    # Actuator walls retain axial history and change wetted area with piston position.
    af = piston_area(g["actuator_piston_diameter_m"], g["actuator_shaft_diameter_m"])
    ag = piston_area(g["actuator_piston_diameter_m"])
    length = g["actuator_fluid_chamber_length_at_zero_m"]+g["actuator_gas_chamber_length_at_zero_m"]
    metal_area = math.pi/4*(th["actuator_outer_diameter_m"]**2-g["actuator_piston_diameter_m"]**2)
    if metal_area <= 0:
        raise ValueError("actuator outer diameter must exceed bore")
    for label in ("actuator_a", "actuator_b"):
        n = len(solid_temperatures_c[label+"_wall"])
        fluid_areas, gas_areas = actuator_wall_cell_areas(g["actuator_piston_diameter_m"],
            g["actuator_fluid_chamber_length_at_zero_m"], g["actuator_gas_chamber_length_at_zero_m"], positions_m[label], n)
        for i in range(n):
            wall = f"s:{label}_wall_{i}"
            exchange("lb:"+label+"_fluid", wall, th["fluid_to_housing_w_m2_k"]*fluid_areas[i])
            exchange("b:"+label+"_gas", wall, th["gas_to_housing_w_m2_k"]*gas_areas[i])
            exchange("ambient:hot", wall, th["actuator_hot_ambient_to_housing_w_m2_k"]*th["actuator_outer_heat_transfer_area_m2_each"]/n)
            if i:
                exchange(f"s:{label}_wall_{i-1}", wall, th["metal_thermal_conductivity_w_m_k"]*metal_area/(length/n))
        exchange("g:"+label+"_gas", "b:"+label+"_gas", th["gas_core_to_boundary_w_m2_k"]*(sum(gas_areas)+ag))
        exchange("l:"+label+"_fluid", "lb:"+label+"_fluid", th["liquid_core_to_boundary_w_m2_k"]*(sum(fluid_areas)+af))
        exchange("lb:"+label+"_fluid", f"s:{label}_piston_fluid", th["fluid_to_housing_w_m2_k"]*af)
        exchange("b:"+label+"_gas", f"s:{label}_piston_gas", th["gas_to_housing_w_m2_k"]*ag)
        exchange(f"s:{label}_piston_fluid", f"s:{label}_piston_gas", th["actuator_piston_through_conductance_w_k"])

    for prefix, fluid_node, gas_node, fluid_area, gas_area, exposed, piston_g, diameter in (
        ("main", "main_fluid", "main_gas", piston_area(g["main_accumulator_piston_diameter_m"]),
         piston_area(g["main_accumulator_piston_diameter_m"]), th["main_accumulator_exposed_area_m2"], th["main_accumulator_piston_through_conductance_w_k"], g["main_accumulator_piston_diameter_m"]),
        ("blowoff", "blowoff_fluid", "blowoff_gas", piston_area(g["blowoff_piston_diameter_m"], g["blowoff_shaft_diameter_m"]),
         piston_area(g["blowoff_piston_diameter_m"]), th["blowoff_exposed_area_m2"], th["blowoff_piston_through_conductance_w_k"], g["blowoff_piston_diameter_m"])):
        # Provisional geometry: each full chamber volume occupies its piston
        # bore, with no unmodelled remote dead volume. Volume changes with the
        # piston; the wetted bore lengths and heat-transfer areas follow it.
        fluid_wetted = math.pi * diameter * liquid_nodes[fluid_node]["volume_m3"] / fluid_area
        gas_wetted = math.pi * diameter * gas_nodes[gas_node]["volume_m3"] / gas_area
        exchange("lb:"+fluid_node, f"s:{prefix}_wall", th["fluid_to_housing_w_m2_k"]*fluid_wetted)
        exchange("b:"+gas_node, f"s:{prefix}_wall", th["gas_to_housing_w_m2_k"]*gas_wetted)
        exchange("g:"+gas_node, "b:"+gas_node, th["gas_core_to_boundary_w_m2_k"]*(gas_wetted+gas_area))
        exchange("l:"+fluid_node, "lb:"+fluid_node, th["liquid_core_to_boundary_w_m2_k"]*(fluid_wetted+fluid_area))
        exchange("lb:"+fluid_node, f"s:{prefix}_piston_fluid", th["fluid_to_housing_w_m2_k"]*fluid_area)
        exchange("b:"+gas_node, f"s:{prefix}_piston_gas", th["gas_to_housing_w_m2_k"]*gas_area)
        exchange(f"s:{prefix}_piston_fluid", f"s:{prefix}_piston_gas", piston_g)
        exchange(f"s:{prefix}_wall", "s:cooler_region", th["accumulator_and_reservoir_housing_to_cooler_w_m2_k"]*exposed)
    reservoir_area = th["low_pressure_reservoir_exposed_area_m2"]
    exchange("g:low_pressure_reservoir_gas", "b:low_pressure_reservoir_gas", th["gas_core_to_boundary_w_m2_k"]*reservoir_area)
    exchange("b:low_pressure_reservoir_gas", "s:reservoir_wall", th["gas_to_housing_w_m2_k"]*reservoir_area)
    exchange("s:reservoir_wall", "s:cooler_region", th["accumulator_and_reservoir_housing_to_cooler_w_m2_k"]*reservoir_area)

    for kind, diameter, length, medium, k, h, endpoints in (
        ("hydraulic", g["hydraulic_pipe_diameter_m"], g["hydraulic_pipe_length_m"], "l:", fluid["thermal_conductivity_w_m_k"], th["hydraulic_pipe_fluid_to_wall_w_m2_k"], ("hydraulic_inlet_junction", "hydraulic_outlet_junction")),
        ("gas", g["gas_pipe_diameter_m"], g["gas_pipe_length_m"], "g:", gas["thermal_conductivity_w_m_k_at_25c"], th["gas_pipe_gas_to_wall_w_m2_k"], ("actuator_gas_junction", "low_pressure_reservoir_gas"))):
        n = len(solid_temperatures_c[kind+"_pipe_wall"])
        dx = length/n
        inner_area = math.pi*diameter*dx
        steel_area = math.pi/4*(th["pipe_outer_diameter_m"]**2-diameter**2)
        gas_or_fluid_area = math.pi/4*diameter**2
        for i in range(n):
            wall = f"s:{kind}_pipe_wall_{i}"
            content = f"{medium}{kind}_pipe_{i}"
            exchange(content, wall, h*inner_area)
            exchange("ambient:pipe", wall, th["pipe_wall_to_surroundings_w_m2_k"]*math.pi*th["pipe_outer_diameter_m"]*dx)
            if i:
                exchange(f"s:{kind}_pipe_wall_{i-1}", wall, th["metal_thermal_conductivity_w_m_k"]*steel_area/dx)
                exchange(f"{medium}{kind}_pipe_{i-1}", content, k*gas_or_fluid_area/dx)
        exchange(medium+endpoints[0], f"{medium}{kind}_pipe_0", k*gas_or_fluid_area/(dx/2))
        exchange(f"{medium}{kind}_pipe_{n-1}", medium+endpoints[1], k*gas_or_fluid_area/(dx/2))
    exchange("ambient:supply", "s:cooler_region", th["cooler_base_conductance_w_k"] + th["cooler_air_speed_conductance_w_k_per_m_s"]*duct_air_speed_m_s)
    heat["s:cooler_region"] += th["cooler_other_heat_input_w"]

    solid_rates = {}
    for name, value in solid_temperatures_c.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            solid_rates[name] = np.array([heat[f"s:{name}_{i}"]/capacity[f"s:{name}_{i}"] for i in range(len(value))])
        else:
            solid_rates[name] = heat["s:"+name]/capacity["s:"+name]
    return {
        "liquid_heat_w": {name: heat["l:"+name] + heat.get("lb:"+name, 0) for name in liquid_nodes},
        "liquid_boundary_heat_w": {name: heat["lb:"+name] for name in liquid_nodes if "boundary_mass_fraction" in liquid_nodes[name]},
        "gas_heat_w": {name: heat["g:"+name] + heat.get("b:"+name, 0) for name in gas_nodes},
        "gas_boundary_heat_w": {name: heat["b:"+name] for name in gas_nodes if "boundary_mass_fraction" in gas_nodes[name]},
        "solid_temperature_rate_c_s": solid_rates,
        "solid_heat_w": {key[2:]: heat[key] for key in capacity},
        "solid_heat_capacity_j_k": {key[2:]: value for key, value in capacity.items()},
        "external_heat_w": -sum(heat["ambient:"+name] for name in ("hot", "pipe", "supply")) + th["cooler_other_heat_input_w"],
    }
