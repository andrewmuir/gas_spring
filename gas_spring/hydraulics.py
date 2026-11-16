"""Hydraulic restrictions and pipe properties, using SI units."""

from dataclasses import dataclass
import math


def piston_area(diameter_m: float, shaft_diameter_m: float = 0.0) -> float:
    if not (diameter_m > 0 and 0 <= shaft_diameter_m < diameter_m):
        raise ValueError("diameters must satisfy 0 <= shaft < piston")
    return math.pi / 4 * (diameter_m**2 - shaft_diameter_m**2)


@dataclass(frozen=True)
class DigressiveCurve:
    """Pressure magnitude p=A*q/(B+q), fitted to two force/speed points."""

    limit_pa: float
    scale_m3_s: float

    @classmethod
    def from_points(cls, area_m2, speed1_m_s, force1_n, speed2_m_s, force2_n):
        q1, q2 = area_m2 * speed1_m_s, area_m2 * speed2_m_s
        if not (area_m2 > 0 and 0 < q1 < q2 and 0 < force1_n < force2_n):
            raise ValueError(
                "points must have positive, strictly increasing speeds and forces"
            )
        p1, p2 = force1_n / area_m2, force2_n / area_m2
        denom = p1 * q2 - p2 * q1
        if denom <= 0:
            raise ValueError(
                "points must be strictly digressive (force/speed decreases)"
            )
        scale = q1 * q2 * (p2 - p1) / denom
        return cls(p1 * (scale + q1) / q1, scale)

    def pressure_drop(self, flow_m3_s: float) -> float:
        """Signed pressure drop; pressure times flow is non-negative."""
        q = abs(flow_m3_s)
        return math.copysign(self.limit_pa * q / (self.scale_m3_s + q), flow_m3_s)


def restriction_pressure_drop(flow_m3_s, compression, extension):
    """Positive flow is compression; return signed inlet-to-outlet pressure loss."""
    return (compression if flow_m3_s >= 0 else extension).pressure_drop(flow_m3_s)


def regularized_restriction_pressure_drop(
    flow_m3_s, compression, extension, tail_fraction=0.95, zero_flow_transition_m3_s=0.0
):
    """Replaceable restriction law with a linear, tangent high-flow tail.

    The original saturating curve has no finite inverse above its asymptote.
    Past `tail_fraction` of that asymptote, the tangent continuation is an
    explicitly provisional extrapolation. The supplied fit points remain on
    the original curve when their pressures fall below this threshold.
    """
    if not 0 < tail_fraction < 1 or not math.isfinite(flow_m3_s):
        raise ValueError("invalid tail fraction or flow")
    if zero_flow_transition_m3_s:
        if (
            not math.isfinite(zero_flow_transition_m3_s)
            or zero_flow_transition_m3_s < 0
        ):
            raise ValueError("invalid zero-flow transition")
        weight = 0.5 * (1 + math.tanh(flow_m3_s / zero_flow_transition_m3_s))
        c = regularized_restriction_pressure_drop(
            flow_m3_s, compression, compression, tail_fraction
        )
        e = regularized_restriction_pressure_drop(
            flow_m3_s, extension, extension, tail_fraction
        )
        return weight * c + (1 - weight) * e
    curve = compression if flow_m3_s >= 0 else extension
    q = abs(flow_m3_s)
    q_cut = curve.scale_m3_s * tail_fraction / (1 - tail_fraction)
    if q <= q_cut:
        return curve.pressure_drop(flow_m3_s)
    p_cut = curve.limit_pa * tail_fraction
    slope = curve.limit_pa * curve.scale_m3_s / (curve.scale_m3_s + q_cut) ** 2
    return math.copysign(p_cut + slope * (q - q_cut), flow_m3_s)


def restriction_flow(pressure_drop_pa, compression, extension, tail_fraction=0.95):
    """Invert the regularized signed restriction law for volumetric flow."""
    if not 0 < tail_fraction < 1 or not math.isfinite(pressure_drop_pa):
        raise ValueError("invalid tail fraction or pressure drop")
    curve = compression if pressure_drop_pa >= 0 else extension
    p = abs(pressure_drop_pa)
    p_cut = curve.limit_pa * tail_fraction
    if p <= p_cut:
        q = curve.scale_m3_s * p / (curve.limit_pa - p)
    else:
        q_cut = curve.scale_m3_s * tail_fraction / (1 - tail_fraction)
        slope = curve.limit_pa * curve.scale_m3_s / (curve.scale_m3_s + q_cut) ** 2
        q = q_cut + (p - p_cut) / slope
    return math.copysign(q, pressure_drop_pa)


def restriction_power_slope(
    flow_m3_s, compression, extension, tail_fraction=0.95, zero_flow_transition_m3_s=0.0
):
    """Derivative of dissipated power q*dp with respect to signed flow."""
    q = flow_m3_s
    pressure = regularized_restriction_pressure_drop(
        q, compression, extension, tail_fraction, zero_flow_transition_m3_s
    )

    def slope(curve):
        cutoff = curve.scale_m3_s * tail_fraction / (1 - tail_fraction)
        magnitude = min(abs(q), cutoff)
        return curve.limit_pa * curve.scale_m3_s / (curve.scale_m3_s + magnitude) ** 2

    if zero_flow_transition_m3_s:
        transition = math.tanh(q / zero_flow_transition_m3_s)
        weight = 0.5 * (1 + transition)
        pc = regularized_restriction_pressure_drop(
            q, compression, compression, tail_fraction
        )
        pe = regularized_restriction_pressure_drop(
            q, extension, extension, tail_fraction
        )
        derivative = weight * slope(compression) + (1 - weight) * slope(extension)
        derivative += (
            0.5 * (1 - transition * transition) * (pc - pe) / zero_flow_transition_m3_s
        )
    else:
        derivative = slope(compression if q >= 0 else extension)
    return pressure + q * derivative


def pipe_power_derivatives(
    flow_m3_s, density_kg_m3, viscosity_pa_s, length_m, diameter_m, roughness_m=1e-6
):
    """Derivatives of Darcy dissipation with respect to flow and viscosity."""
    if min(density_kg_m3, viscosity_pa_s, length_m, diameter_m) <= 0 or roughness_m < 0:
        raise ValueError("invalid pipe or fluid properties")
    q = abs(flow_m3_s)
    if q == 0:
        return 0.0, 0.0
    area = math.pi * diameter_m**2 / 4
    reynolds = density_kg_m3 * q * diameter_m / (area * viscosity_pa_s)
    if reynolds <= 2300:
        resistance_per_mu = 128 * length_m / (math.pi * diameter_m**4)
        return (
            2 * resistance_per_mu * viscosity_pa_s * flow_m3_s,
            resistance_per_mu * q * q,
        )
    laminar = 64 / reynolds
    dlaminar = -64 / reynolds**2
    inside = (roughness_m / (3.7 * diameter_m)) ** 1.11 + 6.9 / reynolds
    h = -1.8 * math.log10(inside)
    turbulent = h**-2
    dturbulent = -2 * h**-3 * (1.8 * 6.9 / (math.log(10) * inside * reynolds**2))
    blend = min(1.0, (reynolds - 2300) / 1700)
    dblend = 1 / 1700 if reynolds < 4000 else 0.0
    friction = (1 - blend) * laminar + blend * turbulent
    dfriction = (
        (1 - blend) * dlaminar + blend * dturbulent + dblend * (turbulent - laminar)
    )
    factor = (length_m / diameter_m) * density_kg_m3 / (2 * area * area)
    dq = math.copysign(
        factor * q * q * (3 * friction + reynolds * dfriction), flow_m3_s
    )
    dmu = -factor * q**3 * dfriction * reynolds / viscosity_pa_s
    return dq, dmu


def pipe_inertance(density_kg_m3, length_m, diameter_m):
    if density_kg_m3 <= 0 or length_m <= 0 or diameter_m <= 0:
        raise ValueError("density, length and diameter must be positive")
    return density_kg_m3 * length_m / (math.pi * diameter_m**2 / 4)


def pipe_pressure_drop(
    flow_m3_s, density_kg_m3, viscosity_pa_s, length_m, diameter_m, roughness_m=1e-6
):
    """Quasi-steady Darcy loss; blend laminar/turbulent in transitional range.

    Haaland approximation for turbulent friction; separate from pipe inertance.
    """
    if min(density_kg_m3, viscosity_pa_s, length_m, diameter_m) <= 0 or roughness_m < 0:
        raise ValueError("invalid pipe or fluid properties")
    area = math.pi * diameter_m**2 / 4
    v = abs(flow_m3_s) / area
    if not v:
        return 0.0
    re = density_kg_m3 * v * diameter_m / viscosity_pa_s
    laminar = 64 / re
    turbulent = (
        -1.8 * math.log10((roughness_m / (3.7 * diameter_m)) ** 1.11 + 6.9 / re)
    ) ** -2
    blend = min(1.0, max(0.0, (re - 2300) / 1700))
    f = (1 - blend) * laminar + blend * turbulent
    return math.copysign(
        f * length_m / diameter_m * density_kg_m3 * v * v / 2, flow_m3_s
    )


def gas_pipe_mass_flow(
    pressure_a_pa,
    pressure_b_pa,
    temperature_k,
    gas_constant_j_kg_k,
    viscosity_pa_s,
    length_m,
    diameter_m,
    roughness_m=1e-6,
    pressure_difference_pa=None,
):
    """Quasi-steady gas mass flow from A to B using mean-density Darcy loss.

    Pressure is absolute and temperature is in kelvin. This is a provisional
    low-Mach approximation; choking and gas-pipe inertance are not represented.
    """
    from scipy.optimize import brentq

    vals = (
        pressure_a_pa,
        pressure_b_pa,
        temperature_k,
        gas_constant_j_kg_k,
        viscosity_pa_s,
        length_m,
        diameter_m,
    )
    if (
        not all(math.isfinite(v) and v > 0 for v in vals)
        or not math.isfinite(roughness_m)
        or roughness_m < 0
    ):
        raise ValueError("invalid gas pipe conditions")
    delta_p = (
        pressure_a_pa - pressure_b_pa
        if pressure_difference_pa is None
        else float(pressure_difference_pa)
    )
    if not math.isfinite(delta_p):
        raise ValueError("invalid gas pressure difference")
    if delta_p == 0:
        return 0.0
    density = (pressure_a_pa + pressure_b_pa) / (
        2 * gas_constant_j_kg_k * temperature_k
    )
    target = abs(delta_p)
    laminar_volume_flow = (
        target * math.pi * diameter_m**4 / (128 * viscosity_pa_s * length_m)
    )
    laminar_re = (
        4 * density * laminar_volume_flow / (math.pi * diameter_m * viscosity_pa_s)
    )
    if laminar_re <= 2300:
        return math.copysign(density * laminar_volume_flow, delta_p)

    def loss(mass_flow):
        return pipe_pressure_drop(
            mass_flow / density,
            density,
            viscosity_pa_s,
            length_m,
            diameter_m,
            roughness_m,
        )

    upper = density * math.pi * diameter_m**2 / 4 * 0.01
    while loss(upper) < target:
        upper *= 2
        if not math.isfinite(upper):
            raise ValueError("could not bracket gas pipe flow")
    return math.copysign(
        brentq(lambda flow: loss(flow) - target, 0.0, upper, xtol=1e-15), delta_p
    )


def liquid_pipe_flow(
    pressure_drop_pa,
    density_kg_m3,
    viscosity_pa_s,
    length_m,
    diameter_m,
    roughness_m=1e-6,
):
    """Invert the quasi-steady liquid Darcy loss for volumetric flow."""
    from scipy.optimize import brentq

    if not math.isfinite(pressure_drop_pa):
        raise ValueError("pressure drop must be finite")
    # Call once to validate all properties, even at zero pressure difference.
    pipe_pressure_drop(
        0.0, density_kg_m3, viscosity_pa_s, length_m, diameter_m, roughness_m
    )
    if pressure_drop_pa == 0:
        return 0.0
    target = abs(pressure_drop_pa)
    upper = math.pi * diameter_m**2 / 4 * 0.01

    def loss(flow):
        return pipe_pressure_drop(
            flow, density_kg_m3, viscosity_pa_s, length_m, diameter_m, roughness_m
        )

    while loss(upper) < target:
        upper *= 2
        if not math.isfinite(upper):
            raise ValueError("could not bracket liquid pipe flow")
    return math.copysign(
        brentq(lambda flow: loss(flow) - target, 0.0, upper, xtol=1e-18),
        pressure_drop_pa,
    )
