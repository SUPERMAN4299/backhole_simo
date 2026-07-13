"""Physical constants and unit system for black hole simulation.

This module defines the fundamental physical constants used throughout the
simulation. Two unit systems are supported:

**SI Units** (for display / conversion):
    Standard metric units — metres, kilograms, seconds.

**Geometrized Units** (for ray-tracing maths):
    G = c = 1.  In these units mass has dimensions of length, and the
    Schwarzschild radius of a black hole of mass M is simply r_s = 2M.
    All internal physics computations use geometrized units; conversion
    helpers are provided for interfacing with the outside world.

References:
    CODATA 2018 recommended values — https://physics.nist.gov/cuu/Constants/
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Fundamental constants  (SI)
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT: float = 2.998e8
"""Speed of light in vacuum  [m s⁻¹]."""

GRAVITATIONAL_CONSTANT: float = 6.67430e-11
"""Newtonian gravitational constant  [m³ kg⁻¹ s⁻²]."""

SOLAR_MASS: float = 1.989e30
"""Solar mass  [kg]."""

PLANCK_CONSTANT: float = 6.62607015e-34
"""Planck constant  [J s]."""

BOLTZMANN_CONSTANT: float = 1.380649e-23
"""Boltzmann constant  [J K⁻¹]."""

# ---------------------------------------------------------------------------
# Derived quantities  (SI)
# ---------------------------------------------------------------------------

SCHWARZSCHILD_RADIUS_SUN: float = (
    2.0 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / (SPEED_OF_LIGHT ** 2)
)
"""Schwarzschild radius of the Sun  [m]  (~2953 m)."""

# ---------------------------------------------------------------------------
# Short aliases (for compact formulae)
# ---------------------------------------------------------------------------

c: float = SPEED_OF_LIGHT
G: float = GRAVITATIONAL_CONSTANT
M_sun: float = SOLAR_MASS
h: float = PLANCK_CONSTANT
k_B: float = BOLTZMANN_CONSTANT
r_s_sun: float = SCHWARZSCHILD_RADIUS_SUN


# ---------------------------------------------------------------------------
# Geometrized-unit helpers
# ---------------------------------------------------------------------------

def mass_to_length(mass_kg: float) -> float:
    """Convert a mass in kg to its equivalent length in geometrized units.

    In geometrized units (G = c = 1), mass has dimensions of length:
        L = G·M / c²

    Args:
        mass_kg: Mass in kilograms.

    Returns:
        Equivalent length in metres.
    """
    return G * mass_kg / (c ** 2)


def schwarzschild_radius(mass_kg: float) -> float:
    """Compute the Schwarzschild radius for a given mass.

    r_s = 2GM / c²

    Args:
        mass_kg: Mass of the black hole in kilograms.

    Returns:
        Schwarzschild radius in metres.
    """
    return 2.0 * mass_to_length(mass_kg)


def isco_radius(mass_kg: float, spin: float = 0.0) -> float:
    """Innermost stable circular orbit radius for a Kerr black hole.

    For a Schwarzschild black hole (spin=0):  r_isco = 6 GM/c² = 3 r_s.
    The full Kerr expression is used when spin ≠ 0.

    Args:
        mass_kg: Mass of the black hole in kilograms.
        spin: Dimensionless spin parameter a* ∈ [0, 1).

    Returns:
        ISCO radius in metres.
    """
    m = mass_to_length(mass_kg)  # GM/c² in metres

    if abs(spin) < 1e-12:
        return 6.0 * m

    a = spin
    z1 = 1.0 + (1.0 - a ** 2) ** (1 / 3) * (
        (1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3)
    )
    z2 = math.sqrt(3.0 * a ** 2 + z1 ** 2)
    # Prograde orbit
    r_isco_m = m * (3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))
    return r_isco_m
