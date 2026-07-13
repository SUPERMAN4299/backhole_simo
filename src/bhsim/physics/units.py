"""Geometrized unit system and SI conversions.

Unit Convention
===============
Throughout the simulation engine we use **geometrized units** where:

    G = c = 1

In these units:
    - Mass, length, and time all have the same dimension.
    - A mass M corresponds to a length G·M/c² and a time G·M/c³.
    - The Schwarzschild radius is simply r_s = 2M.

This module provides conversion helpers between SI and geometrized units
so that physical constants and astrophysical data can be ingested in SI
and all internal computation carried out in the cleaner G = c = 1 system.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fundamental constants (SI)
# ---------------------------------------------------------------------------

C_SI: float = 2.998e8
"""Speed of light in vacuum [m/s]."""

G_SI: float = 6.674e-11
"""Newtonian gravitational constant [m³ kg⁻¹ s⁻²]."""

M_SUN_KG: float = 1.989e30
"""Solar mass [kg]."""

# Derived conversion factors
LENGTH_PER_MASS: float = G_SI / (C_SI ** 2)
"""Metres per kilogram in geometrized units: G/c² [m/kg]."""

TIME_PER_MASS: float = G_SI / (C_SI ** 3)
"""Seconds per kilogram in geometrized units: G/c³ [s/kg]."""


# ---------------------------------------------------------------------------
# Supported unit types for conversion
# ---------------------------------------------------------------------------

class UnitType(str, Enum):
    """Physical quantity types for SI ↔ geometrized conversion."""

    MASS = "mass"
    LENGTH = "length"
    TIME = "time"


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def to_geometrized(value: float, unit_type: UnitType | str) -> float:
    """Convert a value from SI to geometrized (G = c = 1) units.

    In geometrized units every quantity is expressed in metres:
        mass   → G·m / c²   [m]
        length → unchanged   [m]
        time   → c · t       [m]

    Args:
        value: Value in SI units.
        unit_type: One of ``"mass"``, ``"length"``, or ``"time"``.

    Returns:
        Value in geometrized units (metres).

    Raises:
        ValueError: If *unit_type* is not recognised.
    """
    unit_type = UnitType(unit_type)
    if unit_type is UnitType.MASS:
        return value * LENGTH_PER_MASS          # kg → m
    elif unit_type is UnitType.LENGTH:
        return value                            # m → m (identity)
    elif unit_type is UnitType.TIME:
        return value * C_SI                     # s → m
    raise ValueError(f"Unknown unit type: {unit_type}")  # pragma: no cover


def from_geometrized(value: float, unit_type: UnitType | str) -> float:
    """Convert a value from geometrized (G = c = 1) to SI units.

    Args:
        value: Value in geometrized units (metres).
        unit_type: Target SI unit type.

    Returns:
        Value in SI units.

    Raises:
        ValueError: If *unit_type* is not recognised.
    """
    unit_type = UnitType(unit_type)
    if unit_type is UnitType.MASS:
        return value / LENGTH_PER_MASS          # m → kg
    elif unit_type is UnitType.LENGTH:
        return value                            # m → m
    elif unit_type is UnitType.TIME:
        return value / C_SI                     # m → s
    raise ValueError(f"Unknown unit type: {unit_type}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Schwarzschild radius helpers
# ---------------------------------------------------------------------------

def schwarzschild_radius_si(mass_kg: float) -> float:
    """Schwarzschild radius in SI (metres).

    .. math::
        r_s = \\frac{2 G M}{c^2}

    Args:
        mass_kg: Mass in kilograms.

    Returns:
        Schwarzschild radius in metres.
    """
    return 2.0 * G_SI * mass_kg / (C_SI ** 2)


def schwarzschild_radius_geo(mass: float) -> float:
    """Schwarzschild radius in geometrized units.

    .. math::
        r_s = 2 M

    Args:
        mass: Mass in geometrized units (metres).

    Returns:
        Schwarzschild radius (metres) — simply ``2 * mass``.
    """
    return 2.0 * mass
