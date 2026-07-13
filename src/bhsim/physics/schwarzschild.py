"""Schwarzschild metric and derived quantities.

All functions in this module use **geometrized units** (G = c = 1) unless
explicitly noted.  See :mod:`bhsim.physics.units` for conversion helpers.

Coordinate convention: Boyer–Lindquist-like coordinates (t, r, θ, φ) which
reduce to the standard Schwarzschild coordinates for a = 0.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def metric_coefficients(
    r: float | np.ndarray,
    M: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Schwarzschild metric components in Boyer-Lindquist coordinates.

    The line element is:

    .. math::
        ds^2 = -\\left(1 - \\frac{2M}{r}\\right) dt^2
               + \\left(1 - \\frac{2M}{r}\\right)^{-1} dr^2
               + r^2 d\\theta^2
               + r^2 \\sin^2\\theta\\, d\\phi^2

    Because g_{θθ} and g_{φφ} depend on θ this function returns them
    *without* the sin²θ factor — callers must multiply as needed.

    Args:
        r: Radial coordinate(s). Must satisfy r > 2M for the exterior
            metric to be valid; no check is enforced (the caller is
            responsible).
        M: Black hole mass (geometrized).

    Returns:
        Tuple ``(g_tt, g_rr, g_theta_theta, g_phi_phi_over_sin2theta)``
        where each element has the same shape as *r*.

        - ``g_tt = -(1 - 2M/r)``
        - ``g_rr = 1 / (1 - 2M/r)``
        - ``g_theta_theta = r²``
        - ``g_phi_phi_over_sin2theta = r²``  (multiply by sin²θ yourself)
    """
    r = np.asarray(r, dtype=np.float64)
    f = 1.0 - 2.0 * M / r           # lapse function squared
    g_tt = -f
    g_rr = 1.0 / f
    g_thth = r ** 2
    g_phph = r ** 2                  # caller multiplies by sin²θ
    return g_tt, g_rr, g_thth, g_phph


# ---------------------------------------------------------------------------
# Characteristic radii
# ---------------------------------------------------------------------------

def schwarzschild_radius(M: float) -> float:
    """Event-horizon radius for a Schwarzschild (non-spinning) black hole.

    .. math:: r_s = 2M

    Args:
        M: Mass (geometrized).

    Returns:
        Schwarzschild radius.
    """
    return 2.0 * M


def photon_sphere_radius(M: float) -> float:
    """Radius of the photon sphere (unstable circular null orbit).

    .. math:: r_{\\text{ph}} = 3M

    Args:
        M: Mass (geometrized).

    Returns:
        Photon sphere radius.
    """
    return 3.0 * M


def isco_radius(M: float) -> float:
    """Innermost stable circular orbit for a Schwarzschild black hole.

    .. math:: r_{\\text{ISCO}} = 6M

    Args:
        M: Mass (geometrized).

    Returns:
        ISCO radius.
    """
    return 6.0 * M


# ---------------------------------------------------------------------------
# Kinematic quantities
# ---------------------------------------------------------------------------

def escape_velocity(r: float | np.ndarray, M: float) -> np.ndarray:
    """Local escape velocity for a Schwarzschild geometry.

    .. math::
        v_{\\text{esc}} = \\sqrt{\\frac{r_s}{r}} = \\sqrt{\\frac{2M}{r}}

    In geometrized units this equals c (= 1) at the event horizon.

    Args:
        r: Radial coordinate(s).
        M: Mass (geometrized).

    Returns:
        Escape velocity (dimensionless; 1 = c).
    """
    r = np.asarray(r, dtype=np.float64)
    return np.sqrt(2.0 * M / r)


def critical_impact_parameter(M: float) -> float:
    """Critical impact parameter for photon capture.

    A photon with impact parameter b < b_crit is captured by the black
    hole; b > b_crit is scattered.

    .. math:: b_{\\text{crit}} = 3\\sqrt{3}\\, M

    Args:
        M: Mass (geometrized).

    Returns:
        Critical impact parameter.
    """
    return 3.0 * np.sqrt(3.0) * M


# ---------------------------------------------------------------------------
# Effective potential & weak-field deflection
# ---------------------------------------------------------------------------

def effective_potential_photon(
    r: float | np.ndarray,
    L: float,
    M: float,
) -> np.ndarray:
    """Effective potential for photon (null) geodesics.

    For a photon with angular momentum L in the Schwarzschild geometry the
    radial equation of motion can be written as:

    .. math::
        \\left(\\frac{dr}{d\\lambda}\\right)^2 = E^2 - V_{\\text{eff}}(r)

    where

    .. math::
        V_{\\text{eff}}(r) = \\left(1 - \\frac{2M}{r}\\right) \\frac{L^2}{r^2}

    Args:
        r: Radial coordinate(s).
        L: Conserved angular momentum.
        M: Mass (geometrized).

    Returns:
        Effective potential evaluated at *r*.
    """
    r = np.asarray(r, dtype=np.float64)
    return (1.0 - 2.0 * M / r) * (L ** 2) / (r ** 2)


def light_deflection_weak_field(b: float, M: float) -> float:
    """Weak-field light deflection angle for a Schwarzschild black hole.

    Valid when the impact parameter *b* ≫ r_s.

    .. math::
        \\delta\\varphi \\approx \\frac{4M}{b}

    Args:
        b: Impact parameter.
        M: Mass (geometrized).

    Returns:
        Deflection angle in radians.
    """
    if b <= 0.0:
        raise ValueError("Impact parameter must be positive.")
    return 4.0 * M / b
