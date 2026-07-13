#!/usr/bin/env python3
"""Plot null geodesics (photon trajectories) near a Schwarzschild black hole.

Uses the Binet-equation formulation for photon orbits:

    d²u/dφ² + u = 3 M u²

where u = 1/r.  This is rewritten as a first-order system:

    du/dφ  = w
    dw/dφ  = 3 M u² − u

Conserved quantities (E, L) enter via the impact parameter b = L/E.
For a photon approaching from infinity with impact parameter b, the
initial conditions at r → ∞ are  u(0) = 0, w(0) = 1/b.

The script integrates trajectories for several values of b and classifies
them as:
    • **Scattered** (b > b_crit):  deflected but escape
    • **Critical** (b ≈ b_crit):  winds around the photon sphere
    • **Captured** (b < b_crit):  falls into the event horizon

Output: ``scripts/geodesic_plot.png``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np

# ── Ensure the package is importable even if not pip-installed ──────────
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from bhsim.physics.integrators import rk4_integrate
from bhsim.physics.schwarzschild import (
    critical_impact_parameter,
    photon_sphere_radius,
    schwarzschild_radius,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-hole parameters (geometrized, M = 1)
# ---------------------------------------------------------------------------
M = 1.0
R_S = schwarzschild_radius(M)          # 2.0
R_PH = photon_sphere_radius(M)         # 3.0
B_CRIT = critical_impact_parameter(M)  # 3√3 ≈ 5.196


# ---------------------------------------------------------------------------
# Binet ODE system  (u = 1/r)
# ---------------------------------------------------------------------------

def binet_rhs(_phi: float, y: np.ndarray) -> np.ndarray:
    """RHS for the Binet equation.

    State: y = [u, w] where u = 1/r, w = du/dφ.
    """
    u, w = y
    return np.array([w, 3.0 * M * u ** 2 - u])


# ---------------------------------------------------------------------------
# Trace a single geodesic
# ---------------------------------------------------------------------------

def trace_geodesic(
    b: float,
    phi_max: float = 4.0 * np.pi,
    dt: float = 0.001,
    sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a null geodesic with impact parameter *b*.

    Args:
        b: Impact parameter.
        phi_max: Maximum azimuthal angle to integrate over.
        dt: Fixed φ-step for RK4.
        sign: +1 for counter-clockwise, −1 for clockwise.

    Returns:
        ``(x, y)`` arrays of Cartesian coordinates.
    """
    u0 = 1e-8          # start effectively at r → ∞
    w0 = sign / b       # du/dφ = ±1/b  (approaching)

    phi_arr, uy = rk4_integrate(binet_rhs, (0.0, phi_max), np.array([u0, w0]), dt)

    u = uy[:, 0]

    # Stop when the photon hits the horizon (u ≥ 1/r_s) or escapes (u ≤ 0)
    valid = (u > 0) & (u < 1.0 / R_S)
    # Keep only the first contiguous valid block
    if not np.all(valid):
        first_invalid = np.argmax(~valid)
        if first_invalid == 0 and valid[0]:
            first_invalid = len(valid)  # all valid
        elif first_invalid == 0:
            return np.array([]), np.array([])
        phi_arr = phi_arr[:first_invalid]
        u = u[:first_invalid]

    r = 1.0 / u
    x = r * np.cos(phi_arr)
    y = r * np.sin(phi_arr)
    return x, y


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the geodesic plot."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    # ── Draw event horizon ──
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.fill(R_S * np.cos(theta), R_S * np.sin(theta),
            color="#1a1a2e", zorder=5)
    ax.plot(R_S * np.cos(theta), R_S * np.sin(theta),
            color="red", lw=1.5, label=f"Event horizon  r = {R_S:.1f}M", zorder=6)

    # ── Draw photon sphere ──
    ax.plot(R_PH * np.cos(theta), R_PH * np.sin(theta),
            color="yellow", lw=1.0, ls="--",
            label=f"Photon sphere  r = {R_PH:.1f}M", zorder=6)

    # ── Impact parameters to trace ──
    # Scattered (b > b_crit)
    b_scattered = [B_CRIT * f for f in [1.05, 1.15, 1.3, 1.6, 2.0, 3.0]]
    # Near-critical
    b_critical = [B_CRIT * 1.001, B_CRIT * 1.0001]
    # Captured (b < b_crit)
    b_captured = [B_CRIT * f for f in [0.99, 0.9, 0.7, 0.4]]

    # ── Trace & plot scattered rays ──
    for b in b_scattered:
        x, y = trace_geodesic(b, phi_max=3 * np.pi)
        if len(x) > 0:
            ax.plot(x, y, color="cyan", lw=0.7, alpha=0.8, zorder=3)
            # Mirror (negative b)
            x2, y2 = trace_geodesic(b, phi_max=3 * np.pi, sign=-1)
            if len(x2) > 0:
                ax.plot(x2, y2, color="cyan", lw=0.7, alpha=0.8, zorder=3)

    # ── Trace & plot near-critical rays ──
    for b in b_critical:
        x, y = trace_geodesic(b, phi_max=6 * np.pi, dt=0.0005)
        if len(x) > 0:
            ax.plot(x, y, color="#ff6600", lw=1.2, alpha=0.9, zorder=4)
        x2, y2 = trace_geodesic(b, phi_max=6 * np.pi, dt=0.0005, sign=-1)
        if len(x2) > 0:
            ax.plot(x2, y2, color="#ff6600", lw=1.2, alpha=0.9, zorder=4)

    # ── Trace & plot captured rays ──
    for b in b_captured:
        x, y = trace_geodesic(b, phi_max=4 * np.pi)
        if len(x) > 0:
            ax.plot(x, y, color="#ff3366", lw=0.8, alpha=0.8, zorder=3)
        x2, y2 = trace_geodesic(b, phi_max=4 * np.pi, sign=-1)
        if len(x2) > 0:
            ax.plot(x2, y2, color="#ff3366", lw=0.8, alpha=0.8, zorder=3)

    # ── Annotations ──
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("x / M", color="white", fontsize=12)
    ax.set_ylabel("y / M", color="white", fontsize=12)
    ax.set_title(
        "Null Geodesics in Schwarzschild Spacetime (M = 1)",
        color="white", fontsize=14, pad=15,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=1.5, label="Event horizon"),
        Line2D([0], [0], color="yellow", lw=1, ls="--", label="Photon sphere"),
        Line2D([0], [0], color="cyan", lw=1, label=f"Scattered  (b > b_crit ≈ {B_CRIT:.2f}M)"),
        Line2D([0], [0], color="#ff6600", lw=1.2, label="Near-critical  (b ≈ b_crit)"),
        Line2D([0], [0], color="#ff3366", lw=1, label="Captured  (b < b_crit)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              facecolor="#222222", edgecolor="white", labelcolor="white")

    # ── Save ──
    out_path = Path(__file__).resolve().parent / "geodesic_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.info("Saved geodesic plot to %s", out_path)


if __name__ == "__main__":
    main()
