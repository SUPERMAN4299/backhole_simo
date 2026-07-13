"""Tests for numerical ODE integrators.

Validates RK4 and adaptive RK4(5) against analytic solutions for:
- Simple harmonic oscillator
- Exponential decay
- Circular orbits under inverse-square gravity
- Energy conservation over many orbits
"""

from __future__ import annotations

import numpy as np
import pytest

from bhsim.physics.integrators import adaptive_rk45, rk4_integrate


# ---------------------------------------------------------------------------
# Simple harmonic oscillator:  x'' + ω²x = 0
# ---------------------------------------------------------------------------

def _sho_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """dy/dt for SHO with ω = 1.  y = [x, v]."""
    return np.array([y[1], -y[0]])


class TestRK4SimpleHarmonicOscillator:
    """Validate RK4 on x(t) = A cos(ωt)."""

    def test_period_and_amplitude(self) -> None:
        """After one full period T = 2π, x should return to x(0)."""
        A = 1.0
        y0 = np.array([A, 0.0])  # x(0) = A, v(0) = 0
        T = 2.0 * np.pi

        t, y = rk4_integrate(_sho_rhs, (0.0, T), y0, dt=0.001)

        # Position after one period should be ≈ A
        assert y[-1, 0] == pytest.approx(A, abs=1e-4)
        # Velocity should be ≈ 0
        assert y[-1, 1] == pytest.approx(0.0, abs=1e-4)

    def test_amplitude_conservation(self) -> None:
        """Amplitude (energy proxy) should be conserved over many cycles."""
        y0 = np.array([1.0, 0.0])
        t, y = rk4_integrate(_sho_rhs, (0.0, 20.0 * np.pi), y0, dt=0.005)

        # E = 0.5 (x² + v²) should be constant = 0.5
        energy = 0.5 * (y[:, 0] ** 2 + y[:, 1] ** 2)
        assert np.max(np.abs(energy - 0.5)) < 1e-4


# ---------------------------------------------------------------------------
# Exponential decay:  dx/dt = -k x
# ---------------------------------------------------------------------------

def _decay_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """Exponential decay with k = 1."""
    return np.array([-y[0]])


class TestRK4ExponentialDecay:
    """Compare RK4 to x(t) = x0 exp(-t)."""

    def test_exponential_accuracy(self) -> None:
        x0 = 1.0
        t, y = rk4_integrate(_decay_rhs, (0.0, 5.0), np.array([x0]), dt=0.01)
        analytic = x0 * np.exp(-t)
        assert np.max(np.abs(y[:, 0] - analytic)) < 1e-6


# ---------------------------------------------------------------------------
# Circular orbit (inverse-square force in 2D)
# ---------------------------------------------------------------------------

def _gravity_2d(t: float, y: np.ndarray) -> np.ndarray:
    """2D inverse-square gravitational force.

    State y = [x, y_coord, vx, vy].
    F = -GM r̂ / r²  with GM = 1.
    """
    x, yc, vx, vy = y
    r = np.sqrt(x ** 2 + yc ** 2)
    r3 = r ** 3
    ax = -x / r3
    ay = -yc / r3
    return np.array([vx, vy, ax, ay])


class TestRK4CircularOrbit:
    """Test a circular orbit with GM = 1, r = 1 → v = 1, T = 2π."""

    def test_circular_orbit_closure(self) -> None:
        """After one period the particle should return to its start."""
        # Circular orbit: r=1, v=1 tangential
        y0 = np.array([1.0, 0.0, 0.0, 1.0])
        T = 2.0 * np.pi

        t, y = rk4_integrate(_gravity_2d, (0.0, T), y0, dt=0.001)

        assert y[-1, 0] == pytest.approx(1.0, abs=1e-3)
        assert y[-1, 1] == pytest.approx(0.0, abs=1e-3)

    def test_rk4_energy_conservation(self) -> None:
        """Total energy E = v²/2 - 1/r should be conserved over 10 orbits."""
        y0 = np.array([1.0, 0.0, 0.0, 1.0])
        T = 20.0 * np.pi  # 10 orbits

        t, y = rk4_integrate(_gravity_2d, (0.0, T), y0, dt=0.001)

        r = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        v2 = y[:, 2] ** 2 + y[:, 3] ** 2
        energy = 0.5 * v2 - 1.0 / r
        E0 = energy[0]

        # Energy drift should be tiny over 10 orbits
        assert np.max(np.abs(energy - E0)) < 1e-5


# ---------------------------------------------------------------------------
# Adaptive integrator accuracy
# ---------------------------------------------------------------------------

class TestAdaptiveRK45:
    """Validate adaptive RK4(5) Fehlberg integrator."""

    def test_exponential_accuracy(self) -> None:
        """Adaptive integrator should match exp(-t) within tolerance."""
        t, y = adaptive_rk45(
            _decay_rhs,
            (0.0, 5.0),
            np.array([1.0]),
            dt_init=0.1,
            atol=1e-10,
            rtol=1e-10,
        )
        analytic = np.exp(-t)
        assert np.max(np.abs(y[:, 0] - analytic)) < 1e-7

    def test_sho_adaptive(self) -> None:
        """Adaptive integrator on SHO — period and amplitude check."""
        T = 2.0 * np.pi
        t, y = adaptive_rk45(
            _sho_rhs,
            (0.0, T),
            np.array([1.0, 0.0]),
            dt_init=0.1,
            atol=1e-10,
            rtol=1e-10,
        )
        assert y[-1, 0] == pytest.approx(1.0, abs=1e-6)
        assert y[-1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_adaptive_uses_fewer_steps_for_smooth_ode(self) -> None:
        """Adaptive should take far fewer steps than a fine fixed grid."""
        t, y = adaptive_rk45(
            _decay_rhs,
            (0.0, 5.0),
            np.array([1.0]),
            dt_init=0.1,
            atol=1e-8,
            rtol=1e-8,
        )
        # Fixed-step at dt=0.001 would use 5000 steps;
        # adaptive should use far fewer (typically < 100 for exp decay).
        assert len(t) < 500
