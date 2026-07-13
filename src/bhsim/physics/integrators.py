"""Numerical ODE integrators for geodesic and orbit computation.

Provides fixed-step RK4 and adaptive RK4(5) Fehlberg integrators that
operate on arbitrary state vectors (NumPy arrays).  All integrators are
CPU-only and fully decoupled from any rendering backend.

Usage::

    from bhsim.physics.integrators import rk4_integrate

    def harmonic(t, y):
        # y = [x, v], dx/dt = v, dv/dt = -x
        return np.array([y[1], -y[0]])

    sol = rk4_integrate(harmonic, (0.0, 2*np.pi), np.array([1.0, 0.0]), dt=0.01)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias for the right-hand side function  f(t, y) → dy/dt
RHSFunc = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]


# ---------------------------------------------------------------------------
# Fixed-step RK4
# ---------------------------------------------------------------------------

def rk4_step(
    f: RHSFunc,
    t: float,
    y: NDArray[np.floating],
    dt: float,
) -> NDArray[np.floating]:
    """Perform a single classical 4th-order Runge-Kutta step.

    Advances the state *y* from time *t* to *t + dt* using::

        k1 = f(t,           y)
        k2 = f(t + dt/2,    y + dt/2 * k1)
        k3 = f(t + dt/2,    y + dt/2 * k2)
        k4 = f(t + dt,      y + dt   * k3)
        y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        f: Right-hand side of the ODE system  dy/dt = f(t, y).
        t: Current time.
        y: Current state vector.
        dt: Time step.

    Returns:
        Updated state vector at *t + dt*.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    f: RHSFunc,
    t_span: tuple[float, float],
    y0: NDArray[np.floating],
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Integrate an ODE system with fixed-step RK4.

    Args:
        f: Right-hand side  dy/dt = f(t, y).
        t_span: ``(t_start, t_end)``.
        y0: Initial state vector.
        dt: Fixed time step.

    Returns:
        ``(t_array, y_array)`` where ``t_array`` has shape ``(N,)`` and
        ``y_array`` has shape ``(N, len(y0))``.
    """
    t_start, t_end = t_span
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_steps = int(np.ceil((t_end - t_start) / dt))
    t_arr = np.empty(n_steps + 1, dtype=np.float64)
    y_arr = np.empty((n_steps + 1, len(y0)), dtype=np.float64)

    t_arr[0] = t_start
    y_arr[0] = y0.copy()

    t = t_start
    y = y0.copy()

    for i in range(1, n_steps + 1):
        # Don't overshoot
        h = min(dt, t_end - t)
        y = rk4_step(f, t, y, h)
        t += h
        t_arr[i] = t
        y_arr[i] = y

    return t_arr, y_arr


# ---------------------------------------------------------------------------
# RK4(5) Fehlberg (embedded pair for error estimation)
# ---------------------------------------------------------------------------

# Butcher tableau coefficients for RKF45
_A2 = 1.0 / 4.0
_A3 = 3.0 / 8.0
_A4 = 12.0 / 13.0
_A5 = 1.0
_A6 = 1.0 / 2.0

_B21 = 1.0 / 4.0
_B31 = 3.0 / 32.0;      _B32 = 9.0 / 32.0
_B41 = 1932.0 / 2197.0;  _B42 = -7200.0 / 2197.0;  _B43 = 7296.0 / 2197.0
_B51 = 439.0 / 216.0;    _B52 = -8.0;               _B53 = 3680.0 / 513.0;    _B54 = -845.0 / 4104.0
_B61 = -8.0 / 27.0;      _B62 = 2.0;                _B63 = -3544.0 / 2565.0;  _B64 = 1859.0 / 4104.0;  _B65 = -11.0 / 40.0

# 4th-order weights
_C1 = 25.0 / 216.0;  _C3 = 1408.0 / 2565.0;  _C4 = 2197.0 / 4104.0;  _C5 = -1.0 / 5.0

# 5th-order weights (for error estimate)
_D1 = 16.0 / 135.0;  _D3 = 6656.0 / 12825.0;  _D4 = 28561.0 / 56430.0;  _D5 = -9.0 / 50.0;  _D6 = 2.0 / 55.0


def rkf45_step(
    f: RHSFunc,
    t: float,
    y: NDArray[np.floating],
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Single RK4(5) Fehlberg step with embedded error estimate.

    Args:
        f: Right-hand side  dy/dt = f(t, y).
        t: Current time.
        y: Current state vector.
        dt: Proposed time step.

    Returns:
        ``(y4, y5, error)`` where *y4* is the 4th-order solution,
        *y5* is the 5th-order solution, and *error* = |y5 - y4|
        element-wise.
    """
    k1 = dt * f(t, y)
    k2 = dt * f(t + _A2 * dt, y + _B21 * k1)
    k3 = dt * f(t + _A3 * dt, y + _B31 * k1 + _B32 * k2)
    k4 = dt * f(t + _A4 * dt, y + _B41 * k1 + _B42 * k2 + _B43 * k3)
    k5 = dt * f(t + _A5 * dt, y + _B51 * k1 + _B52 * k2 + _B53 * k3 + _B54 * k4)
    k6 = dt * f(t + _A6 * dt, y + _B61 * k1 + _B62 * k2 + _B63 * k3 + _B64 * k4 + _B65 * k5)

    y4 = y + _C1 * k1 + _C3 * k3 + _C4 * k4 + _C5 * k5
    y5 = y + _D1 * k1 + _D3 * k3 + _D4 * k4 + _D5 * k5 + _D6 * k6

    error = np.abs(y5 - y4)
    return y4, y5, error


def adaptive_rk45(
    f: RHSFunc,
    t_span: tuple[float, float],
    y0: NDArray[np.floating],
    dt_init: float = 0.01,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    max_steps: int = 1_000_000,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Adaptive-step RK4(5) Fehlberg integration.

    Step size is adjusted so that the local truncation error satisfies:

        err_i ≤ atol + rtol * |y_i|

    for every component *i* of the state vector.

    Args:
        f: Right-hand side  dy/dt = f(t, y).
        t_span: ``(t_start, t_end)``.
        y0: Initial state vector.
        dt_init: Initial step size guess.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        max_steps: Safety limit on number of steps.

    Returns:
        ``(t_array, y_array)`` with shapes ``(N,)`` and ``(N, len(y0))``.

    Raises:
        RuntimeError: If *max_steps* is exceeded.
    """
    t_start, t_end = t_span
    t = t_start
    y = y0.copy().astype(np.float64)
    dt = dt_init

    ts = [t]
    ys = [y.copy()]

    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0

    for _ in range(max_steps):
        if t >= t_end:
            break

        # Don't overshoot the endpoint
        dt = min(dt, t_end - t)
        if dt <= 0:
            break

        _y4, y5, error = rkf45_step(f, t, y, dt)

        # Compute scaled error norm
        scale = atol + rtol * np.abs(y)
        err_norm = np.max(error / scale) if np.any(scale > 0) else 0.0

        if err_norm <= 1.0:
            # Accept step (use 5th-order solution for local extrapolation)
            t += dt
            y = y5.copy()
            ts.append(t)
            ys.append(y.copy())

            # Grow step size
            if err_norm > 1e-30:
                factor = safety * (1.0 / err_norm) ** 0.2
                factor = min(max(factor, min_factor), max_factor)
                dt *= factor
            else:
                dt *= max_factor
        else:
            # Reject step — shrink
            factor = safety * (1.0 / err_norm) ** 0.25
            factor = max(factor, min_factor)
            dt *= factor
    else:
        raise RuntimeError(
            f"adaptive_rk45 exceeded {max_steps} steps without reaching t_end."
        )

    return np.array(ts), np.array(ys)
