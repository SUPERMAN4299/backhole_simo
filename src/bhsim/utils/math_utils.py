"""Core math utilities for the black hole simulation engine.

Provides vector operations, coordinate transforms, and rotation matrices.
All functions operate on NumPy arrays and are fully decoupled from any
rendering backend.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Normalize a vector to unit length.

    Args:
        v: Input vector of any dimension.

    Returns:
        Unit vector in the same direction as ``v``.

    Raises:
        ValueError: If the vector has zero magnitude.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-30:
        raise ValueError("Cannot normalize a zero-length vector.")
    return v / norm


def dot(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Compute the dot product of two vectors.

    Args:
        a: First vector.
        b: Second vector (same shape as ``a``).

    Returns:
        Scalar dot product a · b.
    """
    return float(np.dot(a, b))


def cross(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the cross product of two 3-vectors.

    Args:
        a: First 3-vector.
        b: Second 3-vector.

    Returns:
        Cross product a × b.
    """
    return np.cross(a, b)


def spherical_to_cartesian(
    r: float, theta: float, phi: float
) -> NDArray[np.floating]:
    """Convert spherical coordinates to Cartesian.

    Uses the physics convention:
        - ``r``     : radial distance (≥ 0)
        - ``theta`` : polar angle from +z axis  [0, π]
        - ``phi``   : azimuthal angle in x-y plane from +x axis [0, 2π)

    Args:
        r: Radial distance.
        theta: Polar angle in radians.
        phi: Azimuthal angle in radians.

    Returns:
        Array [x, y, z].
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z], dtype=np.float64)


def cartesian_to_spherical(
    x: float, y: float, z: float
) -> NDArray[np.floating]:
    """Convert Cartesian coordinates to spherical.

    Args:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.

    Returns:
        Array [r, theta, phi] with r ≥ 0, theta ∈ [0, π], phi ∈ [0, 2π).
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(z / max(r, 1e-30), -1.0, 1.0))
    phi = np.arctan2(y, x) % (2.0 * np.pi)
    return np.array([r, theta, phi], dtype=np.float64)


def rotation_matrix_axis_angle(
    axis: NDArray[np.floating], angle: float
) -> NDArray[np.floating]:
    """Create a 3×3 rotation matrix from an axis and angle (Rodrigues).

    Args:
        axis: Unit rotation axis (will be normalised internally).
        angle: Rotation angle in radians (right-hand rule).

    Returns:
        3×3 rotation matrix R such that R @ v rotates ``v`` by ``angle``
        around ``axis``.
    """
    k = normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    # Skew-symmetric cross-product matrix of k
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    # Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
