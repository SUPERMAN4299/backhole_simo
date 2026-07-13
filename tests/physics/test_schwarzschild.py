"""Tests for Schwarzschild metric and derived quantities.

Each test compares against known analytic results.  All physics functions
use geometrized units (G = c = 1) unless explicitly testing SI conversions.
"""

from __future__ import annotations

import numpy as np
import pytest

from bhsim.physics import schwarzschild as sch
from bhsim.physics.units import (
    C_SI,
    G_SI,
    M_SUN_KG,
    schwarzschild_radius_si,
    to_geometrized,
)


# ---------------------------------------------------------------------------
# Schwarzschild radius
# ---------------------------------------------------------------------------

class TestSchwarzschildRadius:
    """Tests for the event horizon radius."""

    def test_schwarzschild_radius_sun(self) -> None:
        """r_s(Sun) ≈ 2953 m in SI."""
        r_s = schwarzschild_radius_si(M_SUN_KG)
        assert r_s == pytest.approx(2953.0, rel=0.01)

    def test_schwarzschild_radius_sgr_a_star(self) -> None:
        """r_s(Sgr A*) ≈ 1.2 × 10¹⁰ m.

        Sgr A* mass ≈ 4.0 × 10⁶ M_sun.
        """
        mass_sgr = 4.0e6 * M_SUN_KG
        r_s = schwarzschild_radius_si(mass_sgr)
        assert r_s == pytest.approx(1.2e10, rel=0.02)

    def test_geometrized(self) -> None:
        """r_s = 2M in geometrized units."""
        M = 5.0
        assert sch.schwarzschild_radius(M) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Photon sphere and ISCO
# ---------------------------------------------------------------------------

class TestCharacteristicRadii:
    """Tests for photon sphere, ISCO, and critical impact parameter."""

    def test_photon_sphere(self) -> None:
        """r_ph = 1.5 × r_s = 3M."""
        M = 1.0
        r_s = sch.schwarzschild_radius(M)
        r_ph = sch.photon_sphere_radius(M)
        assert r_ph == pytest.approx(1.5 * r_s)
        assert r_ph == pytest.approx(3.0 * M)

    def test_isco(self) -> None:
        """r_isco = 3 × r_s = 6M."""
        M = 1.0
        r_s = sch.schwarzschild_radius(M)
        r_isco = sch.isco_radius(M)
        assert r_isco == pytest.approx(3.0 * r_s)
        assert r_isco == pytest.approx(6.0 * M)

    def test_critical_impact_parameter(self) -> None:
        """b_crit = 3√3 M."""
        M = 1.0
        b = sch.critical_impact_parameter(M)
        assert b == pytest.approx(3.0 * np.sqrt(3.0) * M)

    def test_critical_impact_parameter_scales_with_mass(self) -> None:
        """b_crit should scale linearly with M."""
        b1 = sch.critical_impact_parameter(1.0)
        b2 = sch.critical_impact_parameter(2.0)
        assert b2 == pytest.approx(2.0 * b1)


# ---------------------------------------------------------------------------
# Escape velocity
# ---------------------------------------------------------------------------

class TestEscapeVelocity:
    """Tests for escape velocity in Schwarzschild geometry."""

    def test_escape_velocity_at_infinity(self) -> None:
        """v_esc → 0 as r → ∞."""
        M = 1.0
        v = sch.escape_velocity(1e15, M)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_escape_velocity_at_rs(self) -> None:
        """v_esc → c (= 1 in geometrized) at r = r_s."""
        M = 1.0
        r_s = sch.schwarzschild_radius(M)
        v = sch.escape_velocity(r_s, M)
        assert v == pytest.approx(1.0)

    def test_escape_velocity_decreases_with_r(self) -> None:
        """Escape velocity is a decreasing function of r."""
        M = 1.0
        radii = [3.0, 10.0, 100.0, 1000.0]
        velocities = [float(sch.escape_velocity(r, M)) for r in radii]
        for i in range(len(velocities) - 1):
            assert velocities[i] > velocities[i + 1]


# ---------------------------------------------------------------------------
# Metric coefficients
# ---------------------------------------------------------------------------

class TestMetricCoefficients:
    """Tests for Schwarzschild metric components."""

    def test_flat_space_limit(self) -> None:
        """At r ≫ 2M the metric should approach Minkowski."""
        M = 1.0
        r = 1e10
        g_tt, g_rr, g_thth, g_phph = sch.metric_coefficients(r, M)
        assert float(g_tt) == pytest.approx(-1.0, abs=1e-6)
        assert float(g_rr) == pytest.approx(1.0, abs=1e-6)

    def test_horizon_divergence(self) -> None:
        """g_rr → ∞ as r → 2M (coordinate singularity)."""
        M = 1.0
        r_s = sch.schwarzschild_radius(M)
        # Slightly outside the horizon
        _, g_rr, _, _ = sch.metric_coefficients(r_s + 1e-6, M)
        assert float(g_rr) > 1e5


# ---------------------------------------------------------------------------
# Weak-field light deflection
# ---------------------------------------------------------------------------

class TestWeakFieldDeflection:
    """Tests for the weak-field light bending formula δφ = 4M/b."""

    def test_solar_deflection(self) -> None:
        """Solar limb deflection ≈ 1.75 arcsec.

        In SI:  δφ = 4 G M_sun / (c² R_sun)
        R_sun ≈ 6.957 × 10⁸ m.
        """
        R_sun_m = 6.957e8  # metres
        # Convert to geometrized
        M_geo = to_geometrized(M_SUN_KG, "mass")
        b_geo = R_sun_m  # length is identity in geometrized
        delta_phi = sch.light_deflection_weak_field(b_geo, M_geo)
        arcsec = np.degrees(delta_phi) * 3600.0
        assert arcsec == pytest.approx(1.75, rel=0.01)

    def test_deflection_decreases_with_b(self) -> None:
        """Deflection angle decreases as b grows."""
        M = 1.0
        assert sch.light_deflection_weak_field(10.0, M) > sch.light_deflection_weak_field(100.0, M)

    def test_deflection_zero_b_raises(self) -> None:
        """b ≤ 0 should raise ValueError."""
        with pytest.raises(ValueError):
            sch.light_deflection_weak_field(0.0, 1.0)
        with pytest.raises(ValueError):
            sch.light_deflection_weak_field(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Effective potential
# ---------------------------------------------------------------------------

class TestEffectivePotential:
    """Tests for the photon effective potential."""

    def test_veff_zero_at_origin_limit(self) -> None:
        """V_eff → 0 as r → ∞ for any L."""
        M = 1.0
        L = 10.0
        v = sch.effective_potential_photon(1e12, L, M)
        assert float(v) == pytest.approx(0.0, abs=1e-6)

    def test_veff_peak_at_photon_sphere(self) -> None:
        """V_eff has a maximum at r = 3M (photon sphere)."""
        M = 1.0
        L = 1.0
        r = np.linspace(2.5 * M, 20.0 * M, 10000)
        v = sch.effective_potential_photon(r, L, M)
        r_peak = r[np.argmax(v)]
        assert r_peak == pytest.approx(3.0 * M, rel=0.01)
