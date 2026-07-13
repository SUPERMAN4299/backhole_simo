"""Tests for YAML configuration loading and validation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture()
def config() -> dict:
    """Load the default configuration from disk."""
    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigStructure:
    """Verify the top-level sections exist and have sane defaults."""

    def test_top_level_keys(self, config: dict) -> None:
        expected = {"window", "black_hole", "camera", "rendering"}
        assert expected.issubset(config.keys()), (
            f"Missing config sections: {expected - config.keys()}"
        )

    def test_window_dimensions(self, config: dict) -> None:
        w = config["window"]
        assert isinstance(w["width"], int) and w["width"] > 0
        assert isinstance(w["height"], int) and w["height"] > 0

    def test_window_title(self, config: dict) -> None:
        assert isinstance(config["window"]["title"], str)
        assert len(config["window"]["title"]) > 0

    def test_black_hole_mass_positive(self, config: dict) -> None:
        mass = config["black_hole"]["mass"]
        assert mass > 0, "Black hole mass must be positive"

    def test_black_hole_spin_range(self, config: dict) -> None:
        spin = config["black_hole"]["spin"]
        assert 0.0 <= spin < 1.0, "Spin parameter must be in [0, 1)"

    def test_camera_position_is_3d(self, config: dict) -> None:
        pos = config["camera"]["position"]
        assert isinstance(pos, list) and len(pos) == 3

    def test_camera_fov_sane(self, config: dict) -> None:
        fov = config["camera"]["fov"]
        assert 10.0 <= fov <= 170.0, "Field of view out of sane range"

    def test_rendering_max_steps(self, config: dict) -> None:
        steps = config["rendering"]["max_steps"]
        assert isinstance(steps, int) and steps > 0

    def test_rendering_tonemap_method(self, config: dict) -> None:
        method = config["rendering"]["tonemap_method"]
        assert method in {"aces", "reinhard", "uncharted2"}


class TestConstants:
    """Smoke-test the constants module (no OpenGL needed)."""

    def test_speed_of_light(self) -> None:
        from bhsim.constants import c
        assert abs(c - 2.998e8) < 1e6

    def test_schwarzschild_radius_sun(self) -> None:
        from bhsim.constants import schwarzschild_radius, SOLAR_MASS
        r_s = schwarzschild_radius(SOLAR_MASS)
        # Should be ≈ 2953 m
        assert 2900 < r_s < 3000

    def test_isco_schwarzschild(self) -> None:
        from bhsim.constants import isco_radius, schwarzschild_radius, SOLAR_MASS
        r_isco = isco_radius(SOLAR_MASS, spin=0.0)
        r_s = schwarzschild_radius(SOLAR_MASS)
        # ISCO = 3 r_s for Schwarzschild
        assert math.isclose(r_isco, 3.0 * r_s, rel_tol=1e-9)

    def test_mass_to_length(self) -> None:
        from bhsim.constants import mass_to_length, SOLAR_MASS
        length = mass_to_length(SOLAR_MASS)
        # GM/c² ≈ 1476.6 m
        assert 1470 < length < 1485
