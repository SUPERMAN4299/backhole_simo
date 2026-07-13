# bhsim — Black Hole Visualization Engine

A real-time, physically-accurate black hole visualization engine built with Python, ModernGL, and GLSL compute shaders.

> **Status:** Phase 0 — Project skeleton & toolchain verification.

---

## Features (Planned)

| Phase | Milestone | Description |
|-------|-----------|-------------|
| **0** | Hello Window | Project scaffold, config system, dev toolchain |
| **1** | Schwarzschild Geodesics | Null-geodesic integrator in pure NumPy (CPU) |
| **2** | GPU Ray Marcher | Port integrator to GLSL compute shaders |
| **3** | Accretion Disk | Novikov–Thorne thin-disk model with thermal spectrum |
| **4** | Kerr Metric | Frame-dragging, ergosphere, ISCO shift |
| **5** | Post-Processing | Bloom, gravitational redshift color grading, tone mapping |
| **6** | imgui HUD | Interactive parameter tuning, orbit camera |

## Architecture

```
src/bhsim/
├── physics/        # Pure NumPy/SciPy — metrics, geodesics, disk model
├── gl_core/        # ModernGL wrappers — shaders, textures, framebuffers
├── pipeline/       # Glue: feeds physics params → GPU uniforms
├── ui/             # imgui_bundle overlay
├── app.py          # Application entry point
├── constants.py    # Physical constants & unit helpers
└── logging_config.py
```

**Key rule:** `physics/` never imports from `gl_core/`. They meet only in `pipeline/`.

## Requirements

- Python ≥ 3.11
- OpenGL ≥ 4.3 (for compute shaders)
- System packages: `libgl1-mesa-dev`, `libegl1-mesa-dev` (Linux)

## Installation

```bash
# Clone the repository
git clone <repo-url> blackholesim
cd blackholesim

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e '.[dev]'
```

## Usage

```bash
# Run the visualizer
bhsim

# Or via module
python -m bhsim.app

# Run tests
pytest tests/ -v
```

## Configuration

Default settings live in `config/default.yaml`. Override by passing a custom YAML:

```bash
bhsim --config my_config.yaml   # (not yet implemented)
```

## License

MIT
