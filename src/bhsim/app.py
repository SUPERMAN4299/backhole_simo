"""Application entry point — ModernGL window with game-loop architecture.

Creates an OpenGL window, clears to a dark background, and runs an
update/render loop.  This is the 'Hello Window' milestone: proof that
the toolchain works before any physics or shaders are added.

Usage::

    python -m bhsim.app
    # or, after ``pip install -e .``:
    bhsim
"""

from __future__ import annotations

import array
import logging
import sys
from pathlib import Path
from typing import Any

import moderngl_window as mglw
from moderngl_window import WindowConfig
import yaml

from bhsim import __version__
from bhsim.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
# When installed as a package the config may be relative to the repo root.
# Fall back to a sibling search if the path doesn't resolve.
if not _DEFAULT_CONFIG.exists():
    _DEFAULT_CONFIG = Path.cwd() / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration.

    Args:
        path: Explicit path to a YAML file.  Defaults to
              ``config/default.yaml`` relative to the repo root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file cannot be located.
    """
    cfg_path = path or _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration not found: {cfg_path}")
    with open(cfg_path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s", cfg_path)
    return cfg


# ---------------------------------------------------------------------------
# Window / application
# ---------------------------------------------------------------------------

class BlackHoleApp(WindowConfig):
    """Main application window.

    Inherits from ``moderngl_window.WindowConfig`` and implements a
    clean update/render split.  Phase 0 simply clears the screen to
    a dark navy/charcoal colour and reports FPS in the title bar.
    """

    # -- WindowConfig class-level settings (overridden from config) --------
    gl_version = (4, 3)
    title = "bhsim — Black Hole Visualization Engine"
    resizable = True
    # moderngl-window requires this path to exist and be a directory.
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    resource_dir = _REPO_ROOT / "resources"

    # Background clear colour — dark navy/charcoal
    _CLEAR_COLOR = (0.02, 0.02, 0.04, 1.0)

    def __init__(self, **kwargs: Any) -> None:
        # Load config early so window attributes can be overridden.
        self._cfg = load_config()
        win_cfg = self._cfg.get("window", {})

        # Apply config-driven window parameters before WindowConfig init.
        kwargs.setdefault("title", win_cfg.get("title", self.title))
        kwargs.setdefault(
            "window_size",
            (win_cfg.get("width", 1280), win_cfg.get("height", 720)),
        )
        kwargs.setdefault("vsync", win_cfg.get("vsync", True))
        # moderngl-window uses fullscreen kwarg (bool).
        kwargs.setdefault("fullscreen", win_cfg.get("fullscreen", False))

        super().__init__(**kwargs)

        # FPS tracking ------------------------------------------------------
        self._frame_count: int = 0
        self._fps_timer: float = 0.0
        self._fps: float = 0.0

        # Minimal Phase 0 render (triangle) -------------------------------
        # This ensures the window is not just a black clear.
        self._tri_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(0.15, 0.85, 0.95, 1.0); // cyan-ish
                }
            """,
        )

        # Triangle in NDC (x,y) coords
        vertices = [
            -0.6, -0.5,
             0.6, -0.5,
             0.0,  0.7,
        ]
        self._tri_vbo = self.ctx.buffer(
            array.array("f", vertices).tobytes()
        )

        # vertex attribute format: 2 floats per vertex => '2f'
        self._tri_vao = self.ctx.vertex_array(
            self._tri_program,
            [(self._tri_vbo, "2f", "in_pos")],
        )

        self.ctx.disable(mglw.ALL_DEPTH) if hasattr(mglw, "ALL_DEPTH") else None

        # Log startup info --------------------------------------------------
        ctx = self.ctx
        logger.info("=" * 60)
        logger.info("  bhsim v%s — Black Hole Visualization Engine", __version__)
        logger.info("=" * 60)
        logger.info("OpenGL version : %s", ctx.info["GL_VERSION"])
        logger.info("Renderer       : %s", ctx.info["GL_RENDERER"])
        logger.info("Window size    : %d × %d", self.window_size[0], self.window_size[1])
        logger.info("VSync          : %s", win_cfg.get("vsync", True))
        logger.info("Fullscreen     : %s", win_cfg.get("fullscreen", False))
        logger.info("Max tex size   : %s", ctx.info.get("GL_MAX_TEXTURE_SIZE", "?"))
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def update(self, time_delta: float) -> None:
        """Logic update (physics, camera, UI state).

        Args:
            time_delta: Seconds elapsed since the last frame.
        """
        # FPS counter -------------------------------------------------------
        self._frame_count += 1
        self._fps_timer += time_delta
        if self._fps_timer >= 1.0:
            self._fps = self._frame_count / self._fps_timer
            self._frame_count = 0
            self._fps_timer = 0.0
            self.wnd.title = f"bhsim  |  {self._fps:.0f} FPS"

    def on_render(self, time: float, frame_time: float) -> None:
        """Render callback expected by moderngl-window."""
        self.update(frame_time)
        self.ctx.clear(*self._CLEAR_COLOR)

        # Draw Phase-0 placeholder triangle
        self._tri_vao.render(mode=self.ctx.TRIANGLES)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean shutdown hook."""
        logger.info("Shutting down bhsim.  Goodbye.")
        super().close()

    def key_event(self, key: Any, action: Any, modifiers: Any) -> None:
        """Handle key events.

        Pressing Escape or Q closes the window.
        """
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.ESCAPE or key == keys.Q:
                self.wnd.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the black hole visualizer."""
    setup_logging()
    logger.info("Starting bhsim v%s …", __version__)

    # Ensure resource_dir exists so moderngl-window doesn't crash in Phase 0.
    BlackHoleApp.resource_dir.mkdir(parents=True, exist_ok=True)

    try:
        mglw.run_window_config(BlackHoleApp)
    except Exception:
        logger.exception("Fatal error — bhsim is shutting down.")
        sys.exit(1)


if __name__ == "__main__":
    main()
