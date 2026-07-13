"""Structured logging configuration for bhsim.

Provides a single ``setup_logging()`` entry point that configures the root
logger and per-module log levels so that noisy subsystems (e.g. OpenGL
driver chatter) can be silenced independently.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Format string
# ---------------------------------------------------------------------------

_LOG_FMT = (
    "%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s"
)
_DATE_FMT = "%H:%M:%S"

# Default per-module overrides (module-name → level)
_MODULE_LEVELS: dict[str, int] = {
    "bhsim":               logging.DEBUG,
    "bhsim.app":           logging.DEBUG,
    "bhsim.physics":       logging.DEBUG,
    "bhsim.gl_core":       logging.INFO,
    "bhsim.pipeline":      logging.INFO,
    "moderngl":            logging.WARNING,
    "moderngl_window":     logging.WARNING,
    "PIL":                 logging.WARNING,
}


def setup_logging(
    level: int = logging.DEBUG,
    module_levels: Optional[dict[str, int]] = None,
) -> None:
    """Configure the application-wide logging.

    Call this once at startup — before any other ``getLogger`` calls — to
    install a uniform formatter and set per-module verbosity.

    Args:
        level: The root logger level (default ``DEBUG``).
        module_levels: Optional dict mapping logger names to levels.
            Merged on top of the built-in defaults so callers can
            override individual modules without touching the rest.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. in tests).
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        root.addHandler(handler)

    # Apply per-module levels.
    effective = {**_MODULE_LEVELS, **(module_levels or {})}
    for name, lvl in effective.items():
        logging.getLogger(name).setLevel(lvl)
