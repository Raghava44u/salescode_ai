"""
Structured logging utilities for the Screen Recapture Detection project.

Design Decisions:
    - Dual-handler setup: coloured console output for interactive use,
      plain-text file output for CI/production log ingestion.
    - ISO-8601 timestamps in file logs for machine parseability.
    - Module-name propagation via ``%(name)s`` so log origin is always traceable.
    - Factory function ``setup_logger`` returns a stdlib ``logging.Logger`` —
      no custom logger classes, keeping the API familiar.

Usage:
    from src.utils.logging_utils import setup_logger
    logger = setup_logger("my_module", log_dir=Path("logs"))
    logger.info("Pipeline started.")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI colour codes for console readability
_COLOURS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",  # bold red
    "RESET": "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    """Formatter that applies ANSI colour codes to log-level names.

    Used exclusively for console (stderr) output.  File handlers should
    use the plain ``logging.Formatter`` to keep logs machine-readable.
    """

    def __init__(self, fmt: str, datefmt: Optional[str] = None) -> None:
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Wrap the level-name in ANSI colour escapes."""
        colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
        record.levelname = f"{colour}{record.levelname:<8}{_COLOURS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a logger with console and optional file handlers.

    If the logger already has handlers (e.g., from a previous call), they
    are cleared to prevent duplicate output.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
        log_dir: Directory for log files.  Created automatically if it
            does not exist.  Ignored when ``log_to_file`` is ``False``.
        level: Minimum severity level (default: ``logging.INFO``).
        log_to_file: Whether to write logs to a file in ``log_dir``.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    # ── Console handler (stderr) ──────────────────────────────────────
    console_fmt = "%(levelname)s │ %(name)s │ %(message)s"
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColouredFormatter(console_fmt))
    logger.addHandler(console_handler)

    if log_to_file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = log_filename if log_filename else f"{name}.log"
        file_path = log_dir / filename
        file_fmt = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(file_fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(file_handler)

    return logger
