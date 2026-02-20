"""Output directory setup and display formatting helpers."""

from __future__ import annotations

from pathlib import Path

from stockdownloader.util.constants import DEFAULT_OUTPUT_DIR


def setup_output_dir(log_name: str) -> Path:
    """Ensure :data:`DEFAULT_OUTPUT_DIR` exists and return the log path.

    Parameters
    ----------
    log_name:
        Filename for the log file (e.g. ``"tournament_results.log"``).

    Returns
    -------
    Path
        Full path to ``DEFAULT_OUTPUT_DIR / log_name``.
    """
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / log_name


def print_banner(title: str, width: int = 40) -> None:
    """Print a centered banner header."""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()
