"""Unified strategy pipeline — backtest, optimize, re-backtest, validate."""

from stockdownloader.app.pipeline.cli import main, _OPTIMIZE_MODES
from stockdownloader.app.pipeline.models import SlotResult

__all__ = ["main", "SlotResult", "_OPTIMIZE_MODES"]
