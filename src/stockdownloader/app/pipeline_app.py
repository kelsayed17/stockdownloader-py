"""Backward-compatibility re-exports for the pipeline subpackage.

The pipeline has moved to :mod:`stockdownloader.app.pipeline`.
This module re-exports the public API so that existing console
scripts and imports continue to work.
"""
from stockdownloader.app.pipeline.cli import main, _OPTIMIZE_MODES  # noqa: F401
from stockdownloader.app.pipeline.models import SlotResult  # noqa: F401
