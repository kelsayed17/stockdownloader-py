"""GME analysis -- re-exports from gme/ package for backward compatibility.

All analysis functions and dataclasses have been moved to the
``stockdownloader.analysis.gme`` sub-package.  This module re-exports
them so that existing imports like::

    from stockdownloader.analysis.gme_analyzer import run_event_study

continue to work without changes.
"""

from stockdownloader.analysis.gme import *  # noqa: F401,F403
