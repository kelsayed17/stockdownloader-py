"""Shared CLI application helpers — re-exports from focused modules.

This module previously contained all helpers in a single file.  The
implementation now lives in three focused modules:

- :mod:`._cli_args` — argparse argument builders
- :mod:`._data_loaders` — CSV / Yahoo data loading utilities
- :mod:`._output` — output directory setup and display formatting

Everything is re-exported here so existing ``from stockdownloader.app.app_helpers import …``
statements continue to work unchanged.
"""

from stockdownloader.app._cli_args import *  # noqa: F401,F403
from stockdownloader.app._data_loaders import *  # noqa: F401,F403
from stockdownloader.app._output import *  # noqa: F401,F403

# Re-export convenience constants that were previously available here.
from stockdownloader.util.constants import (  # noqa: F401
    DEFAULT_DATA_FILE,
    DEFAULT_OUTPUT_DIR,
    INITIAL_CAPITAL,
    OPTIONS_COMMISSION,
    RISK_PER_TRADE,
)
