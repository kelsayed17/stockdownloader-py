"""Project-wide constants.

Single source of truth for default values used across the codebase.
These can be overridden at runtime via CLI arguments or config files,
but these values serve as sensible defaults.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

# ======================================================================
# Capital & risk
# ======================================================================

INITIAL_CAPITAL = Decimal("100000.00")
RISK_PER_TRADE = Decimal("0.01")  # 1 % of capital per trade
OPTIONS_COMMISSION = Decimal("0.65")  # Per-contract options commission

# ======================================================================
# Project paths (relative to project root)
# ======================================================================

#: Project root (3 levels up from this file: util → stockdownloader → src → root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

#: Default path to the SPY 5-minute bar CSV.
DEFAULT_DATA_FILE: Path = PROJECT_ROOT / "data" / "spy" / "5m_bars.csv"

#: Default output directory for logs and reports.
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# ======================================================================
# Walk-forward defaults
# ======================================================================

WF_WINDOWS = 5
WF_IS_RATIO = 0.7  # 70 % in-sample, 30 % out-of-sample
