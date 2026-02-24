"""Analysis tools for signals, patterns, and valuations.

Alerts & signals
-----------------
- :func:`generate_alert` — confluence-based buy/sell alert with options recs
- :class:`SignalAdvisor` — portfolio-level signal advisor
- :class:`AlertStore` — persistent alert storage

Valuation & screening
---------------------
- :class:`FormulaCalculator` — DCF / Graham / DDM valuation
- :class:`ValueScreener` — multi-factor value screening

Options analysis
----------------
- :class:`OptionsGammaAnalyzer` — gamma exposure & max-pain analysis
- :class:`OptionsFlowReport` — result container for gamma analysis

Pattern encoding
----------------
- :class:`BarEncoder` — N-gram bar encoding for pattern discovery
- :class:`PatternEncoder` — alias: same as BarEncoder

Subpackages
-----------
pattern_discovery : N-gram pattern mining, filtering, catalog
gme : GME-specific event study, regime, distribution analysis
"""

from stockdownloader.analysis.formula import FormulaCalculator, ValuationInputs
from stockdownloader.analysis.pattern_analyzer import analyze as analyze_patterns
from stockdownloader.analysis.pattern_analyzer import print_results as print_pattern_results
from stockdownloader.analysis.alert_generator import generate_alert
from stockdownloader.analysis.signal_advisor import SignalAdvisor, AdvisorConfig
from stockdownloader.analysis.alert_store import AlertStore
from stockdownloader.analysis.options.gamma_analyzer import (
    OptionsFlowReport,
    OptionsGammaAnalyzer,
)
from stockdownloader.analysis.pattern_encoder import BarEncoder
from stockdownloader.analysis.value.screener import ValueScreener

__all__ = [
    # Alerts & signals
    "generate_alert",
    "SignalAdvisor",
    "AdvisorConfig",
    "AlertStore",
    # Valuation & screening
    "FormulaCalculator",
    "ValuationInputs",
    "ValueScreener",
    # Pattern analysis
    "analyze_patterns",
    "print_pattern_results",
    "BarEncoder",
    # Options analysis
    "OptionsFlowReport",
    "OptionsGammaAnalyzer",
]
