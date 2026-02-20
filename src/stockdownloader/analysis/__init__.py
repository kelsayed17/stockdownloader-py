"""Analysis tools for signals, patterns, and valuations."""

from stockdownloader.analysis.formula_calculator import FormulaCalculator, ValuationInputs
from stockdownloader.analysis.pattern_analyzer import analyze as analyze_patterns
from stockdownloader.analysis.pattern_analyzer import print_results as print_pattern_results
from stockdownloader.analysis.signal_generator import generate_alert
from stockdownloader.analysis.signal_advisor import SignalAdvisor, AdvisorConfig
from stockdownloader.analysis.alert_store import AlertStore
from stockdownloader.analysis.options_gamma_analyzer import OptionsGammaAnalyzer

__all__ = [
    "FormulaCalculator",
    "ValuationInputs",
    "analyze_patterns",
    "print_pattern_results",
    "generate_alert",
    "SignalAdvisor",
    "AdvisorConfig",
    "AlertStore",
    "OptionsGammaAnalyzer",
]
