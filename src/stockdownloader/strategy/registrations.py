"""Centralized strategy registration for StrategyRegistry.

Call :func:`ensure_registered` from any entry point that needs the registry
populated (CLI apps, tests that query by name).  Idempotent — safe to call
multiple times.

Previously, registrations lived as module-level side effects in
``daily/__init__.py``, ``options/__init__.py``, and individual strategy
modules.  That design forced **every** strategy to load whenever **any**
strategy submodule was imported, creating a circular import chain through
``strategy/__init__.py``.

Moving registrations here decouples "import a strategy class" from
"populate the global registry".
"""
from __future__ import annotations

_registered = False


def ensure_registered() -> None:
    """Populate :class:`StrategyRegistry` with all built-in strategies.

    Imports are inside this function body so that merely importing
    ``registrations`` doesn't trigger a heavy module-loading cascade.
    """
    global _registered
    if _registered:
        return
    _registered = True

    from stockdownloader.strategy.registry import StrategyRegistry

    # ------------------------------------------------------------------
    # Daily strategies (7)
    # ------------------------------------------------------------------
    from stockdownloader.strategy.daily.sma_crossover_strategy import (
        SMACrossoverStrategy,
    )
    from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
    from stockdownloader.strategy.daily.macd_strategy import MACDStrategy
    from stockdownloader.strategy.daily.bollinger_band_rsi_strategy import (
        BollingerBandRSIStrategy,
    )
    from stockdownloader.strategy.daily.breakout_strategy import BreakoutStrategy
    from stockdownloader.strategy.daily.momentum_confluence_strategy import (
        MomentumConfluenceStrategy,
    )
    from stockdownloader.strategy.daily.multi_indicator_strategy import (
        MultiIndicatorStrategy,
    )

    StrategyRegistry.register(
        name="sma",
        display_name="SMA Crossover",
        category="daily",
        factory=SMACrossoverStrategy,
        default_kwargs={"short_period": 9, "long_period": 21},
        param_space={
            "short_period": [5, 9, 12, 15, 20],
            "long_period": [21, 30, 50, 100, 200],
        },
    )

    StrategyRegistry.register(
        name="rsi",
        display_name="RSI Strategy",
        category="daily",
        factory=RSIStrategy,
        default_kwargs={"period": 14, "oversold": 30.0, "overbought": 70.0},
        param_space={
            "period": [7, 10, 14, 21],
            "oversold": [20.0, 25.0, 30.0, 35.0],
            "overbought": [65.0, 70.0, 75.0, 80.0],
        },
    )

    StrategyRegistry.register(
        name="macd",
        display_name="MACD Strategy",
        category="daily",
        factory=MACDStrategy,
        default_kwargs={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        param_space={
            "fast_period": [8, 10, 12, 15],
            "slow_period": [20, 26, 30, 35],
            "signal_period": [5, 7, 9, 12],
        },
    )

    StrategyRegistry.register(
        name="bollinger",
        display_name="Bollinger Band + RSI",
        category="daily",
        factory=BollingerBandRSIStrategy,
        default_kwargs={},
        param_space={
            "bb_period": [15, 20, 25],
            "bb_std_dev": [1.5, 2.0, 2.5],
            "rsi_period": [10, 14, 21],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "adx_threshold": [20, 25, 30],
        },
    )

    StrategyRegistry.register(
        name="breakout",
        display_name="Breakout Strategy",
        category="daily",
        factory=BreakoutStrategy,
        default_kwargs={},
        param_space={
            "bb_period": [15, 20, 25],
            "squeeze_lookback": [60, 90, 120, 150],
            "volume_multiplier": [1.2, 1.5, 2.0],
        },
    )

    StrategyRegistry.register(
        name="momentum",
        display_name="Momentum Confluence",
        category="daily",
        factory=MomentumConfluenceStrategy,
        default_kwargs={},
        param_space={
            "fast_ema": [8, 12, 15],
            "slow_ema": [21, 26, 30],
            "signal_period": [7, 9, 12],
            "ema_trend_filter": [100, 150, 200],
            "adx_strength_threshold": [20, 25, 30],
            "adx_weak_threshold": [15, 20, 25],
        },
    )

    StrategyRegistry.register(
        name="multi",
        display_name="Multi-Indicator Confluence",
        category="daily",
        factory=MultiIndicatorStrategy,
        default_kwargs={},
        param_space={
            "buy_threshold": [3, 4, 5],
            "sell_threshold": [3, 4, 5],
        },
    )

    # ------------------------------------------------------------------
    # Options strategies (2)
    # ------------------------------------------------------------------
    from stockdownloader.strategy.options.covered_call_strategy import (
        CoveredCallStrategy,
    )
    from stockdownloader.strategy.options.protective_put_strategy import (
        ProtectivePutStrategy,
    )

    StrategyRegistry.register(
        name="covered-call",
        display_name="Covered Call",
        category="options",
        factory=CoveredCallStrategy,
        default_kwargs={"ma_period": 20},
        param_space={
            "ma_period": [10, 20, 50],
            "otm_percent": [0.03, 0.05, 0.07],
            "days_to_expiry": [15, 30, 45],
        },
    )

    StrategyRegistry.register(
        name="protective-put",
        display_name="Protective Put",
        category="options",
        factory=ProtectivePutStrategy,
        default_kwargs={"ma_period": 20},
        param_space={
            "ma_period": [10, 20, 50],
            "otm_percent": [0.03, 0.05, 0.07],
            "days_to_expiry": [30, 45, 60],
        },
    )

    # ------------------------------------------------------------------
    # Intraday strategies (5 standalone)
    # ------------------------------------------------------------------
    from decimal import Decimal as D

    from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
    from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy
    from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
    from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
    from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy

    StrategyRegistry.register(
        name="vwap-pullback",
        display_name="VWAP Pullback",
        category="intraday",
        factory=PullbackStrategy,
        default_kwargs={},
        param_space={
            "rr": [D("1.0"), D("1.2"), D("1.4"), D("1.6"), D("1.8"), D("2.0")],
            "sl_atr": [D("0.8"), D("1.0"), D("1.3"), D("1.5"), D("1.8")],
            "sl_cap": [D("1.00"), D("1.50"), D("2.00"), D("2.50")],
            "be_trigger": [D("0.3"), D("0.5"), D("0.7"), D("1.0")],
            "min_score": [2, 3, 4, 5],
            "min_score_long": [3, 4, 5],
            "adx_thresh": [D("18"), D("21"), D("25")],
            "max_day": [1, 2, 3],
            "spacing": [2, 3, 5],
            "pb_zone": [D("0.3"), D("0.5"), D("0.7"), D("1.0")],
            "pb_body": [D("0.10"), D("0.15"), D("0.20")],
            "trail_buf": [D("0.10"), D("0.15"), D("0.20"), D("0.30")],
        },
    )
    StrategyRegistry.register(
        name="vwap-reversal",
        display_name="VWAP Reversal",
        category="intraday",
        factory=ReversalStrategy,
        default_kwargs={},
        param_space={
            "rev_body": [D("0.15"), D("0.20"), D("0.25")],
            "rev_sl_atr": [D("0.8"), D("1.0"), D("1.3")],
            "rev_min_rr": [D("0.2"), D("0.3"), D("0.5")],
            "rev_hug_limit": [10, 15, 20, 30],
            "rev_shorts": [True, False],
            "adx_thresh": [D("18"), D("21"), D("25")],
            "be_trigger": [D("0.3"), D("0.5"), D("0.7")],
        },
    )
    StrategyRegistry.register(
        name="vwap-orb",
        display_name="OR Breakout",
        category="intraday",
        factory=ORBreakoutStrategy,
        default_kwargs={},
        param_space={
            "orb_window": [30, 45, 60, 78],
            "orb_rvol": [D("0.6"), D("0.8"), D("1.0"), D("1.5")],
            "orb_body_min": [D("0.05"), D("0.10"), D("0.15"), D("0.20")],
            "orb_trail_atr": [D("1.0"), D("1.5"), D("2.0")],
            "orb_sl_mode": ["OR Opposite", "OR Midpoint", "ATR-Based"],
            "orb_entry_mode": ["aggressive", "retest"],
            "orb_tp_mode": ["trail_only", "or_range", "2x_or_range"],
            "orb_gap_filter": [True, False],
            "orb_adx_filter": [True, False],
            "be_trigger": [D("0.3"), D("0.5"), D("0.7")],
        },
    )
    StrategyRegistry.register(
        name="vwap-orr",
        display_name="OR Reversal",
        category="intraday",
        factory=ORReversalStrategy,
        default_kwargs={},
        param_space={
            "orr_window": [30, 45, 60, 78],
            "orr_prox": [D("0.4"), D("0.6"), D("0.8"), D("1.0")],
            "orr_sl_atr": [D("0.3"), D("0.5"), D("0.8")],
            "orr_tp_mode": ["VWAP", "OR Mid", "OR Opposite"],
            "orr_min_rr": [D("0.2"), D("0.3"), D("0.5")],
            "orr_rvol": [D("0.6"), D("0.8"), D("1.0")],
            "orr_vwap_disagree": [True, False],
            "orr_gap_filter": [True, False],
            "orr_adx_filter": [True, False],
            "orr_require_break": [True, False],
            "be_trigger": [D("0.3"), D("0.5"), D("0.7")],
        },
    )
    StrategyRegistry.register(
        name="vwap-ps",
        display_name="Pattern Scalp",
        category="intraday",
        factory=PatternScalpStrategy,
        default_kwargs={},
        param_space={
            "ps_window": [20, 35, 50, 65],
            "ps_tp_pct": [D("50.0"), D("75.0"), D("100.0")],
            "ps_rvol": [D("0.6"), D("0.8"), D("1.0"), D("1.5")],
            "ps_engulf": [D("0.10"), D("0.15"), D("0.20"), D("0.30")],
            "ps_sl_mode": ["Day Extreme", "ATR-Based"],
            "ps_sma_filter": [True, False],
            "be_trigger": [D("0.3"), D("0.5"), D("0.7")],
        },
    )
