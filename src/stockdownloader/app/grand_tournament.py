"""Grand tournament — institutional-grade strategy evaluation.

Evaluates all strategy candidates (daily strategies, signal stacks,
ensemble) using walk-forward validation and regime-aware scoring.

Ranks strategies by out-of-sample score to surface genuinely robust
configurations while flagging likely overfitters via degradation ratio.

Usage::

    python -m stockdownloader.app.grand_tournament
    python -m stockdownloader.app.grand_tournament --mode quick

Output written to ``output/grand_tournament.log``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from decimal import Decimal
from operator import attrgetter
from pathlib import Path
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.regime_detector import MarketRegime, MarketRegimeDetector
from stockdownloader.strategy.regime_strategy_map import RegimeStrategyMapper
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.file_helper import TeeWriter

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "spy_5m_bars.csv"
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"
_LOG_FILE = _OUTPUT_DIR / "grand_tournament.log"

_INITIAL_CAPITAL = Decimal("100000")
_RISK_PER_TRADE = Decimal("0.01")


def _build_daily_candidates() -> list[tuple[str, callable]]:
    """Build factories for the 7 daily strategies with optimized params.

    Uses the best parameters from the optimizer tournament
    (docs/optimizer_tournament_results.md).
    """
    from stockdownloader.strategy.registrations import ensure_registered
    from stockdownloader.strategy.registry import StrategyRegistry
    ensure_registered()

    # Optimized parameters from the tournament
    optimized = {
        "sma": {"short_period": 20, "long_period": 21},
        "rsi": {"period": 7, "oversold": 35.0, "overbought": 65.0},
        "macd": {"fast_period": 8, "slow_period": 35, "signal_period": 5},
        "bollinger": {
            "rsi_period": 10, "rsi_oversold": 25, "rsi_overbought": 65,
            "bb_std_dev": 1.5, "adx_threshold": 30,
        },
        "breakout": {
            "bb_period": 25, "squeeze_lookback": 120,
            "volume_multiplier": 1.2,
        },
        "momentum": {
            "fast_ema": 8, "slow_ema": 21, "signal_period": 9,
            "ema_trend_filter": 150, "adx_strength_threshold": 20,
            "adx_weak_threshold": 15,
        },
        "multi": {"buy_threshold": 3, "sell_threshold": 5},
    }

    adapter_params = {
        "sl_atr_mult": Decimal("1.5"),
        "rr": Decimal("1.5"),
        "sl_cap": Decimal("2.00"),
    }

    candidates: list[tuple[str, callable]] = []

    for name, kwargs in optimized.items():
        entry = StrategyRegistry.get(name)
        if entry is None:
            continue

        # Create factory that produces a fresh adapter each time
        def _factory(entry=entry, kwargs=kwargs, name=name):
            strategy = entry.factory(**kwargs)
            return DailyToIntradayAdapter(
                strategy=strategy,
                allow_shorts=name not in ("sma", "breakout"),
                **adapter_params,
            )

        candidates.append((f"Daily:{entry.display_name}", _factory))

    return candidates


def _build_signal_stack_candidates() -> list[tuple[str, callable]]:
    """Build factories for the top signal stack combinations.

    Uses the best combos from the signal stack tournament.
    """
    from stockdownloader.strategy.signals.multi_timeframe_aligner import (
        TimeframeSignalSpec,
    )
    from stockdownloader.strategy.signals.signal_registry import (
        SignalGeneratorRegistry,
    )
    from stockdownloader.strategy.signals.stacked_signal_engine import (
        AggregationMode,
        StackConfig,
    )
    from stockdownloader.strategy.signals.stacked_intraday_strategy import (
        StackedIntradayStrategy,
    )
    from stockdownloader.util.timeframe_aggregator import Timeframe

    # Import generators to trigger registration
    import stockdownloader.strategy.signals.generators  # noqa: F401

    entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}

    # Top combos from the signal stack tournament
    top_combos = [
        (("macd", "obv"), 0.2, 0.3, AggregationMode.WEIGHTED_AVERAGE),
        (("macd", "obv", "stochastic"), 0.2, 0.3, AggregationMode.WEIGHTED_AVERAGE),
        (("macd", "obv", "volume_surge"), 0.2, 0.3, AggregationMode.WEIGHTED_AVERAGE),
        (("obv", "rsi", "sma_cross"), 0.2, 0.3, AggregationMode.WEIGHTED_AVERAGE),
        (("macd", "obv", "ema_trend"), 0.2, 0.3, AggregationMode.WEIGHTED_AVERAGE),
    ]

    candidates: list[tuple[str, callable]] = []

    for combo, buy_th, sell_th, mode in top_combos:
        combo_name = "+".join(combo)

        def _factory(
            combo=combo, buy_th=buy_th, sell_th=sell_th, mode=mode,
            combo_name=combo_name,
        ):
            specs = []
            for name in combo:
                entry = entries[name]
                gen = entry.factory(**entry.default_kwargs)
                specs.append(TimeframeSignalSpec(
                    generator=gen,
                    timeframe=Timeframe.M5,
                    weight=1.0,
                ))

            config = StackConfig(
                buy_threshold=buy_th,
                sell_threshold=sell_th,
                mode=mode,
                require_fire=True,
            )

            return StackedIntradayStrategy(
                name=f"Stack:{combo_name}",
                config=config,
                specs=specs,
                allow_shorts=True,
            )

        candidates.append((f"Stack:{combo_name}", _factory))

    return candidates


def _run_tournament(
    data: list[IntradayPriceData],
    candidates: list[tuple[str, callable]],
    n_windows: int,
    tee: TeeWriter | None,
) -> list[WalkForwardResult]:
    """Run walk-forward validation for all candidates."""

    def _print(msg: str) -> None:
        if tee:
            tee.write(msg + "\n")
        else:
            print(msg)

    engine = IntradayBacktestEngine(
        initial_capital=_INITIAL_CAPITAL,
        risk_per_trade=_RISK_PER_TRADE,
    )

    validator = WalkForwardValidator(
        data=data,
        n_windows=n_windows,
        is_ratio=0.7,
    )

    _print(f"\nWalk-forward windows: {len(validator.windows)}")
    for w in validator.windows:
        _print(
            f"  Window {w.window_id}: IS [{w.in_sample_start}:{w.in_sample_end}] "
            f"OOS [{w.out_of_sample_start}:{w.out_of_sample_end}]"
        )

    results: list[WalkForwardResult] = []
    start_time = time.time()

    for i, (name, factory) in enumerate(candidates):
        _print(f"\n[{i + 1}/{len(candidates)}] Evaluating {name}...")
        t0 = time.time()

        try:
            wf_result = validator.validate(
                strategy_factory=factory,
                engine=engine,
                strategy_name=name,
            )
            results.append(wf_result)
            elapsed = time.time() - t0
            _print(
                f"  IS={wf_result.in_sample_score:+.2f}  "
                f"OOS={wf_result.out_of_sample_score:+.2f}  "
                f"degradation={wf_result.degradation_ratio:.2f}  "
                f"({elapsed:.1f}s)"
            )
        except Exception as e:
            _print(f"  FAILED: {e}")
            logger.warning("Failed to evaluate %s: %s", name, e)

    total_elapsed = time.time() - start_time
    _print(f"\nTotal evaluation time: {total_elapsed:.1f}s")

    return results


def _print_rankings(
    results: list[WalkForwardResult],
    tee: TeeWriter | None,
) -> None:
    """Print the grand ranking tables."""

    def _print(msg: str) -> None:
        if tee:
            tee.write(msg + "\n")
        else:
            print(msg)

    # Sort by OOS score (descending)
    ranked = sorted(results, key=attrgetter("out_of_sample_score"), reverse=True)

    _print("\n" + "=" * 100)
    _print("GRAND TOURNAMENT — RANKED BY OUT-OF-SAMPLE SCORE")
    _print("=" * 100)
    _print(
        f"{'Rank':>4}  {'OOS Score':>10}  {'IS Score':>10}  "
        f"{'Degrade':>8}  {'Flag':>6}  Strategy"
    )
    _print("-" * 100)

    for i, r in enumerate(ranked, 1):
        flag = ""
        if r.degradation_ratio < 0.5 and r.in_sample_score > 0:
            flag = "⚠ OVER"
        elif r.degradation_ratio > 1.5:
            flag = "✓ GOOD"
        elif r.out_of_sample_score > 0:
            flag = "  OK"
        else:
            flag = "✗ NEG"

        _print(
            f"{i:>4}  {r.out_of_sample_score:>+10.2f}  "
            f"{r.in_sample_score:>+10.2f}  "
            f"{r.degradation_ratio:>8.2f}  "
            f"{flag:>6}  {r.strategy_name}"
        )

    _print("=" * 100)

    # IS vs OOS comparison
    _print("\n" + "=" * 80)
    _print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    _print("=" * 80)
    _print(
        f"{'Strategy':>35}  {'IS Score':>10}  {'OOS Score':>10}  {'Ratio':>8}"
    )
    _print("-" * 80)

    for r in ranked:
        _print(
            f"{r.strategy_name:>35}  {r.in_sample_score:>+10.2f}  "
            f"{r.out_of_sample_score:>+10.2f}  {r.degradation_ratio:>8.2f}"
        )

    _print("=" * 80)

    # Summary
    _print("\n" + "=" * 60)
    _print("SUMMARY")
    _print("=" * 60)
    if ranked:
        best = ranked[0]
        _print(f"  Best OOS strategy: {best.strategy_name}")
        _print(f"  OOS Score: {best.out_of_sample_score:+.2f}")
        _print(f"  IS Score:  {best.in_sample_score:+.2f}")
        _print(f"  Degradation: {best.degradation_ratio:.2f}")

        # Count overfitters
        overfit = sum(
            1 for r in ranked
            if r.degradation_ratio < 0.5 and r.in_sample_score > 0
        )
        _print(f"  Likely overfit strategies: {overfit}/{len(ranked)}")
    _print("=" * 60)


def main(mode: str = "quick", log_path: Path | None = None) -> None:
    """Run the grand tournament.

    Parameters
    ----------
    mode:
        ``"quick"`` — 3 windows, daily strategies only.
        ``"standard"`` — 5 windows, daily + top signal stacks.
        ``"full"`` — 5 windows, daily + stacks + ensemble.
    log_path:
        Optional custom log file path.
    """
    actual_log = log_path or _LOG_FILE
    actual_log.parent.mkdir(parents=True, exist_ok=True)

    with open(actual_log, "w", encoding="utf-8") as log_fh:
        tee = TeeWriter(log_fh)

        tee.write("=" * 80 + "\n")
        tee.write("GRAND TOURNAMENT — INSTITUTIONAL STRATEGY EVALUATION\n")
        tee.write(f"Mode: {mode}\n")
        tee.write("=" * 80 + "\n")

        # Load data
        tee.write(f"\nLoading data from {_DATA_FILE}...\n")
        if not _DATA_FILE.exists():
            tee.write(f"ERROR: Data file not found: {_DATA_FILE}\n")
            return

        data = IntradayCsvLoader.load_from_file(_DATA_FILE)
        tee.write(f"Loaded {len(data)} bars\n")

        trading_days = len({d.date[:10] for d in data})
        tee.write(f"Trading days: {trading_days}\n")

        # Build candidates
        tee.write("\nBuilding strategy candidates...\n")
        candidates: list[tuple[str, callable]] = []

        # Always include daily strategies
        daily = _build_daily_candidates()
        candidates.extend(daily)
        tee.write(f"  Daily strategies: {len(daily)}\n")

        if mode in ("standard", "full"):
            stacks = _build_signal_stack_candidates()
            candidates.extend(stacks)
            tee.write(f"  Signal stacks: {len(stacks)}\n")

        tee.write(f"  Total candidates: {len(candidates)}\n")

        # Configure walk-forward windows
        n_windows = 3 if mode == "quick" else 5

        # Run the tournament
        results = _run_tournament(data, candidates, n_windows, tee)

        # Print rankings
        _print_rankings(results, tee)

        tee.write(f"\nLog saved to: {actual_log}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grand tournament — institutional strategy evaluation",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Tournament scope (default: quick)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Custom log file path",
    )
    args = parser.parse_args()
    main(mode=args.mode, log_path=args.log_file)
