"""Price action pattern discovery — mine, filter, validate, trade.

Scans historical bar data across multiple timeframes for recurring
multi-bar price patterns, measures their statistical edge, applies
rigorous filtering (t-test, Benjamini-Hochberg FDR, walk-forward
stability, indicator confirmation), and produces a validated pattern
catalog that can be traded as a strategy.

Usage::

    python3 -m stockdownloader.app.pattern_discovery
    python3 -m stockdownloader.app.pattern_discovery --mode discover
    python3 -m stockdownloader.app.pattern_discovery --timeframes 5m,15m,1h
    python3 -m stockdownloader.app.pattern_discovery --min-occurrences 30
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from stockdownloader.analysis.pattern_discovery import (
    DiscoveredPattern,
    FilterConfig,
    PatternCatalog,
    PatternMiner,
    PatternStats,
    apply_fdr_correction,
    deduplicate_patterns,
    filter_patterns,
)
from stockdownloader.analysis.pattern_encoder import BarEncoder
from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.optimization.scoring import score_v2
from stockdownloader.backtesting.tournament.engine import (
    classify_timeframe_bars,
    run_monte_carlo,
)
from stockdownloader.core.config import INITIAL_CAPITAL, RISK_PER_TRADE
from stockdownloader.backtesting.optimization.walk_forward import WalkForwardValidator
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.core.io import TeeWriter
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader.core.timeframe import Timeframe, TimeframeAggregator

from stockdownloader.app.helpers import (
    DEFAULT_DATA_FILE as _DATA_FILE,
    DEFAULT_OUTPUT_DIR as _OUTPUT_DIR,
    box_title as _box_title,
)


_TF_MAP: dict[str, Timeframe] = {
    "5m": Timeframe.M5,
    "15m": Timeframe.M15,
    "30m": Timeframe.M30,
    "1h": Timeframe.H1,
    "4h": Timeframe.H4,
    "1d": Timeframe.DAILY,
}

# Default temporal gap (bars) between discovery and validation per TF.
# Prevents indicator leakage from ATR/EMA lookback bleeding across boundary.
_DEFAULT_GAP: dict[str, int] = {
    "5m": 100,    # ~8 hours
    "15m": 34,    # ~8.5 hours
    "30m": 34,    # ~17 hours
    "1h": 10,     # ~10 hours
    "4h": 5,      # ~20 hours
    "1d": 5,      # 5 days
}


# ======================================================================
# CLI
# ======================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price Action Pattern Discovery Engine",
    )
    parser.add_argument(
        "--mode",
        choices=["discover", "backtest", "full"],
        default="full",
        help="discover: mine only. backtest: load catalog + backtest. full: all (default).",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="5m",
        help="Comma-separated timeframes to scan: 5m,15m,30m,1h,4h,1d (default: 5m)",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=20,
        help="Minimum pattern occurrences for 5m (default: 20, auto-scaled for higher TFs)",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=50,
        help="Maximum patterns to keep per timeframe (default: 50)",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=_DATA_FILE,
        help="Path to 5m CSV data file",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=None,
        help="Path to load/save pattern catalog JSON",
    )
    parser.add_argument(
        "--discovery-split",
        type=float,
        default=0.6,
        help="Fraction of data for discovery (default: 0.6 = 60%%)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Mine only for this regime (e.g. mean_reverting, strong_trend_up)",
    )
    parser.add_argument(
        "--gap-bars",
        type=int,
        default=None,
        help="Temporal gap (bars) between discovery and validation. "
             "Default: auto per TF (100 for 5m, 34 for 15m, etc.)",
    )
    return parser.parse_args(argv)


# ======================================================================
# Pipeline stages
# ======================================================================


def _discover_patterns(
    data,
    split_idx: int,
    config: FilterConfig,
    out,
    *,
    timeframe: str = "5m",
    regime_map=None,
) -> tuple[PatternCatalog, dict]:
    """Mine and filter patterns on discovery data.

    Returns
    -------
    tuple of (PatternCatalog, stats_dict)
    """
    discovery_data = data[:split_idx]

    out()
    out(_box_title(f"STAGE 1: PATTERN ENCODING ({timeframe.upper()})"))
    out()
    out(f"  Encoding {len(discovery_data):,} bars into categorical features...")

    hub = IndicatorHub()
    encoder = BarEncoder(hub)
    miner = PatternMiner(encoder)

    # Warmup: need enough bars for ATR(14), EMA(21), etc.
    warmup = max(50, 21)  # at least 50 bars for stable indicators

    t0 = time.time()
    raw_patterns = miner.mine(
        discovery_data, start=warmup, end=len(discovery_data),
        regime_map=regime_map,
    )
    encode_time = time.time() - t0

    total_windows = sum(
        len(discovery_data) - warmup - n + 1
        for n in miner._pattern_lengths
        if len(discovery_data) - warmup - n + 1 > 0
    )
    total_outcomes = sum(s.occurrences for s in raw_patterns.values())

    out(f"  Encoding + mining completed in {encode_time:.1f}s")
    out(f"  Windows scanned: {total_windows:,} (lengths {'-'.join(str(n) for n in miner._pattern_lengths)})")
    out(f"  Unique pattern keys: {len(raw_patterns):,}")
    out(f"  Total pattern occurrences: {total_outcomes:,}")

    # ── Filter ────────────────────────────────────────────────────────
    out()
    out(_box_title(f"STAGE 2: STATISTICAL FILTERING ({timeframe.upper()})"))
    out()

    # Count at each stage for reporting
    n_raw = len(raw_patterns)
    after_min = sum(1 for s in raw_patterns.values() if s.occurrences >= config.min_occurrences)
    out(f"  Raw patterns: {n_raw:,}")
    out(f"  After min-occurrence ({config.min_occurrences}): {after_min}")

    if config.regime_filter:
        out(f"  Regime filter: {config.regime_filter}")
    if config.min_effect_size > 0:
        out(f"  Min effect size (Cohen's d): {config.min_effect_size}")

    t0 = time.time()
    discovered = filter_patterns(raw_patterns, config, timeframe=timeframe)
    n_filtered = len(discovered)
    conf_count = sum(1 for p in discovered if p.confirmation)
    out(f"  After t-test + effect size + WR + R:R + 3-fold WF: {n_filtered}")
    out(f"  Patterns with indicator confirmation: {conf_count}")

    # FDR correction: optional when multi-filter pipeline (effect size,
    # walk-forward, R:R) already controls false discovery rate.
    if config.apply_fdr:
        discovered = apply_fdr_correction(discovered)
        n_fdr = len(discovered)
    else:
        n_fdr = n_filtered
    if config.apply_fdr:
        out(f"  After Benjamini-Hochberg FDR correction (5%%): {n_fdr}")
    else:
        out(f"  FDR correction: skipped (multi-filter pipeline active)")

    discovered = deduplicate_patterns(discovered)
    n_dedup = len(discovered)
    out(f"  After deduplication: {n_dedup}")

    filter_time = time.time() - t0
    out(f"  Filtering completed in {filter_time:.1f}s")

    # Build catalog
    date_start = data[0].date[:10] if data else ""
    date_end = data[split_idx - 1].date[:10] if split_idx > 0 else ""
    catalog = PatternCatalog(
        patterns=tuple(discovered),
        discovery_date=datetime.now(timezone.utc).isoformat()[:19],
        data_range=f"{date_start} to {date_end}",
        total_bars_scanned=len(discovery_data) - warmup,
        total_patterns_tested=n_raw,
    )

    stats = {
        "n_raw": n_raw,
        "after_min": after_min,
        "n_filtered": n_filtered,
        "n_fdr": n_fdr,
        "n_dedup": n_dedup,
        "total_windows": total_windows,
    }
    return catalog, stats


def _print_pattern_catalog(
    catalog: PatternCatalog,
    out,
    *,
    title: str = "DISCOVERED PATTERNS",
) -> None:
    """Print the discovered pattern catalog table."""
    if not catalog.patterns:
        out()
        out("  No patterns survived filtering.")
        return

    out()
    out(_box_title(title))
    out()
    out(
        f"  {'#':<3s} {'Pattern':<50s} {'Dir':<6s} "
        f"{'Horiz':>5s} {'Count':>5s} {'AvgRet':>7s} {'WR':>6s} "
        f"{'t-stat':>7s} {'p-val':>7s} {'d':>5s} {'R:R':>5s} {'WF':>3s} {'Conf':>6s}"
    )
    out("  " + "\u2500" * 125)

    for i, p in enumerate(catalog.patterns, 1):
        wf_str = "OK" if p.walk_forward_stable else "  "
        conf_str = "YES" if p.confirmation else "-"
        out(
            f"  {i:<3d} {p.human_label:<50s} {p.direction.upper():<6s} "
            f"{p.best_horizon:>5d} {p.occurrences:>5d} "
            f"{p.avg_return:>+6.3f}% {p.win_rate * 100:>5.1f}% "
            f"{p.t_stat:>7.2f} {p.p_value:>7.4f} "
            f"{p.effect_size:>5.2f} "
            f"{p.risk_reward:>5.1f} {wf_str:>3s} {conf_str:>6s}"
        )
        if p.confirmation:
            rules = ", ".join(f"{k}={v}" for k, v in p.confirmation.items())
            out(f"      \u2514\u2500 confirm: {rules}")
        if p.regime_breakdown:
            rb_str = ", ".join(
                f"{r}:{c}" for r, c in sorted(
                    p.regime_breakdown.items(), key=lambda x: -x[1],
                )
            )
            out(f"      \u2514\u2500 regimes: {rb_str}"
                f" (dominant: {p.dominant_regime or 'N/A'})")


def _print_combined_leaderboard(
    all_catalogs: dict[str, PatternCatalog],
    out,
) -> None:
    """Print a combined leaderboard of patterns from all timeframes."""
    all_patterns: list[tuple[str, DiscoveredPattern]] = []
    for tf_label, catalog in all_catalogs.items():
        for p in catalog.patterns:
            all_patterns.append((tf_label, p))

    if not all_patterns:
        out()
        out("  No patterns discovered across any timeframe.")
        return

    # Sort by edge strength
    all_patterns.sort(
        key=lambda x: abs(x[1].t_stat) * math.sqrt(x[1].occurrences),
        reverse=True,
    )

    out()
    out(_box_title("COMBINED MULTI-TIMEFRAME LEADERBOARD"))
    out()
    out(
        f"  {'#':<3s} {'TF':<5s} {'Pattern':<45s} {'Dir':<6s} "
        f"{'Horiz':>5s} {'Count':>5s} {'AvgRet':>7s} {'WR':>6s} "
        f"{'t-stat':>7s} {'Conf':>6s}"
    )
    out("  " + "\u2500" * 105)

    for i, (tf, p) in enumerate(all_patterns[:50], 1):
        conf_str = "YES" if p.confirmation else "-"
        out(
            f"  {i:<3d} {tf:<5s} {p.human_label:<45s} {p.direction.upper():<6s} "
            f"{p.best_horizon:>5d} {p.occurrences:>5d} "
            f"{p.avg_return:>+6.3f}% {p.win_rate * 100:>5.1f}% "
            f"{p.t_stat:>7.2f} {conf_str:>6s}"
        )


def _backtest_patterns(
    catalog: PatternCatalog,
    validation_data,
    out,
    *,
    tf_label: str = "5m",
) -> None:
    """Backtest discovered patterns on out-of-sample data."""
    from stockdownloader.strategies.intraday.pattern_discovery import (
        PatternDiscoveryStrategy,
    )

    if not catalog.patterns:
        out()
        out("  No patterns to backtest.")
        return

    out()
    out(_box_title(f"OUT-OF-SAMPLE BACKTEST ({tf_label.upper()})"))
    out()

    strategy = PatternDiscoveryStrategy(catalog)
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        dd_throttle=True,
    )

    t0 = time.time()
    result = engine.run(strategy, validation_data)
    elapsed = time.time() - t0

    trading_days = len({d.date[:10] for d in validation_data})
    oos_score = score_v2(result, trading_days)

    out(f"  Validation period: {validation_data[0].date[:10]} to "
        f"{validation_data[-1].date[:10]} ({len(validation_data):,} bars, "
        f"{trading_days} days)")
    out(f"  Backtest completed in {elapsed:.1f}s")
    out()
    out(f"  Total Trades:    {result.total_trades}")
    out(f"  Win Rate:        {result.win_rate:.1f}%")
    out(f"  Profit Factor:   {result.profit_factor:.2f}")
    out(f"  Total P/L:       ${result.total_pnl:+,.2f}")
    out(f"  Total Return:    {result.total_return:+.2f}%")
    out(f"  Max Drawdown:    {result.max_drawdown:.2f}%")
    out(f"  Score (v2):      {oos_score:.1f}")

    # ── Monte Carlo validation ────────────────────────────────────────
    if result.total_trades >= 5:
        out()
        out("  Monte Carlo Robustness (1000 simulations):")
        pnls = [float(t.profit_loss) for t in result.closed_trades]
        from stockdownloader.backtesting.tournament.engine import ComboKey
        dummy_key = ComboKey(f"pattern-discovery-{tf_label}", tf_label)
        _, mc, _, error = run_monte_carlo(
            dummy_key, pnls, float(INITIAL_CAPITAL), 1000,
        )
        if mc is not None:
            robust_str = "YES" if mc.is_robust else "NO"
            out(f"    Bootstrap Return p5:  {mc.bootstrap_return.p5:+.2f}%")
            out(f"    Bootstrap Return p50: {mc.bootstrap_return.p50:+.2f}%")
            out(f"    Max Drawdown p95:     {mc.max_drawdown.p95:.2f}%")
            out(f"    Robust:               {robust_str}")
        elif error:
            out(f"    MC Error: {error}")
        else:
            out("    MC: insufficient trades")

    # ── Walk-forward validation on OOS data ────────────────────────
    if result.total_trades >= 10:
        out()
        out("  Walk-Forward Validation (3 windows):")
        try:
            def _strategy_factory():
                return PatternDiscoveryStrategy(catalog)

            wfv = WalkForwardValidator(validation_data, n_windows=3)
            wf_engine = IntradayBacktestEngine(
                initial_capital=INITIAL_CAPITAL,
                risk_per_trade=RISK_PER_TRADE,
                dd_throttle=True,
            )
            wf_result = wfv.validate(
                _strategy_factory,
                wf_engine,
                strategy_name=f"pattern-discovery-{tf_label}",
            )
            out(f"    IS score:       {wf_result.in_sample_score:.1f}")
            out(f"    OOS score:      {wf_result.out_of_sample_score:.1f}")
            out(f"    Degradation:    {wf_result.degradation_ratio:.2f}"
                f"  ({'robust' if wf_result.degradation_ratio >= 0.5 else 'OVERFIT'})")
        except Exception as e:
            out(f"    WF error: {e}")


# ======================================================================
# Main
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """Run the pattern discovery pipeline."""
    args = _parse_args(argv)

    # Setup output
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUTPUT_DIR / "pattern_discovery.log"
    log_file = open(log_path, "w")
    tee = TeeWriter(log_file)

    def out(msg: str = "") -> None:
        tee.write(msg + "\n")

    t_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    out(_box_title("PRICE ACTION PATTERN DISCOVERY ENGINE"))
    out()

    data_path = args.data_file
    out(f"Loading data from {data_path}...")
    raw_data = IntradayCsvLoader.load_from_file(str(data_path))
    out(f"Loaded {len(raw_data):,} 5-minute bars")
    out(f"Date range: {raw_data[0].date} to {raw_data[-1].date}")

    # ── Parse timeframes ──────────────────────────────────────────────
    tf_labels = [s.strip() for s in args.timeframes.split(",")]
    tf_labels = [tf for tf in tf_labels if tf in _TF_MAP]
    if not tf_labels:
        out("ERROR: No valid timeframes specified")
        log_file.close()
        return

    out(f"Timeframes: {', '.join(tf_labels)}")
    out()

    # ── Build data per timeframe ──────────────────────────────────────
    agg = TimeframeAggregator(raw_data)
    tf_datasets: dict[str, list] = {}
    for tf_label in tf_labels:
        tf_data = agg.as_intraday_price_data(_TF_MAP[tf_label])
        tf_datasets[tf_label] = tf_data
        out(f"  {tf_label}: {len(tf_data):,} bars")

    # ── Base filter config ────────────────────────────────────────────
    base_filter_config = FilterConfig(
        min_occurrences=args.min_occurrences,
        max_patterns=args.max_patterns,
        regime_filter=args.regime,
    )

    all_catalogs: dict[str, PatternCatalog] = {}
    catalog_path = args.catalog_file or (_OUTPUT_DIR / "patterns/catalog.json")

    # ── Discovery per timeframe ───────────────────────────────────────
    if args.mode in ("discover", "full"):
        for tf_label in tf_labels:
            tf_data = tf_datasets[tf_label]
            split_idx = int(len(tf_data) * args.discovery_split)

            # Temporal gap: skip bars between discovery and validation
            gap_bars = args.gap_bars if args.gap_bars is not None else _DEFAULT_GAP.get(tf_label, 100)
            val_start = min(split_idx + gap_bars, len(tf_data))

            out()
            out("=" * 100)
            out(f"  TIMEFRAME: {tf_label.upper()} ({len(tf_data):,} bars)")
            out(f"  Discovery: bars 0-{split_idx:,} ({args.discovery_split:.0%})")
            out(f"  Temporal gap: {gap_bars} bars (indicator leakage buffer)")
            out(f"  Validation: bars {val_start:,}-{len(tf_data):,}")
            out("=" * 100)

            # Pre-classify regimes on discovery data
            out()
            out("  Classifying market regimes...")
            regime_map = classify_timeframe_bars(tf_data[:split_idx])
            regime_counts: dict[str, int] = {}
            for regime in regime_map.values():
                r = regime.value
                regime_counts[r] = regime_counts.get(r, 0) + 1
            for r_name, r_count in sorted(regime_counts.items(), key=lambda x: -x[1]):
                out(f"    {r_name}: {r_count} days")

            # Scale filter config for this timeframe
            tf_config = base_filter_config.for_timeframe(tf_label)

            catalog, stats = _discover_patterns(
                tf_data, split_idx, tf_config, out,
                timeframe=tf_label,
                regime_map=regime_map,
            )

            _print_pattern_catalog(
                catalog, out,
                title=f"DISCOVERED PATTERNS ({tf_label.upper()})",
            )

            # Save per-timeframe catalog
            tf_catalog_path = _OUTPUT_DIR / f"patterns/spy/catalog_{tf_label}.json"
            catalog.save(tf_catalog_path)
            out()
            out(f"  Catalog saved to: {tf_catalog_path}")

            all_catalogs[tf_label] = catalog

        # Combined leaderboard (if multiple TFs)
        if len(tf_labels) > 1:
            _print_combined_leaderboard(all_catalogs, out)

        # Save combined catalog
        combined_patterns: list[DiscoveredPattern] = []
        for catalog in all_catalogs.values():
            combined_patterns.extend(catalog.patterns)

        combined_catalog = PatternCatalog(
            patterns=tuple(combined_patterns),
            discovery_date=datetime.now(timezone.utc).isoformat()[:19],
            data_range="multi-timeframe",
            total_bars_scanned=sum(
                c.total_bars_scanned for c in all_catalogs.values()
            ),
            total_patterns_tested=sum(
                c.total_patterns_tested for c in all_catalogs.values()
            ),
        )
        combined_catalog.save(catalog_path)
        out()
        out(f"  Combined catalog saved to: {catalog_path}")

    elif args.mode == "backtest":
        # Load existing catalog
        out(f"Loading catalog from {catalog_path}...")
        combined_catalog = PatternCatalog.load(catalog_path)
        out(f"Loaded {len(combined_catalog.patterns)} patterns")
        # Group by timeframe for per-TF backtesting
        for p in combined_catalog.patterns:
            tf = p.timeframe
            if tf not in all_catalogs:
                all_catalogs[tf] = PatternCatalog(patterns=())
        for tf in all_catalogs:
            tf_patterns = tuple(
                p for p in combined_catalog.patterns if p.timeframe == tf
            )
            all_catalogs[tf] = PatternCatalog(patterns=tf_patterns)

    # ── Backtest per timeframe ────────────────────────────────────────
    if args.mode in ("backtest", "full"):
        for tf_label, catalog in all_catalogs.items():
            if tf_label not in tf_datasets:
                continue
            tf_data = tf_datasets[tf_label]
            split_idx = int(len(tf_data) * args.discovery_split)
            gap_bars = args.gap_bars if args.gap_bars is not None else _DEFAULT_GAP.get(tf_label, 100)
            val_start = min(split_idx + gap_bars, len(tf_data))
            validation_data = tf_data[val_start:]
            if validation_data and catalog.patterns:
                _backtest_patterns(
                    catalog, validation_data, out,
                    tf_label=tf_label,
                )

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_patterns = sum(len(c.patterns) for c in all_catalogs.values())
    out()
    out("=" * 100)
    out(f"  Pattern discovery completed in {elapsed:.1f}s")
    out(f"  Timeframes scanned: {', '.join(tf_labels)}")
    out(f"  Total patterns discovered: {total_patterns}")
    out(f"  Results logged to: {log_path}")
    out("=" * 100)
    out()

    log_file.close()


if __name__ == "__main__":
    main()
