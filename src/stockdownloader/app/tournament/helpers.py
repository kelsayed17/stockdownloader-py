"""Tournament helper utilities — formatting, data loading, skip logic."""
from __future__ import annotations

from pathlib import Path

from stockdownloader.app.app_helpers import STANDARD_TIMEFRAMES as _TIMEFRAMES
from stockdownloader.app.app_helpers import box_title as _box_title, status_label as _status_label
from stockdownloader.backtesting.tournament.engine import ComboResult
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.timeframe import Timeframe, TimeframeAggregator


# ======================================================================
# Data loading
# ======================================================================


def _load_data(
    csv_file: Path,
    extra_csvs: dict[str, Path],
    timeframe_filter: list[str] | None,
    out,
) -> dict[str, list[IntradayPriceData]]:
    """Load price data for each timeframe.

    Returns dict mapping timeframe label -> price data.
    """
    out(f"Loading base 5m data from {csv_file}...")
    raw_data = IntradayCsvLoader.load_from_file(csv_file)
    if not raw_data:
        out(f"ERROR: Could not load data from {csv_file}")
        return {}

    out(f"Loaded {len(raw_data):,} 5-minute bars")
    out(f"Date range: {raw_data[0].date} to {raw_data[-1].date}")
    trading_days = len({d.date[:10] for d in raw_data})
    out(f"Trading days: {trading_days}")

    # Determine which timeframes to test
    all_tfs = [(tf, label) for tf, label in _TIMEFRAMES]
    if timeframe_filter:
        all_tfs = [(tf, label) for tf, label in all_tfs if label in timeframe_filter]

    if not all_tfs:
        out("ERROR: No valid timeframes selected")
        return {}

    out(f"Timeframes: {', '.join(label for _, label in all_tfs)}")
    out()

    # Build data for each timeframe
    agg = TimeframeAggregator(raw_data)
    tf_data: dict[str, list[IntradayPriceData]] = {}

    for tf_enum, tf_label in all_tfs:
        if tf_label in extra_csvs:
            # Load from external CSV
            out(f"  Loading {tf_label} from {extra_csvs[tf_label]}...")
            ext_data = IntradayCsvLoader.load_from_file(extra_csvs[tf_label])
            if ext_data:
                tf_data[tf_label] = ext_data
                out(f"    {len(ext_data):,} bars loaded")
            else:
                out(f"    WARNING: Could not load {extra_csvs[tf_label]}")
        else:
            # Resample from 5m bars
            data = agg.as_intraday_price_data(tf_enum)
            tf_data[tf_label] = data
            out(f"  {tf_label}: {len(data):,} bars (resampled)")

    out()
    return tf_data


# ======================================================================
# Skip-set builder
# ======================================================================


def _build_skip_set(results: list[ComboResult]) -> set[tuple[str, str]]:
    """Build set of (strategy_name, timeframe) pairs to skip optimization.

    If a strategy has 0 trades on a timeframe, skip it and all higher TFs.
    Timeframes are ordered: 5m < 15m < 30m < 1h < 4h < 1d.
    """
    tf_order = ["5m", "15m", "30m", "1h", "4h", "1d"]
    skip: set[tuple[str, str]] = set()

    # Group by strategy
    by_strategy: dict[str, dict[str, ComboResult]] = {}
    for r in results:
        by_strategy.setdefault(r.key.strategy_name, {})[r.key.timeframe] = r

    for strat_name, tf_map in by_strategy.items():
        found_zero = False
        for tf in tf_order:
            if found_zero:
                skip.add((strat_name, tf))
                continue
            combo = tf_map.get(tf)
            if combo and combo.best_result is not None and combo.best_result.total_trades == 0:
                found_zero = True
                skip.add((strat_name, tf))

    return skip
