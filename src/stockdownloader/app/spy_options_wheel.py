"""SPY ML-Guided Weekly Wheel Strategy pipeline.

End-to-end pipeline that:
  [1/5] Downloads SPY daily data
  [2/5] Runs ML walk-forward ensemble for probability predictions
  [3/5] Downloads option chain data from Polygon (with caching)
  [4/5] Runs wheel backtest (ML-filtered vs mechanical)
  [5/5] Prints comparison results

Usage::

    spy-options-wheel                        # Full pipeline with ML filter
    spy-options-wheel --no-ml-filter         # Pure mechanical wheel
    spy-options-wheel --delta 0.20           # Conservative strike selection
    spy-options-wheel --contracts 2          # 2 contracts per trade
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from stockdownloader.app.ml_helpers import add_common_ml_args, init_ml_env
from stockdownloader.core.config import DEFAULT_ML_PIPELINE_DIR


_FULL_GRID = [
    ("gradient_boosting", False),
    ("random_forest", False),
    ("logistic_regression", False),
    ("gradient_boosting", True),
    ("random_forest", True),
]

_QUICK_GRID = [
    ("gradient_boosting", False),
    ("random_forest", False),
    ("logistic_regression", False),
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``spy-options-wheel``."""
    parser = argparse.ArgumentParser(
        prog="spy-options-wheel",
        description=(
            "SPY ML-Guided Weekly Wheel: download SPY data -> "
            "ML walk-forward predictions -> Polygon option prices -> "
            "wheel backtest with ML filter."
        ),
    )

    add_common_ml_args(parser)

    parser.add_argument(
        "--delta", type=float, default=0.30,
        help="Target delta for strike selection (default: 0.30)",
    )
    parser.add_argument(
        "--contracts", type=int, default=1,
        help="Number of option contracts per trade (default: 1)",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Initial capital in dollars (default: 100000)",
    )
    parser.add_argument(
        "--from-date", type=str, default="2022-01-01",
        help="Backtest start date YYYY-MM-DD (default: 2022-01-01)",
    )
    parser.add_argument(
        "--to-date", type=str, default=None,
        help="Backtest end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--no-ml-filter", action="store_true",
        help="Run pure mechanical wheel without ML filter",
    )
    parser.add_argument(
        "--skip-put-thresh", type=float, default=0.35,
        help="Skip selling puts when ML prob < this (default: 0.35)",
    )
    parser.add_argument(
        "--skip-call-thresh", type=float, default=0.65,
        help="Skip selling calls when ML prob > this (default: 0.65)",
    )
    parser.add_argument(
        "--walk-forward-windows", type=int, default=5,
        help="Number of expanding windows for walk-forward (default: 5)",
    )
    parser.add_argument(
        "--top-models", type=int, default=3,
        help="Number of top models in ensemble (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for Polygon options cache (default: data/options_cache/SPY)",
    )

    return parser


def _get_fridays_in_range(from_date: date, to_date: date) -> list[date]:
    """Return all Fridays between from_date and to_date inclusive."""
    fridays: list[date] = []
    current = from_date
    while current.weekday() != 4:
        current += timedelta(days=1)
    while current <= to_date:
        fridays.append(current)
        current += timedelta(days=7)
    return fridays


def _get_previous_trading_day(d: date) -> date:
    """Return the Monday before a Friday (entry day for weekly options)."""
    return d - timedelta(days=4)


def _print_results_table(
    ml_metrics: dict[str, float] | None,
    mech_metrics: dict[str, float],
    bh_return_pct: float,
) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("SPY WEEKLY WHEEL BACKTEST RESULTS")
    print("=" * 70)

    header_line = f"  {'':24}"
    if ml_metrics:
        header_line += f"{'ML Wheel':>14}"
    header_line += f"{'Mechanical':>14}{'Buy & Hold':>14}"
    print(header_line)
    print("  " + "-" * (66 if ml_metrics else 52))

    ml = ml_metrics or {}

    def _row(label: str, ml_val: str, mech_val: str, bh_val: str) -> None:
        parts = [f"  {label:<24}"]
        if ml_metrics:
            parts.append(f"{ml_val:>14}")
        parts.extend([f"{mech_val:>14}", f"{bh_val:>14}"])
        print("".join(parts))

    _row(
        "Total Return",
        f"+{ml.get('total_return_pct', 0):.1f}%" if ml else "",
        f"+{mech_metrics['total_return_pct']:.1f}%",
        f"+{bh_return_pct:.1f}%",
    )
    _row(
        "Annualized Return",
        f"+{ml.get('annualized_return_pct', 0):.1f}%" if ml else "",
        f"+{mech_metrics['annualized_return_pct']:.1f}%",
        "N/A",
    )
    _row(
        "Premium Collected",
        f"${ml.get('total_premium_collected', 0):,.0f}" if ml else "",
        f"${mech_metrics['total_premium_collected']:,.0f}",
        "N/A",
    )
    _row(
        "Assignments",
        f"{int(ml.get('n_assignments', 0))}" if ml else "",
        f"{int(mech_metrics['n_assignments'])}",
        "N/A",
    )
    _row(
        "Calls Exercised",
        f"{int(ml.get('n_calls_exercised', 0))}" if ml else "",
        f"{int(mech_metrics['n_calls_exercised'])}",
        "N/A",
    )
    if ml:
        _row(
            "Weeks Skipped (ML)",
            f"{int(ml.get('n_puts_skipped', 0) + ml.get('n_calls_skipped', 0))}",
            "N/A",
            "N/A",
        )
    _row(
        "Sharpe",
        f"{ml.get('sharpe', 0):.2f}" if ml else "",
        f"{mech_metrics['sharpe']:.2f}",
        "N/A",
    )
    _row(
        "Max Drawdown",
        f"-{ml.get('max_drawdown_pct', 0):.1f}%" if ml else "",
        f"-{mech_metrics['max_drawdown_pct']:.1f}%",
        "N/A",
    )
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    """Run the SPY Weekly Wheel pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    init_ml_env(args.verbose)

    import numpy as np

    from stockdownloader.analysis.options.pricing import estimate_volatility
    from stockdownloader.backtesting.engines.wheel import (
        WeekRecord,
        WheelBacktestEngine,
    )
    from stockdownloader.data.market.polygon_options_client import (
        PolygonOptionsClient,
        load_chain_cache,
        save_chain_cache,
        select_strike_by_delta,
    )
    from stockdownloader.data.market.yahoo_data_client import YahooDataClient
    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.feature_extractor import FeatureExtractor

    t0 = time.time()

    grid = _QUICK_GRID if args.quick else _FULL_GRID
    to_date_str = args.to_date or str(date.today())

    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_ML_PIPELINE_DIR / "spy_wheel"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path("data/options_cache/SPY")
    )

    print("=" * 60)
    print("SPY ML-GUIDED WEEKLY WHEEL PIPELINE")
    print("=" * 60)
    print(f"  Delta:         {args.delta}")
    print(f"  Contracts:     {args.contracts}")
    print(f"  Capital:       ${args.initial_capital:,.0f}")
    print(f"  Period:        {args.from_date} to {to_date_str}")
    print(f"  ML Filter:     {'OFF' if args.no_ml_filter else 'ON'}")
    if not args.no_ml_filter:
        print(f"  Skip put <     {args.skip_put_thresh}")
        print(f"  Skip call >    {args.skip_call_thresh}")
    print(f"  Cache:         {cache_dir}")
    print()

    # [1/5] Download SPY daily data
    print("[1/5] Downloading SPY daily data (10y)...")
    t1 = time.time()

    yahoo = YahooDataClient()
    daily_data = yahoo.fetch_price_data("SPY", range_="10y")

    if len(daily_data) < 500:
        print(f"ERROR: Insufficient data ({len(daily_data)} bars)", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(daily_data)} bars ({daily_data[0].date} to {daily_data[-1].date})")
    print(f"  [{time.time() - t1:.1f}s]")

    date_close: dict[str, float] = {}
    for bar in daily_data:
        date_close[bar.date] = float(bar.close)

    # [2/5] ML walk-forward predictions
    ml_probs: dict[str, float] = {}

    if not args.no_ml_filter:
        print("\n[2/5] Running ML walk-forward ensemble...")
        t2 = time.time()

        from stockdownloader.app.spy_ml_ensemble import (
            _generate_walk_forward_predictions,
        )

        label_config = LabelConfig(
            forward_period=int(args.forward_periods.split(",")[0])
            if hasattr(args, "forward_periods") else 10,
            profit_threshold=float(args.profit_thresholds.split(",")[0])
            if hasattr(args, "profit_thresholds") else 0.005,
        )
        extractor = FeatureExtractor()
        builder = DatasetBuilder(extractor, label_config)
        dataset = builder.build(daily_data)

        oos_preds, oos_dates, n_windows = _generate_walk_forward_predictions(
            dataset, daily_data, grid,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            top_models=args.top_models,
            use_direct_ensemble=True,
            n_windows=args.walk_forward_windows,
        )

        for dt, prob in zip(oos_dates, oos_preds):
            ml_probs[dt] = float(prob)

        print(f"  {len(ml_probs)} daily predictions generated")
        print(f"  [{time.time() - t2:.1f}s]")
    else:
        print("\n[2/5] Skipped (--no-ml-filter)")

    # [3/5] Download option chain data from Polygon
    print("\n[3/5] Downloading option chain data from Polygon...")
    t3 = time.time()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_dt = datetime.strptime(to_date_str, "%Y-%m-%d").date()
    fridays = _get_fridays_in_range(from_dt, to_dt)

    polygon = PolygonOptionsClient()

    all_contracts = polygon.fetch_option_contracts(
        "SPY", from_date=from_dt, to_date=to_dt, expired=True,
    )
    print(f"  {len(all_contracts)} total contracts discovered")

    contracts_by_exp: dict[str, list[dict]] = {}
    for c in all_contracts:
        exp = c.get("expiration_date", "")
        contracts_by_exp.setdefault(exp, []).append(c)

    print(f"  {len(fridays)} Fridays in range")
    print(f"  [{time.time() - t3:.1f}s]")

    # [4/5] Build weekly schedule and run wheel backtest
    print("\n[4/5] Running wheel backtest...")
    t4 = time.time()

    close_prices = [Decimal(str(bar.close)) for bar in daily_data]
    hist_vol = float(estimate_volatility(close_prices, 20))

    week_records: list[WeekRecord] = []
    n_cached = 0
    n_fetched = 0

    for week_num, friday in enumerate(fridays):
        exp_str = str(friday)
        monday = _get_previous_trading_day(friday)
        mon_str = str(monday)

        spy_at_entry = date_close.get(mon_str)
        spy_at_expiry = date_close.get(exp_str)

        if spy_at_entry is None or spy_at_expiry is None:
            continue

        week_contracts = contracts_by_exp.get(exp_str, [])
        if not week_contracts:
            continue

        put_contract = select_strike_by_delta(
            week_contracts, contract_type="put", spot=spy_at_entry,
            target_delta=args.delta, days_to_expiry=5, volatility=hist_vol,
        )
        call_contract = select_strike_by_delta(
            week_contracts, contract_type="call", spot=spy_at_entry,
            target_delta=args.delta, days_to_expiry=5, volatility=hist_vol,
        )

        if put_contract is None or call_contract is None:
            continue

        cached = load_chain_cache(cache_dir, exp_str)
        if cached and put_contract["ticker"] in cached.get("bars", {}):
            put_bar = cached["bars"].get(put_contract["ticker"])
            call_bar = cached["bars"].get(call_contract["ticker"])
            n_cached += 1
        else:
            put_bar = polygon.fetch_option_daily_bar(put_contract["ticker"], monday)
            call_bar = polygon.fetch_option_daily_bar(call_contract["ticker"], monday)
            bars_data = {}
            if put_bar:
                bars_data[put_contract["ticker"]] = put_bar
            if call_bar:
                bars_data[call_contract["ticker"]] = call_bar
            cache_data = {
                "expiration_date": exp_str,
                "contracts": week_contracts,
                "bars": bars_data,
            }
            save_chain_cache(cache_dir, exp_str, cache_data)
            n_fetched += 1

        put_premium = put_bar["c"] if put_bar else 0.0
        call_premium = call_bar["c"] if call_bar else 0.0

        if put_premium <= 0 and call_premium <= 0:
            continue

        ml_prob = 0.50
        if ml_probs:
            for lookback in range(5):
                check_date = str(monday - timedelta(days=lookback))
                if check_date in ml_probs:
                    ml_prob = ml_probs[check_date]
                    break

        week_records.append(WeekRecord(
            week_num=week_num,
            expiration_date=exp_str,
            entry_date=mon_str,
            spy_price_at_entry=spy_at_entry,
            spy_price_at_expiry=spy_at_expiry,
            put_strike=put_contract["strike_price"],
            call_strike=call_contract["strike_price"],
            put_premium=put_premium,
            call_premium=call_premium,
            ml_prob=ml_prob,
        ))

    print(f"  {len(week_records)} tradeable weeks built")
    print(f"  Cache hits: {n_cached}, API fetches: {n_fetched}")

    # Run mechanical wheel
    mech_engine = WheelBacktestEngine(
        initial_capital=args.initial_capital,
        contracts=args.contracts,
    )
    for w in week_records:
        mech_engine.process_week(w, use_ml_filter=False)
    mech_metrics = mech_engine.compute_metrics()

    # Run ML-filtered wheel
    ml_metrics = None
    if not args.no_ml_filter and ml_probs:
        ml_engine = WheelBacktestEngine(
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            skip_put_thresh=args.skip_put_thresh,
            skip_call_thresh=args.skip_call_thresh,
        )
        for w in week_records:
            ml_engine.process_week(w, use_ml_filter=True)
        ml_metrics = ml_engine.compute_metrics()

    print(f"  [{time.time() - t4:.1f}s]")

    # [5/5] Print results
    bh_start = date_close.get(str(fridays[0])) if fridays else None
    bh_end = date_close.get(str(fridays[-1])) if fridays else None
    bh_return_pct = 0.0
    if bh_start and bh_end and bh_start > 0:
        bh_return_pct = ((bh_end - bh_start) / bh_start) * 100

    _print_results_table(ml_metrics, mech_metrics, bh_return_pct)

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
