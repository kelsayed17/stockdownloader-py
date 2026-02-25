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
        "--buy-write", action="store_true",
        help=(
            "Buy-write mode: hold shares + sell covered calls for income. "
            "Captures buy-and-hold returns plus premium."
        ),
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Combined buy-write + CSP mode: sell CCs on shares + CSPs on idle cash.",
    )
    # Override commission default from add_common_ml_args (10.0 -> 0.65)
    parser.set_defaults(commission=0.65)
    parser.add_argument(
        "--iv-filter", action="store_true",
        help="Enable IV-based filtering (skip selling when IV too low)",
    )
    parser.add_argument(
        "--min-iv-percentile", type=float, default=0.30,
        help="Minimum IV percentile to sell options (default: 0.30)",
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
    *,
    buy_write: bool = False,
) -> None:
    """Print side-by-side comparison table."""
    mode = "BUY-WRITE" if buy_write else "WHEEL"
    ml_label = f"ML {mode.title()}" if not buy_write else "ML Buy-Write"
    mech_label = "Mechanical" if not buy_write else "Buy-Write"

    print("\n" + "=" * 70)
    print(f"SPY WEEKLY {mode} BACKTEST RESULTS")
    print("=" * 70)

    header_line = f"  {'':24}"
    if ml_metrics:
        header_line += f"{ml_label:>14}"
    header_line += f"{mech_label:>14}{'Buy & Hold':>14}"
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
    if buy_write:
        _row(
            "Calls Exercised",
            f"{int(ml.get('n_calls_exercised', 0))}" if ml else "",
            f"{int(mech_metrics['n_calls_exercised'])}",
            "N/A",
        )
        _row(
            "Re-buys",
            f"{int(ml.get('n_rebuys', 0))}" if ml else "",
            f"{int(mech_metrics.get('n_rebuys', 0))}",
            "N/A",
        )
    else:
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
            "Calls Skipped (ML)",
            f"{int(ml.get('n_calls_skipped', 0))}",
            "N/A",
            "N/A",
        )
        if not buy_write:
            _row(
                "Puts Skipped (ML)",
                f"{int(ml.get('n_puts_skipped', 0))}",
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


def _print_comparison_table(
    ml_combined: dict[str, float] | None,
    mech_combined: dict[str, float],
    bw_metrics: dict[str, float],
    wheel_metrics: dict[str, float],
    bh_return_pct: float,
) -> None:
    """Print comparison: ML Combined | Combined | Buy-Write | Wheel | Buy & Hold."""
    print("\n" + "=" * 92)
    print("SPY WEEKLY COMBINED STRATEGY COMPARISON")
    print("=" * 92)

    cols = []
    if ml_combined:
        cols.append(("ML Combined", ml_combined))
    cols.append(("Combined", mech_combined))
    cols.append(("Buy-Write", bw_metrics))
    cols.append(("Wheel", wheel_metrics))

    header = f"  {'':24}"
    for name, _ in cols:
        header += f"{name:>14}"
    header += f"{'Buy & Hold':>14}"
    print(header)
    print("  " + "-" * (24 + 14 * (len(cols) + 1)))

    def _row(label: str, key: str, fmt: str = ".1f", prefix: str = "", suffix: str = "%") -> None:
        parts = [f"  {label:<24}"]
        for _, m in cols:
            val = m.get(key, 0)
            parts.append(f"{prefix}{val:{fmt}}{suffix}".rjust(14))
        if key == "total_return_pct":
            parts.append(f"+{bh_return_pct:.1f}%".rjust(14))
        else:
            parts.append("N/A".rjust(14))
        print("".join(parts))

    def _row_int(label: str, key: str) -> None:
        parts = [f"  {label:<24}"]
        for _, m in cols:
            parts.append(f"{int(m.get(key, 0))}".rjust(14))
        parts.append("N/A".rjust(14))
        print("".join(parts))

    def _row_dollar(label: str, key: str) -> None:
        parts = [f"  {label:<24}"]
        for _, m in cols:
            parts.append(f"${m.get(key, 0):,.0f}".rjust(14))
        parts.append("N/A".rjust(14))
        print("".join(parts))

    _row("Total Return", "total_return_pct", prefix="+")
    _row("Annualized Return", "annualized_return_pct", prefix="+")
    _row_dollar("Premium Collected", "total_premium_collected")
    _row_dollar("Commissions", "total_commissions")
    _row_int("Puts Sold", "n_puts_sold")
    _row_int("Calls Sold", "n_calls_sold")
    _row_int("Assignments", "n_assignments")
    _row_int("Calls Exercised", "n_calls_exercised")
    _row_int("Re-buys", "n_rebuys")
    _row("Sharpe", "sharpe", fmt=".2f", prefix="", suffix="")
    _row("Max Drawdown", "max_drawdown_pct", prefix="-")

    print("=" * 92)


def main(argv: list[str] | None = None) -> None:
    """Run the SPY Weekly Wheel pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    init_ml_env(args.verbose)

    import numpy as np

    from stockdownloader.analysis.options.pricing import (
        estimate_volatility,
        implied_volatility,
    )
    from stockdownloader.core.models.options import OptionType
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

    if args.combined:
        mode_label = "COMBINED"
    elif args.buy_write:
        mode_label = "BUY-WRITE"
    else:
        mode_label = "WHEEL"

    print("=" * 60)
    print(f"SPY ML-GUIDED WEEKLY {mode_label} PIPELINE")
    print("=" * 60)
    print(f"  Mode:          {mode_label}")
    print(f"  Delta:         {args.delta}")
    print(f"  Contracts:     {args.contracts}")
    print(f"  Capital:       ${args.initial_capital:,.0f}")
    print(f"  Period:        {args.from_date} to {to_date_str}")
    print(f"  ML Filter:     {'OFF' if args.no_ml_filter else 'ON'}")
    if not args.no_ml_filter:
        if not args.buy_write:
            print(f"  Skip put <     {args.skip_put_thresh}")
        print(f"  Skip call >    {args.skip_call_thresh}")
    print(f"  Cache:         {cache_dir}")
    print(f"  Commission:    ${args.commission:.2f}/contract")
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
    iv_history: list[float] = []
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

        put_premium = put_bar.get("vw", put_bar.get("c", 0.0)) if put_bar else 0.0
        call_premium = call_bar.get("vw", call_bar.get("c", 0.0)) if call_bar else 0.0

        # Weekly IV estimation
        weekly_vol = hist_vol
        if put_bar and call_bar and spy_at_entry > 0:
            try:
                put_iv = float(implied_volatility(
                    OptionType.PUT, Decimal(str(put_premium)),
                    Decimal(str(spy_at_entry)), Decimal(str(put_contract["strike_price"])),
                    Decimal(str(5 / 365)), Decimal("0.05"),
                ))
                call_iv = float(implied_volatility(
                    OptionType.CALL, Decimal(str(call_premium)),
                    Decimal(str(spy_at_entry)), Decimal(str(call_contract["strike_price"])),
                    Decimal(str(5 / 365)), Decimal("0.05"),
                ))
                weekly_vol = (put_iv + call_iv) / 2
            except Exception:
                pass

        iv_history.append(weekly_vol)
        iv_percentile = 0.5
        if len(iv_history) >= 10:
            sorted_ivs = sorted(iv_history)
            rank = sum(1 for v in sorted_ivs if v <= weekly_vol)
            iv_percentile = rank / len(sorted_ivs)

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

    # Run mechanical (no ML filter)
    mech_engine = WheelBacktestEngine(
        initial_capital=args.initial_capital,
        contracts=args.contracts,
        buy_write=args.buy_write,
        commission_per_contract=args.commission,
    )
    for w in week_records:
        mech_engine.process_week(w, use_ml_filter=False)
    mech_metrics = mech_engine.compute_metrics()

    # Run ML-filtered
    ml_metrics = None
    if not args.no_ml_filter and ml_probs:
        ml_engine = WheelBacktestEngine(
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            skip_put_thresh=args.skip_put_thresh,
            skip_call_thresh=args.skip_call_thresh,
            buy_write=args.buy_write,
            commission_per_contract=args.commission,
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

    if args.combined:
        # Run all engines for comparison
        # 1. ML Combined (if ML is on)
        ml_combined_metrics = None
        if not args.no_ml_filter and ml_probs:
            ml_combined = WheelBacktestEngine(
                initial_capital=args.initial_capital,
                contracts=args.contracts,
                skip_put_thresh=args.skip_put_thresh,
                skip_call_thresh=args.skip_call_thresh,
                combined=True,
                commission_per_contract=args.commission,
            )
            for w in week_records:
                ml_combined.process_week(w, use_ml_filter=True)
            ml_combined_metrics = ml_combined.compute_metrics()

        # 2. Mechanical Combined (always run)
        mech_combined = WheelBacktestEngine(
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            combined=True,
            commission_per_contract=args.commission,
        )
        for w in week_records:
            mech_combined.process_week(w, use_ml_filter=False)
        mech_combined_metrics = mech_combined.compute_metrics()

        # 3. Mechanical Buy-Write
        bw_engine = WheelBacktestEngine(
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            buy_write=True,
            commission_per_contract=args.commission,
        )
        for w in week_records:
            bw_engine.process_week(w, use_ml_filter=False)
        bw_metrics = bw_engine.compute_metrics()

        # 4. Mechanical Wheel
        wheel_engine = WheelBacktestEngine(
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            commission_per_contract=args.commission,
        )
        for w in week_records:
            wheel_engine.process_week(w, use_ml_filter=False)
        wheel_metrics = wheel_engine.compute_metrics()

        _print_comparison_table(
            ml_combined_metrics, mech_combined_metrics,
            bw_metrics, wheel_metrics, bh_return_pct,
        )
    else:
        _print_results_table(
            ml_metrics, mech_metrics, bh_return_pct,
            buy_write=args.buy_write,
        )

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
