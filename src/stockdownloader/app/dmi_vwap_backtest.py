"""Backtest the DMI+VWAP strategy on SPY 5-minute data.

Usage::

    python -m stockdownloader.app.dmi_vwap_backtest [--capital CAPITAL]

Produces detailed trade log, summary statistics, and walk-forward
validation results.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapConfig, DmiVwapStrategy
from stockdownloader.util.big_decimal_math import ZERO

logger = logging.getLogger(__name__)

_QUANT2 = Decimal("0.01")

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

def _load_data(path: Path) -> list[IntradayPriceData]:
    bars: list[IntradayPriceData] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                IntradayPriceData(
                    date=row["Datetime"],
                    open=Decimal(row["Open"]),
                    high=Decimal(row["High"]),
                    low=Decimal(row["Low"]),
                    close=Decimal(row["Close"]),
                    adj_close=Decimal(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )
    return bars

def _run_backtest(
    data: list[IntradayPriceData],
    capital: Decimal,
    config: DmiVwapConfig | None = None,
) -> None:
    engine = IntradayBacktestEngine(
        initial_capital=capital,
        risk_per_trade=Decimal("0.01"),
    )
    strategy = DmiVwapStrategy(config)

    t0 = time.time()
    result = engine.run(strategy, data)
    elapsed = time.time() - t0

    # ---- Summary Statistics ----
    trades = result.closed_trades
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win())
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    total_pnl = sum(t.profit_loss for t in trades)
    gross_profit = sum(t.profit_loss for t in trades if t.is_win())
    gross_loss = sum(t.profit_loss for t in trades if not t.is_win())

    avg_win = gross_profit / Decimal(str(wins)) if wins > 0 else ZERO
    avg_loss = abs(gross_loss) / Decimal(str(losses)) if losses > 0 else ZERO
    avg_rr = float(avg_win / avg_loss) if avg_loss > ZERO else 0.0

    max_dd = result.max_drawdown
    _BARS_PER_YEAR = 252 * 78  # 5-min bars: 78 bars/day × 252 days
    sharpe = result.sharpe_ratio(trading_days_per_year=_BARS_PER_YEAR)
    sortino = result.sortino_ratio(trading_days_per_year=_BARS_PER_YEAR)
    calmar = result.calmar_ratio(bars_per_year=_BARS_PER_YEAR)
    pf = result.profit_factor
    consec_losses = result.max_consecutive_losses
    consec_wins = result.max_consecutive_wins
    avg_duration = result.avg_trade_duration_bars
    v2_score = score_v2(result)

    # ---- Print Results ----
    print("=" * 80)
    print("DMI + VWAP STRATEGY BACKTEST RESULTS")
    print("=" * 80)
    print()
    print(f"  Data: {data[0].date[:10]} to {data[-1].date[:10]}")
    print(f"  Bars: {len(data)} ({len(data) // 78} trading days)")
    print(f"  Initial Capital: ${float(capital):,.2f}")
    print(f"  Final Capital:   ${float(result.final_capital):,.2f}")
    print(f"  Runtime: {elapsed:.2f}s")
    print()

    print("-" * 60)
    print("SUMMARY STATISTICS")
    print("-" * 60)
    print(f"  Total Trades:         {total_trades}")
    print(f"  Wins:                 {wins}")
    print(f"  Losses:               {losses}")
    print(f"  Win Rate:             {win_rate:.1f}%")
    print(f"  Avg Win:              ${float(avg_win):,.2f}")
    print(f"  Avg Loss:             ${float(avg_loss):,.2f}")
    print(f"  Avg R:R:              {avg_rr:.2f}")
    print(f"  Total P/L:            ${float(total_pnl):,.2f}")
    print(f"  Return:               {float(total_pnl / capital * Decimal('100')):.2f}%")
    print(f"  Profit Factor:        {float(pf):.2f}")
    print(f"  Max Drawdown:         {float(max_dd):.2f}%")
    print(f"  Sharpe Ratio:         {float(sharpe):.2f}")
    print(f"  Sortino Ratio:        {float(sortino):.2f}")
    print(f"  Calmar Ratio:         {float(calmar):.2f}")
    print(f"  Max Consec Wins:      {consec_wins}")
    print(f"  Max Consec Losses:    {consec_losses}")
    print(f"  Avg Duration (bars):  {avg_duration:.1f}")
    print(f"  Score V2:             {float(v2_score):.2f}")
    print()

    # ---- Trade Log ----
    if total_trades > 0:
        print("-" * 120)
        print("TRADE LOG")
        print("-" * 120)
        print(
            f"{'#':>4}  {'Date':>12}  {'Dir':>5}  "
            f"{'Entry Time':>10}  {'Exit Time':>10}  "
            f"{'Entry $':>10}  {'Exit $':>10}  "
            f"{'Shares':>7}  {'P/L':>12}  {'Ret%':>8}"
        )
        print("-" * 120)
        for idx, t in enumerate(trades, 1):
            entry_date = t.entry_date[:10] if t.entry_date else "?"
            entry_time = t.entry_date[11:16] if t.entry_date and len(t.entry_date) > 11 else "?"
            exit_time = t.exit_date[11:16] if t.exit_date and len(t.exit_date) > 11 else "?"
            direction = "LONG" if t.direction == Direction.LONG else "SHORT"
            entry_px = float(t.entry_price)
            exit_px = float(t.exit_price) if t.exit_price else 0.0
            pnl = float(t.profit_loss)
            ret = float(t.return_pct)
            print(
                f"{idx:>4}  {entry_date:>12}  {direction:>5}  "
                f"{entry_time:>10}  {exit_time:>10}  "
                f"${entry_px:>9.2f}  ${exit_px:>9.2f}  "
                f"{t.shares:>7}  ${pnl:>11.2f}  {ret:>7.2f}%"
            )

    # ---- Markdown Table ----
    print()
    print("-" * 120)
    print("MARKDOWN TABLE")
    print("-" * 120)
    print()
    print("| # | Date | Direction | Entry Time | Exit Time | Entry Price | Exit Price | Shares | P/L | Return % |")
    print("|---|------|-----------|------------|-----------|-------------|------------|--------|-----|----------|")
    for idx, t in enumerate(trades, 1):
        entry_date = t.entry_date[:10] if t.entry_date else "?"
        entry_time = t.entry_date[11:16] if t.entry_date and len(t.entry_date) > 11 else "?"
        exit_time = t.exit_date[11:16] if t.exit_date and len(t.exit_date) > 11 else "?"
        direction = "LONG" if t.direction == Direction.LONG else "SHORT"
        entry_px = float(t.entry_price)
        exit_px = float(t.exit_price) if t.exit_price else 0.0
        pnl = float(t.profit_loss)
        ret = float(t.return_pct)
        print(
            f"| {idx} | {entry_date} | {direction} | {entry_time} | {exit_time} "
            f"| ${entry_px:.2f} | ${exit_px:.2f} | {t.shares} "
            f"| ${pnl:.2f} | {ret:.2f}% |"
        )

    # ---- Long/Short breakdown ----
    longs = [t for t in trades if t.direction == Direction.LONG]
    shorts = [t for t in trades if t.direction == Direction.SHORT]
    print()
    print("-" * 60)
    print("DIRECTION BREAKDOWN")
    print("-" * 60)
    if longs:
        long_wins = sum(1 for t in longs if t.is_win())
        long_pnl = sum(t.profit_loss for t in longs)
        print(f"  Longs:  {len(longs)} trades, {long_wins} wins ({long_wins / len(longs) * 100:.0f}%), P/L: ${float(long_pnl):,.2f}")
    if shorts:
        short_wins = sum(1 for t in shorts if t.is_win())
        short_pnl = sum(t.profit_loss for t in shorts)
        print(f"  Shorts: {len(shorts)} trades, {short_wins} wins ({short_wins / len(shorts) * 100:.0f}%), P/L: ${float(short_pnl):,.2f}")

    # ---- Exit Mode Breakdown ----
    if hasattr(result, 'trade_modes'):
        modes = getattr(result, 'trade_modes', [])
        if modes:
            from collections import Counter
            mode_counts = Counter(modes)
            print()
            print("-" * 60)
            print("ENTRY MODE DISTRIBUTION")
            print("-" * 60)
            for mode, count in mode_counts.most_common():
                print(f"  {mode}: {count} ({count / len(modes) * 100:.0f}%)")

def main() -> None:
    parser = argparse.ArgumentParser(description="DMI+VWAP Strategy Backtest")
    parser.add_argument(
        "--capital",
        type=float,
        default=25_000.0,
        help="Initial capital (default: $25,000)",
    )
    parser.add_argument(
        "--adx-threshold",
        type=float,
        default=25.0,
        help="ADX threshold for entries (default: 25)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=2.0,
        help="Reward/risk ratio (default: 2.0)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to CSV data file",
    )
    args = parser.parse_args()

    data_path = Path(args.data_file) if args.data_file else DATA_DIR / "spy_5m_bars.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    data = _load_data(data_path)
    print(f"Loaded {len(data)} bars ({len(data) // 78} trading days)")
    print()

    config = DmiVwapConfig(
        adx_threshold=Decimal(str(args.adx_threshold)),
        rr=Decimal(str(args.rr)),
    )

    _run_backtest(data, Decimal(str(args.capital)), config)

if __name__ == "__main__":
    main()
