"""Prepare MarketContext CSV from Polygon VIX + SPY daily data.

Usage:
    PYTHONPATH=src python3 scripts/prepare_market_context.py

Outputs:
    data/SPY/market_context.csv
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.strategies.intraday.market_context import vix_regime_from_level

# -- FOMC dates (press conference days) ------------------------------------
# Source: Federal Reserve website
# Format: YYYY-MM-DD for press conference days (every other meeting)
FOMC_PRESS_CONF_DATES = {
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
}

FOMC_ALL_DATES = FOMC_PRESS_CONF_DATES | {
    # Non-press-conference FOMC meetings
    # 2024
    "2024-03-19", "2024-04-30", "2024-06-11", "2024-07-30",
    "2024-09-17", "2024-11-06", "2024-12-17",
    # 2025
    "2025-01-28", "2025-03-18", "2025-05-06", "2025-06-17",
    "2025-07-29", "2025-09-16", "2025-10-28", "2025-12-16",
}


def _is_opex(date_str: str) -> bool:
    """Third Friday of the month = monthly options expiration."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() != 4:  # Not Friday
        return False
    day = dt.day
    return 15 <= day <= 21


def _compute_rsi(closes: list[Decimal], period: int = 2) -> Decimal:
    """Compute RSI from a list of daily closes."""
    if len(closes) < period + 1:
        return Decimal("50")  # Neutral default
    gains = []
    losses = []
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    return Decimal("100") - Decimal("100") / (1 + rs)


def _compute_sma(values: list[Decimal], period: int) -> Decimal:
    """Compute simple moving average."""
    if len(values) < period:
        return values[-1] if values else Decimal("0")
    return sum(values[-period:]) / period


def main() -> None:
    """Build market_context.csv from local data files."""
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Load SPY daily bars (aggregated from 5m or fetched separately)
    spy_5m_path = data_dir / "SPY" / "5m_bars.csv"
    if not spy_5m_path.exists():
        print(f"SPY 5m data not found at {spy_5m_path}")
        sys.exit(1)

    # Aggregate 5m bars to daily
    print("Aggregating SPY 5m bars to daily...")
    daily_bars: dict[str, dict] = {}
    with open(spy_5m_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Datetime"][:10]
            o = Decimal(row["Open"])
            h = Decimal(row["High"])
            low = Decimal(row["Low"])
            c = Decimal(row["Close"])
            v = int(row["Volume"])
            if date not in daily_bars:
                daily_bars[date] = {
                    "open": o, "high": h, "low": low,
                    "close": c, "volume": v,
                }
            else:
                d = daily_bars[date]
                d["high"] = max(d["high"], h)
                d["low"] = min(d["low"], low)
                d["close"] = c
                d["volume"] += v

    dates = sorted(daily_bars.keys())
    closes = [daily_bars[d]["close"] for d in dates]

    # Try loading VIX data -- fall back to synthetic if not available
    vix_path = data_dir / "VIX" / "daily.csv"
    vix_data: dict[str, Decimal] = {}
    if vix_path.exists():
        with open(vix_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vix_data[row["Date"][:10]] = Decimal(row["Close"])
        print(f"Loaded {len(vix_data)} VIX daily records")
    else:
        print("VIX daily data not found -- using synthetic VIX proxy (ATR-based)")
        # Synthetic VIX proxy from 20-day realized volatility
        for i, date in enumerate(dates):
            if i < 20:
                vix_data[date] = Decimal("18")  # Default
            else:
                returns = []
                for j in range(i - 20, i):
                    if closes[j - 1] > 0:
                        ret = abs((closes[j] - closes[j - 1]) / closes[j - 1])
                        returns.append(ret)
                avg_ret = (
                    sum(returns) / len(returns)
                    if returns
                    else Decimal("0.01")
                )
                vix_data[date] = avg_ret * Decimal("1590")  # Annualized approx

    # Build context CSV
    out_path = data_dir / "SPY" / "market_context.csv"
    vix_closes: list[Decimal] = []
    print(f"Writing market context for {len(dates)} dates...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "vix_close", "vix_sma20", "is_fomc",
            "is_fomc_pc", "is_opex", "rsi2", "above_sma200",
        ])
        for i, date in enumerate(dates):
            vix = vix_data.get(date, Decimal("18"))
            vix_closes.append(vix)
            vix_sma20 = _compute_sma(vix_closes, 20)
            rsi2 = _compute_rsi(closes[: i + 1], period=2)
            sma200 = _compute_sma(closes[: i + 1], 200)
            above_sma200 = closes[i] > sma200

            writer.writerow([
                date,
                f"{vix:.2f}",
                f"{vix_sma20:.2f}",
                "1" if date in FOMC_ALL_DATES else "0",
                "1" if date in FOMC_PRESS_CONF_DATES else "0",
                "1" if _is_opex(date) else "0",
                f"{rsi2:.2f}",
                "1" if above_sma200 else "0",
            ])

    print(f"Wrote {out_path} ({len(dates)} rows)")


if __name__ == "__main__":
    main()
