"""Pipeline helper utilities — output, data loading."""
from __future__ import annotations

from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.util.io import TeeWriter


def make_print_fn(tee: TeeWriter | None):
    """Return a print function that writes to TeeWriter or stdout."""
    def _print(msg: str = "") -> None:
        if tee:
            tee.write(msg + "\n")
        else:
            print(msg)
    return _print


def load_intraday_data(csv_path: str, out) -> list[IntradayPriceData]:
    """Load 5-min bars from CSV."""
    out(f"Loading intraday data from {csv_path}...")
    data = IntradayCsvLoader.load_from_file(csv_path)
    if not data:
        out(f"ERROR: No data loaded from {csv_path}")
        return []
    trading_days = len({bar.date[:10] for bar in data})
    out(f"Loaded {len(data):,} bars across {trading_days} trading days")
    out(f"Date range: {data[0].date[:10]} to {data[-1].date[:10]}")
    return data


def load_daily_data(intraday_data: list[IntradayPriceData], out):
    """Aggregate 5-min bars to daily PriceData for options backtesting."""
    from stockdownloader.util.timeframe import TimeframeAggregator, Timeframe
    out("Aggregating 5-min bars to daily for options strategies...")
    agg = TimeframeAggregator(intraday_data)
    daily = agg.as_price_data(Timeframe.DAILY)
    out(f"  {len(daily)} daily bars")
    return daily


def unique_days(data: list[IntradayPriceData]) -> int:
    """Count unique trading days in intraday data."""
    return len({d.date[:10] for d in data})
