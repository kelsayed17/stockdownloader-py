"""Accumulates intraday 5-minute bar data over time by merging fresh
fetches with an existing CSV file on disk.

Supports both Yahoo Finance and Polygon.io as data sources.  Yahoo is
limited to ~60 days of intraday history; Polygon's free tier provides
2 years of minute-level data.

Typical usage::

    from stockdownloader.data.intraday_data_accumulator import IntradayDataAccumulator
    from stockdownloader.data.polygon_data_client import PolygonDataClient

    client = PolygonDataClient(api_key="YOUR_KEY")
    acc = IntradayDataAccumulator(client=client, fetch_days=730)
    bars = acc.accumulate("SPY", "data/spy/5m_bars.csv")
    print(f"Total bars on disk: {len(bars)}")

Or from the CLI::

    intraday-accumulate                              # Yahoo, SPY, 60 days
    intraday-accumulate --source polygon --days 730  # Polygon, SPY, 2 years
"""
from __future__ import annotations

import logging
from operator import attrgetter
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from stockdownloader.data.intraday_csv import IntradayCsvLoader, normalize_tz, write_to_file
from stockdownloader.model.intraday_price_data import IntradayPriceData

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("data")


@runtime_checkable
class IntradayClient(Protocol):
    """Protocol for any client that can fetch intraday bar history."""

    def fetch_intraday_history(
        self,
        symbol: str,
        total_days: int = ...,
        interval: str = ...,
        **kwargs: Any,
    ) -> list[IntradayPriceData]: ...


def default_csv_path(symbol: str) -> Path:
    """Return the default CSV path for *symbol*."""
    return _DEFAULT_DIR / symbol.lower() / "5m_bars.csv"


class IntradayDataAccumulator:
    """Fetches the latest intraday bars from a data source and merges them
    with bars already stored in a local CSV file.

    The merge is deduplicated by datetime string so overlapping windows
    are handled gracefully.

    Parameters
    ----------
    client:
        Any object with a ``fetch_intraday_history()`` method.  Defaults
        to :class:`~stockdownloader.data.yahoo_data_client.YahooDataClient`.
    fetch_days:
        Number of calendar days to fetch (default: 60).
    interval:
        Bar interval string (default: ``"5m"``).
    """

    def __init__(
        self,
        client: IntradayClient | None = None,
        fetch_days: int = 60,
        interval: str = "5m",
    ) -> None:
        if client is None:
            from stockdownloader.data.yahoo_data_client import YahooDataClient
            client = YahooDataClient()
        self._client = client
        self._fetch_days = fetch_days
        self._interval = interval

    def accumulate(
        self,
        symbol: str,
        csv_path: str | Path | None = None,
    ) -> list[IntradayPriceData]:
        """Fetch latest bars, merge with existing CSV, write back, return all.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"SPY"``).
        csv_path:
            Path to the CSV file.  If ``None``, uses
            ``data/<symbol>_5m_bars.csv``.

        Returns
        -------
        list[IntradayPriceData]
            The full merged dataset sorted by datetime ascending.
        """
        csv_path = Path(csv_path) if csv_path else default_csv_path(symbol)

        # Load existing bars from disk
        existing: list[IntradayPriceData] = []
        if csv_path.exists():
            existing = IntradayCsvLoader.load_from_file(csv_path)
            logger.info(
                "Loaded %d existing bars from %s", len(existing), csv_path
            )

        # Fetch fresh bars from data source
        source_name = type(self._client).__name__
        logger.info(
            "Fetching %s %s bars (%d days) from %s...",
            symbol,
            self._interval,
            self._fetch_days,
            source_name,
        )
        fresh = self._client.fetch_intraday_history(
            symbol,
            total_days=self._fetch_days,
            interval=self._interval,
        )
        logger.info("Fetched %d fresh bars", len(fresh))

        # Merge and deduplicate
        merged = _merge_bars(existing, fresh)
        logger.info(
            "Merged: %d existing + %d fresh = %d total (after dedup)",
            len(existing),
            len(fresh),
            len(merged),
        )

        # Write back
        write_to_file(merged, csv_path)

        return merged


def _normalize_bar(bar: IntradayPriceData) -> IntradayPriceData:
    """Return a copy of *bar* with a normalised datetime string."""
    normed = normalize_tz(bar.date)
    if normed == bar.date:
        return bar
    return IntradayPriceData(
        date=normed,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        adj_close=bar.adj_close,
        volume=bar.volume,
    )


def _merge_bars(
    existing: list[IntradayPriceData],
    fresh: list[IntradayPriceData],
) -> list[IntradayPriceData]:
    """Merge two bar lists, deduplicating by datetime string.

    When the same datetime appears in both lists, the *fresh* bar wins
    (it may have corrected data from Yahoo).  Timezone offsets are
    normalised (``-0500`` → ``-05:00``) before comparison so bars from
    different sources match correctly.
    """
    by_date: dict[str, IntradayPriceData] = {}

    for bar in existing:
        normed = _normalize_bar(bar)
        by_date[normed.date] = normed

    # Fresh bars overwrite existing bars for the same datetime
    for bar in fresh:
        normed = _normalize_bar(bar)
        by_date[normed.date] = normed

    merged = sorted(by_date.values(), key=attrgetter("date"))
    return merged
