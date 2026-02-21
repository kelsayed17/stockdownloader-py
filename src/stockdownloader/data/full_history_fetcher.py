"""Fetches complete price history across multiple timeframes.

Uses **Polygon.io** for 5-minute bars (2-year free tier) and
**Yahoo Finance** for full daily history (back to IPO).  Any
intermediate timeframe (15m, 30m, 1h, 4h) is resampled from 5-min
bars.  Results are persisted as CSV.

Usage::

    fetcher = FullHistoryFetcher()
    daily = fetcher.fetch_full_daily_history("GME")   # → data/GME/daily_bars.csv
    intraday = fetcher.fetch_intraday_history("GME")   # → data/GME/5m_bars.csv
    all_tf = fetcher.fetch_all_timeframes("GME")
"""
from __future__ import annotations

import csv
import json
import logging
import os
from decimal import Decimal
from pathlib import Path

from stockdownloader.data.intraday_csv import (
    IntradayCsvLoader,
    write_to_file,
)
from stockdownloader.data.intraday_data_accumulator import _merge_bars
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.price_data import PriceData

logger = logging.getLogger(__name__)

_POLYGON_INTRADAY_DAYS = 730

_DAILY_CSV_HEADER = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


class FullHistoryFetcher:
    """Fetches complete price history using the best source per timeframe.

    * **Daily bars**: Yahoo Finance (full history back to IPO).
    * **5-min bars**: Polygon.io (2-year free tier).
    * **15m / 30m / 1h / 4h**: Resampled from 5-min bars.

    Parameters
    ----------
    cache_dir:
        Directory for daily JSON caches (default: ``data/cache``).
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_full_daily_history(self, symbol: str) -> list[PriceData]:
        """Fetch entire daily history, preferring Polygon.io.

        **Primary source**: Polygon.io (``fetch_daily_history``) — provides
        complete adjusted daily bars going back to IPO.

        **Fallback**: Yahoo Finance with epoch timestamps (``period1=0``).

        Results are cached both as JSON (``data/cache/{SYMBOL}_max_1d.json``)
        and as CSV (``data/{SYMBOL}/daily_bars.csv``).

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).

        Returns
        -------
        list[PriceData]
            Full daily history sorted by date ascending, or empty list
            on failure.
        """
        # Check CSV cache first
        sym_dir = Path("data") / symbol.upper()
        sym_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sym_dir / "daily_bars.csv"

        # Legacy migration from old flat path
        legacy_csv = Path("data") / f"{symbol.lower()}_daily_bars.csv"
        if not csv_path.exists() and legacy_csv.exists():
            legacy_csv.rename(csv_path)
            logger.info("Migrated %s → %s", legacy_csv, csv_path)

        cached = self._load_daily_csv(csv_path)
        if cached:
            logger.info(
                "Loaded %d daily bars from CSV cache for %s",
                len(cached), symbol,
            )
            return cached

        # Check JSON cache
        json_cached = self._load_daily_cache(symbol)
        if json_cached:
            logger.info(
                "Loaded %d daily bars from JSON cache for %s",
                len(json_cached), symbol,
            )
            self._save_daily_csv(json_cached, csv_path)
            return json_cached

        data: list[PriceData] = []

        # Yahoo Finance provides full daily history back to IPO via
        # epoch-based period1=0.  Polygon free tier only provides 2
        # years of daily data, so Yahoo is the better choice for daily
        # bars.  (5-min bars use Polygon exclusively.)
        try:
            from stockdownloader.data.yahoo_data_client import YahooDataClient

            client = YahooDataClient()
            data = client.fetch_price_data(
                symbol, range_="max", interval="1d",
            )
            if data:
                logger.info(
                    "Yahoo: fetched %d daily bars for %s (%s to %s)",
                    len(data), symbol, data[0].date, data[-1].date,
                )
        except Exception as exc:
            logger.warning(
                "Yahoo daily fetch failed for %s: %s", symbol, exc,
            )

        if not data:
            logger.warning("No daily data available for %s", symbol)
            return []

        # Warn if the result seems suspiciously small
        if len(data) < 500:
            logger.warning(
                "Full daily history for %s returned only %d bars "
                "(range: %s to %s) — this may indicate truncated data.",
                symbol, len(data), data[0].date, data[-1].date,
            )

        # Persist to both JSON cache and CSV
        self._save_daily_cache(symbol, data)
        self._save_daily_csv(data, csv_path)

        logger.info(
            "Fetched %d daily bars for %s (%s to %s)",
            len(data), symbol, data[0].date, data[-1].date,
        )
        return data

    def fetch_intraday_history(
        self, symbol: str, days: int = _POLYGON_INTRADAY_DAYS
    ) -> list[IntradayPriceData]:
        """Fetch 5-min data exclusively from Polygon.io.

        Polygon provides 2 years of 5-minute bars on the free tier.
        Any timeframe (15m, 30m, 1h, 4h, daily) can be resampled
        from these 5-min bars.  Caches to ``data/{SYMBOL}/5m_bars.csv``.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        days:
            Total calendar days of history to request (default: 730).

        Returns
        -------
        list[IntradayPriceData]
            5-minute bars sorted by datetime ascending.
        """
        sym_dir = Path("data") / symbol.upper()
        sym_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sym_dir / "5m_bars.csv"

        # Legacy migration from old flat path
        legacy_csv = Path("data") / f"{symbol.lower()}_5m_bars.csv"
        if not csv_path.exists() and legacy_csv.exists():
            legacy_csv.rename(csv_path)
            logger.info("Migrated %s → %s", legacy_csv, csv_path)

        # Load existing bars from disk
        existing: list[IntradayPriceData] = []
        if csv_path.exists():
            existing = IntradayCsvLoader.load_from_file(csv_path)
            logger.info(
                "Loaded %d existing 5m bars from %s", len(existing), csv_path
            )

        # Fetch from Polygon (2-year history)
        polygon_key = os.environ.get("POLYGON_API_KEY", "")
        if not polygon_key:
            logger.warning(
                "POLYGON_API_KEY not set — cannot fetch 5-min bars"
            )
            return existing

        polygon_bars: list[IntradayPriceData] = []
        try:
            from stockdownloader.data.polygon_data_client import (
                PolygonDataClient,
            )

            polygon_client = PolygonDataClient(api_key=polygon_key)
            polygon_bars = polygon_client.fetch_intraday_history(
                symbol, total_days=days
            )
            logger.info(
                "Fetched %d 5m bars from Polygon for %s",
                len(polygon_bars), symbol,
            )
        except Exception as exc:
            logger.warning(
                "Polygon fetch failed for %s: %s", symbol, exc,
            )

        # Merge with existing cache (Polygon wins on conflicts)
        merged = _merge_bars(existing, polygon_bars)

        logger.info(
            "Intraday data for %s: %d existing + %d polygon = %d total",
            symbol, len(existing), len(polygon_bars), len(merged),
        )

        # Write back to CSV
        if merged:
            write_to_file(merged, csv_path)

        return merged

    def fetch_all_timeframes(self, symbol: str) -> dict[str, list]:
        """Fetch daily + 5m, then resample to 15m/30m/1h/4h.

        Returns a dict with keys: ``"1d"``, ``"5m"``, ``"15m"``,
        ``"30m"``, ``"1h"``, ``"4h"``.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"SPY"``).

        Returns
        -------
        dict[str, list]
            Keys are timeframe labels; values are bar lists.
            ``"1d"`` contains :class:`PriceData`, all others contain
            :class:`IntradayPriceData`.
        """
        daily = self.fetch_full_daily_history(symbol)
        bars_5m = self.fetch_intraday_history(symbol)

        result: dict[str, list] = {
            "1d": daily,
            "5m": bars_5m,
            "15m": self._resample_bars(bars_5m, 3),
            "30m": self._resample_bars(bars_5m, 6),
            "1h": self._resample_bars(bars_5m, 12),
            "4h": self._resample_bars(bars_5m, 48),
        }

        for label, bars in result.items():
            logger.info(
                "%s %s: %d bars", symbol, label, len(bars)
            )

        return result

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_bars(
        bars_5m: list[IntradayPriceData], multiplier: int
    ) -> list[IntradayPriceData]:
        """Resample 5m bars to a higher timeframe by grouping.

        Groups consecutive 5m bars within the same trading session into
        *multiplier*-bar candles.  Only complete groups are included
        (partial candles at session boundaries are excluded).

        Parameters
        ----------
        bars_5m:
            Source 5-minute bars sorted chronologically.
        multiplier:
            Number of 5m bars per output candle (3 for 15m, 6 for 30m,
            12 for 1h, 48 for 4h).

        Returns
        -------
        list[IntradayPriceData]
            Resampled bars sorted chronologically.
        """
        if not bars_5m or multiplier <= 1:
            return list(bars_5m)

        # Group by trading session (date portion)
        sessions: dict[str, list[IntradayPriceData]] = {}
        for bar in bars_5m:
            trading_date = bar.date[:10]
            sessions.setdefault(trading_date, []).append(bar)

        result: list[IntradayPriceData] = []

        for _date in sorted(sessions):
            session_bars = sessions[_date]
            complete_groups = len(session_bars) // multiplier

            for g in range(complete_groups):
                base = g * multiplier
                group = session_bars[base : base + multiplier]

                o = group[0].open
                h = max(b.high for b in group)
                lo = min(b.low for b in group)
                c = group[-1].close
                vol = sum(b.volume for b in group)

                result.append(
                    IntradayPriceData(
                        date=group[0].date,
                        open=o,
                        high=h,
                        low=lo,
                        close=c,
                        adj_close=c,
                        volume=vol,
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Daily cache helpers
    # ------------------------------------------------------------------

    def _load_daily_cache(self, symbol: str) -> list[PriceData] | None:
        """Load cached daily data from JSON, or return ``None``."""
        path = self._cache_dir / f"{symbol.upper()}_max_1d.json"
        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text())
            return [
                PriceData(
                    date=r["date"],
                    open=Decimal(r["open"]),
                    high=Decimal(r["high"]),
                    low=Decimal(r["low"]),
                    close=Decimal(r["close"]),
                    adj_close=Decimal(r["adj_close"]),
                    volume=r["volume"],
                )
                for r in raw
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Daily cache corrupt at %s — refetching.", path
            )
            return None

    def _save_daily_cache(
        self, symbol: str, data: list[PriceData]
    ) -> None:
        """Serialise daily price data to JSON for caching."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{symbol.upper()}_max_1d.json"

        rows = [
            {
                "date": d.date,
                "open": str(d.open),
                "high": str(d.high),
                "low": str(d.low),
                "close": str(d.close),
                "adj_close": str(d.adj_close),
                "volume": d.volume,
            }
            for d in data
        ]
        path.write_text(json.dumps(rows))
        logger.info("Cached %d daily bars to %s", len(data), path)

    # ------------------------------------------------------------------
    # Daily CSV I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_daily_csv(path: Path) -> list[PriceData] | None:
        """Load daily bars from a CSV file, or return ``None``."""
        if not path.exists():
            return None

        try:
            bars: list[PriceData] = []
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) < 7:
                        continue
                    bars.append(
                        PriceData(
                            date=row[0].strip(),
                            open=Decimal(row[1]),
                            high=Decimal(row[2]),
                            low=Decimal(row[3]),
                            close=Decimal(row[4]),
                            adj_close=Decimal(row[5]),
                            volume=int(Decimal(row[6]).to_integral_value()),
                        )
                    )
            return bars if bars else None
        except (csv.Error, ValueError, OSError) as exc:
            logger.warning("Failed to load daily CSV %s: %s", path, exc)
            return None

    @staticmethod
    def _save_daily_csv(data: list[PriceData], path: Path) -> None:
        """Write daily bars to CSV format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_DAILY_CSV_HEADER)
            for d in data:
                writer.writerow([
                    d.date,
                    d.open,
                    d.high,
                    d.low,
                    d.close,
                    d.adj_close,
                    d.volume,
                ])
        logger.info("Wrote %d daily bars to %s", len(data), path)
