"""Resumable Polygon options data fetcher with checkpoint support.

Downloads option contract metadata and daily bars month-by-month,
saving progress to a checkpoint file so interrupted fetches can resume
where they left off.

Usage::

    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient

    config = GMEOptionsConfig.from_env()
    client = PolygonOptionsClient(api_key=config.polygon_api_key)
    fetcher = OptionsDataFetcher(config, client)
    fetcher.run()
"""
from __future__ import annotations

import calendar
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from stockdownloader.data.market.polygon_options_client import (
        PolygonOptionsClient,
    )
    from stockdownloader.gme.options.config import GMEOptionsConfig

logger = logging.getLogger(__name__)


class FetchCheckpoint:
    """Tracks which months have been fully fetched for resumability.

    Persists state to a JSON file so that a long-running fetch can be
    interrupted and resumed without re-downloading completed months.

    Parameters
    ----------
    path:
        Path to the checkpoint JSON file.  Created on first :meth:`save`
        if it does not exist.
    """

    __slots__ = ("_path", "completed_months", "last_contract")

    def __init__(self, path: Path) -> None:
        self._path = path
        self.completed_months: set[str] = set()
        self.last_contract: str | None = None

        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.completed_months = set(data.get("completed_months", []))
            self.last_contract = data.get("last_contract")

    def mark_month_complete(self, month: str) -> None:
        """Record *month* (``YYYY-MM``) as fully fetched."""
        self.completed_months.add(month)

    def is_month_done(self, month: str) -> bool:
        """Return ``True`` if *month* has already been fetched."""
        return month in self.completed_months

    def save(self) -> None:
        """Persist checkpoint state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                {
                    "completed_months": sorted(self.completed_months),
                    "last_contract": self.last_contract,
                },
                f,
                indent=2,
            )
        logger.debug("Checkpoint saved: %s", self._path)


class OptionsDataFetcher:
    """Fetches options data from Polygon month-by-month with resume support.

    Parameters
    ----------
    config:
        Pipeline configuration (date range, data directory, rate limits).
    client:
        Polygon API client for fetching contracts and bars.
    """

    def __init__(
        self,
        config: GMEOptionsConfig,
        client: PolygonOptionsClient,
    ) -> None:
        self.config = config
        self.client = client
        self.checkpoint = FetchCheckpoint(
            config.data_dir / "checkpoint.json",
        )

    def _expiration_months(self) -> list[str]:
        """Return all ``YYYY-MM`` strings from start_date to end_date."""
        months: list[str] = []
        current_year = self.config.start_date.year
        current_month = self.config.start_date.month
        end_year = self.config.end_date.year
        end_month = self.config.end_date.month

        while (current_year, current_month) <= (end_year, end_month):
            months.append(f"{current_year:04d}-{current_month:02d}")
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1

        return months

    def _remaining_months(self) -> list[str]:
        """Return months that have not yet been completed."""
        return [
            m for m in self._expiration_months()
            if not self.checkpoint.is_month_done(m)
        ]

    @staticmethod
    def _month_date_range(month: str) -> tuple[date, date]:
        """Return (first_day, last_day) for a ``YYYY-MM`` month string."""
        year, mon = (int(x) for x in month.split("-"))
        first = date(year, mon, 1)
        last_day = calendar.monthrange(year, mon)[1]
        last = date(year, mon, last_day)
        return first, last

    def fetch_contracts_for_month(self, month: str) -> list[dict]:
        """Fetch all option contracts expiring in *month*.

        Delegates to the Polygon client's ``fetch_option_contracts`` method.
        """
        from_date, to_date = self._month_date_range(month)
        return self.client.fetch_option_contracts(
            self.config.symbol,
            from_date,
            to_date,
        )

    def _fetch_contract_bars(self, contract: dict) -> list[dict]:
        """Fetch all daily bars for a contract across its tradable lifetime.

        Returns a list of bar dicts enriched with contract metadata,
        one per trading day.  Column names match the schema expected by
        :class:`~stockdownloader.gme.options.state_engine.OptionsStateEngine`:

        ``date``, ``option_ticker``, ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``vwap``, ``strike``, ``option_type``, ``expiration``,
        ``open_interest``.
        """
        ticker = contract["ticker"]
        expiration_date = date.fromisoformat(contract["expiration_date"])
        strike_price = contract.get("strike_price")
        contract_type = contract.get("contract_type")

        # Fetch bars from config start to expiration (or end_date)
        from_date = self.config.start_date
        to_date = min(expiration_date, self.config.end_date)

        if from_date > to_date:
            self.checkpoint.last_contract = ticker
            return []

        if self.config.rate_limit_delay > 0:
            time.sleep(self.config.rate_limit_delay)

        bars = self.client.fetch_option_daily_bars_range(
            ticker, from_date, to_date,
        )

        if not bars:
            self.checkpoint.last_contract = ticker
            return []

        self.checkpoint.last_contract = ticker
        result: list[dict] = []
        for bar in bars:
            # Polygon timestamps are Unix ms → convert to YYYY-MM-DD
            bar_ts = bar.get("t", 0)
            bar_date = date.fromtimestamp(bar_ts / 1000).isoformat()
            result.append(
                {
                    "date": bar_date,
                    "option_ticker": ticker,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v", 0),
                    "vwap": bar.get("vw"),
                    "strike": strike_price,
                    "option_type": contract_type,
                    "expiration": contract["expiration_date"],
                    "open_interest": 0,  # Not in Polygon aggs API
                },
            )
        return result

    def fetch_bars_for_month(self, month: str) -> pd.DataFrame:
        """Fetch daily bars for all contracts expiring in *month*.

        Returns a DataFrame with one row per contract-day observation.
        """
        import pandas as pd

        contracts = self.fetch_contracts_for_month(month)
        all_bars: list[dict] = []

        for contract in contracts:
            bars = self._fetch_contract_bars(contract)
            all_bars.extend(bars)

        return pd.DataFrame(all_bars)

    def save_month(self, month: str, df: pd.DataFrame) -> Path:
        """Save a month's data as a Parquet file.

        Returns the path to the written file.
        """
        out_dir = self.config.data_dir / "monthly"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{month}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)
        return path

    @staticmethod
    def load_month(path: Path) -> pd.DataFrame:
        """Load a month's Parquet file into a DataFrame."""
        import pandas as pd

        return pd.read_parquet(path)

    def run(self) -> None:
        """Main entry point: fetch all remaining months, save, checkpoint."""
        remaining = self._remaining_months()
        total = len(remaining)
        logger.info("Fetching %d remaining months", total)

        for i, month in enumerate(remaining, 1):
            logger.info("Fetching month %d/%d: %s", i, total, month)
            df = self.fetch_bars_for_month(month)

            if not df.empty:
                self.save_month(month, df)

            self.checkpoint.mark_month_complete(month)
            self.checkpoint.last_contract = None
            self.checkpoint.save()
            logger.info("Completed month %s (%d rows)", month, len(df))

        logger.info("All months complete")
