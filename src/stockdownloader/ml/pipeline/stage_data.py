"""Stage 1: Data Acquisition.

Fetches historical price data from a CSV file, JSON cache, or Yahoo
Finance (fallback) and optionally loads alternative data (FTD, short
interest, dark pool, ownership, borrow rates) into an
:class:`AlternativeDataStore`.
"""

from __future__ import annotations

import csv
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Callable

from stockdownloader.ml.pipeline.config import AltDataConfig, DataConfig
from stockdownloader.ml.pipeline.results import DataResult
from stockdownloader.core.models.price import PriceData

logger = logging.getLogger(__name__)


class DataStage:
    """Fetch and optionally cache historical price data.

    Parameters
    ----------
    config:
        Data acquisition configuration.
    print_fn:
        Callable for progress output (default: :func:`print`).
    """

    def __init__(
        self,
        config: DataConfig,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self._cfg = config
        self._out = print_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> DataResult:
        """Fetch data, cache if enabled, return :class:`DataResult`.

        Priority: CSV file → JSON cache → Yahoo Finance.
        """
        # 1. CSV file is the authoritative source when configured
        if self._cfg.csv_file:
            csv_path = Path(self._cfg.csv_file)
            if csv_path.exists():
                data = self._load_csv(csv_path)
                if data:
                    self._out(
                        f"Loaded {len(data)} bars from CSV: {csv_path} "
                        f"({data[0].date} to {data[-1].date})"
                    )
                    alt_store = (
                        self._fetch_alt_data(data) if self._cfg.alt_data else None
                    )
                    return DataResult(
                        symbol=self._cfg.symbol,
                        data=data,
                        date_range=(data[0].date, data[-1].date),
                        bar_count=len(data),
                        cached=True,
                        alt_data_store=alt_store,
                    )
            else:
                logger.warning("CSV file not found: %s — falling back", csv_path)

        # 2. Try JSON cache
        cache_path = self._cache_path()
        if self._cfg.use_cache and cache_path.exists():
            data = self._load_cache(cache_path)
            if data:
                self._out(
                    f"Loaded {len(data)} bars from cache: {cache_path}"
                )
                alt_store = self._fetch_alt_data(data) if self._cfg.alt_data else None
                return DataResult(
                    symbol=self._cfg.symbol,
                    data=data,
                    date_range=(data[0].date, data[-1].date),
                    bar_count=len(data),
                    cached=True,
                    alt_data_store=alt_store,
                )

        # 3. Fetch from Yahoo Finance
        from stockdownloader.app.app_helpers import fetch_daily_data

        data = fetch_daily_data(
            self._cfg.symbol,
            period=self._cfg.range_,
            interval=self._cfg.interval,
        )

        if not data:
            raise RuntimeError(
                f"Could not fetch data for {self._cfg.symbol}"
            )

        # Save to cache
        if self._cfg.use_cache:
            self._save_cache(data, cache_path)
            self._out(f"Cached {len(data)} bars to {cache_path}")

        self._out(
            f"Fetched {len(data)} bars "
            f"({data[0].date} to {data[-1].date})"
        )

        # Fetch alternative data if configured
        alt_store = self._fetch_alt_data(data) if self._cfg.alt_data else None

        return DataResult(
            symbol=self._cfg.symbol,
            data=data,
            date_range=(data[0].date, data[-1].date),
            bar_count=len(data),
            alt_data_store=alt_store,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path) -> list[PriceData]:
        """Load price data from a CSV file.

        Expected columns: Date, Open, High, Low, Close, Adj Close, Volume
        """
        try:
            data: list[PriceData] = []
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    data.append(
                        PriceData(
                            date=row["Date"],
                            open=Decimal(row["Open"]),
                            high=Decimal(row["High"]),
                            low=Decimal(row["Low"]),
                            close=Decimal(row["Close"]),
                            adj_close=Decimal(row["Adj Close"]),
                            volume=int(float(row["Volume"])),
                        )
                    )
            return data
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("CSV parse error at %s: %s", path, exc)
            return []

    def _cache_path(self) -> Path:
        return (
            Path(self._cfg.cache_dir)
            / f"{self._cfg.symbol}_{self._cfg.range_}_{self._cfg.interval}.json"
        )

    @staticmethod
    def _load_cache(path: Path) -> list[PriceData]:
        """Deserialise cached price data from JSON."""
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
            logger.warning("Cache corrupt at %s — refetching.", path)
            return []

    @staticmethod
    def _save_cache(data: list[PriceData], path: Path) -> None:
        """Serialise price data to JSON for caching."""
        path.parent.mkdir(parents=True, exist_ok=True)
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

    # ------------------------------------------------------------------
    # Alternative data
    # ------------------------------------------------------------------

    def _fetch_alt_data(self, data: list[PriceData]) -> object | None:
        """Fetch all configured alternative data sources.

        Each source is wrapped in ``try/except`` for graceful degradation
        — if one source fails, the others still load.

        Returns an :class:`AlternativeDataStore` or ``None`` on total failure.
        """
        from stockdownloader.ml.alternative_data_store import AlternativeDataStore

        alt_cfg: AltDataConfig = self._cfg.alt_data  # type: ignore[assignment]
        symbol = self._cfg.symbol
        price_dates = [d.date[:10] for d in data]

        ftd_records = None
        si_records = None
        dp_records = None
        ownership_records = None
        borrow_records = None

        # --- FTD ---
        if alt_cfg.enable_ftd:
            try:
                from stockdownloader.data.sec.ftd_client import SecFtdClient

                client = SecFtdClient(
                    cache_dir=alt_cfg.ftd_cache_dir,
                    user_agent=alt_cfg.user_agent,
                )
                ftd_records = client.fetch_ftd_data(
                    symbol, start_year=alt_cfg.ftd_start_year,
                )
                self._out(
                    f"  FTD: {len(ftd_records)} records loaded"
                )
            except Exception as exc:
                logger.warning("FTD fetch failed for %s: %s", symbol, exc)
                self._out(f"  FTD: FAILED ({exc})")

        # --- Short interest ---
        if alt_cfg.enable_short_interest:
            try:
                from stockdownloader.data.finra.short_interest_client import (
                    FinraShortInterestClient,
                )

                client = FinraShortInterestClient(
                    client_id=alt_cfg.finra_client_id or None,
                    client_secret=alt_cfg.finra_client_secret or None,
                    data_dir=alt_cfg.data_dir,
                )
                si_records = client.fetch_short_interest(symbol)
                self._out(
                    f"  Short Interest: {len(si_records)} records loaded"
                )
            except Exception as exc:
                logger.warning("SI fetch failed for %s: %s", symbol, exc)
                self._out(f"  Short Interest: FAILED ({exc})")

        # --- Dark pool ---
        if alt_cfg.enable_dark_pool:
            try:
                from stockdownloader.data.finra.dark_pool_client import (
                    FinraDarkPoolClient,
                )

                client = FinraDarkPoolClient(
                    client_id=alt_cfg.finra_client_id or None,
                    client_secret=alt_cfg.finra_client_secret or None,
                    data_dir=alt_cfg.data_dir,
                )
                dp_records = client.fetch_dark_pool_volume(symbol)
                self._out(
                    f"  Dark Pool: {len(dp_records)} records loaded"
                )
            except Exception as exc:
                logger.warning("Dark pool fetch failed for %s: %s", symbol, exc)
                self._out(f"  Dark Pool: FAILED ({exc})")

        # --- Institutional ownership ---
        if alt_cfg.enable_ownership:
            try:
                from stockdownloader.data.sec.ownership_client import (
                    SecOwnershipClient,
                )

                client = SecOwnershipClient(
                    data_dir=alt_cfg.data_dir,
                    user_agent=alt_cfg.user_agent,
                )
                ownership_records = client.fetch_ownership_snapshots(symbol)
                self._out(
                    f"  Ownership: {len(ownership_records)} snapshots loaded"
                )
            except Exception as exc:
                logger.warning("Ownership fetch failed for %s: %s", symbol, exc)
                self._out(f"  Ownership: FAILED ({exc})")

        # --- Borrow rate ---
        if alt_cfg.enable_borrow_rate and si_records:
            try:
                from stockdownloader.data.market.borrow_rate import (
                    BorrowRateProxy,
                )

                proxy = BorrowRateProxy()
                borrow_records = proxy.estimate_borrow_rates(
                    symbol, si_records,
                )
                self._out(
                    f"  Borrow Rate: {len(borrow_records)} estimates computed"
                )
            except Exception as exc:
                logger.warning("Borrow rate failed for %s: %s", symbol, exc)
                self._out(f"  Borrow Rate: FAILED ({exc})")

        # Build the store
        store = AlternativeDataStore()
        store.load(
            symbol=symbol,
            ftd_records=ftd_records,
            si_records=si_records,
            dp_records=dp_records,
            ownership=ownership_records,
            borrow_rates=borrow_records,
            price_dates=price_dates,
        )

        if store.record_count == 0:
            self._out("  No alternative data loaded.")
            return None

        self._out(
            f"  Alternative data store: {store.record_count} dates aligned"
        )
        return store
