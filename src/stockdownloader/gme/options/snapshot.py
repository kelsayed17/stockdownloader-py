"""Snapshot-based options chain collector with OI, greeks, and IV.

Fetches the current-day options chain snapshot from Polygon's bulk
endpoint and saves per-contract data as daily Parquet files.

Usage::

    from stockdownloader.gme.options.snapshot import SnapshotCollector

    collector = SnapshotCollector(config, client)
    df = collector.run()  # collect + save
"""
from __future__ import annotations

import logging
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


class SnapshotCollector:
    """Fetches and stores daily options chain snapshots.

    Parameters
    ----------
    config:
        Pipeline configuration (data directory).
    client:
        Polygon API client with ``fetch_options_chain_snapshot``.
    """

    def __init__(
        self,
        config: GMEOptionsConfig,
        client: PolygonOptionsClient,
    ) -> None:
        self.config = config
        self.client = client

    @property
    def _snapshots_dir(self) -> Path:
        return self.config.data_dir / "snapshots"

    def collect(self, collection_date: date | None = None) -> pd.DataFrame:
        """Fetch the full options chain snapshot and return as DataFrame.

        Parameters
        ----------
        collection_date:
            Date to stamp on each row.  Defaults to today.

        Returns
        -------
        DataFrame with one row per contract.
        """
        import pandas as pd

        if collection_date is None:
            collection_date = date.today()

        raw = self.client.fetch_options_chain_snapshot(self.config.symbol)

        rows: list[dict] = []
        for snap in raw:
            details = snap.get("details", {})
            day = snap.get("day", {})
            greeks = snap.get("greeks", {})
            underlying = snap.get("underlying_asset", {})

            rows.append({
                "date": str(collection_date),
                "option_ticker": details.get("ticker", ""),
                "strike": details.get("strike_price", 0.0),
                "option_type": details.get("contract_type", ""),
                "expiration": details.get("expiration_date", ""),
                "close": day.get("close", 0.0),
                "volume": day.get("volume", 0),
                "open_interest": snap.get("open_interest", 0),
                "implied_volatility": snap.get("implied_volatility", float("nan")),
                "delta": greeks.get("delta", float("nan")),
                "gamma": greeks.get("gamma", float("nan")),
                "theta": greeks.get("theta", float("nan")),
                "vega": greeks.get("vega", float("nan")),
                "underlying_price": underlying.get("price", 0.0),
            })

        df = pd.DataFrame(rows)
        logger.info(
            "Collected snapshot: %d contracts for %s on %s",
            len(df), self.config.symbol, collection_date,
        )
        return df

    def save(self, df: pd.DataFrame, collection_date: date) -> Path:
        """Save a snapshot DataFrame as a dated Parquet file.

        Returns the path to the written file.
        """
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshots_dir / f"{collection_date}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved snapshot (%d rows) to %s", len(df), path)
        return path

    def load(self, collection_date: date) -> pd.DataFrame | None:
        """Load a snapshot Parquet file for a given date.

        Returns ``None`` if no snapshot exists for that date.
        """
        import pandas as pd

        path = self._snapshots_dir / f"{collection_date}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def run(self, collection_date: date | None = None) -> pd.DataFrame:
        """Collect the snapshot and save to disk in one call.

        Returns the collected DataFrame.
        """
        if collection_date is None:
            collection_date = date.today()

        df = self.collect(collection_date)
        if not df.empty:
            self.save(df, collection_date)
        return df
