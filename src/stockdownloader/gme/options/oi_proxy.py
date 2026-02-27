"""OI proxy estimation and enrichment for historical options data.

Estimates open interest from volume data using exponential-decay
cumulative volume, calibrated against real snapshot OI where available.

Usage::

    estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.03)
    enriched = estimator.estimate(bars_df)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OIProxyEstimator:
    """Estimates open interest from volume using exponential-decay proxy.

    Parameters
    ----------
    calibration_ratio:
        Multiplier applied to cumulative volume to estimate OI.
        Learned from real snapshot data via :meth:`calibrate`.
    decay_rate:
        Exponential decay rate per day (lambda).  Higher values
        mean OI decays faster as volume ages.
    """

    DEFAULT_RATIO: float = 0.15

    def __init__(
        self,
        calibration_ratio: float | None = None,
        decay_rate: float = 0.03,
    ) -> None:
        self.calibration_ratio = calibration_ratio or self.DEFAULT_RATIO
        self.decay_rate = decay_rate

    def calibrate(
        self,
        snapshot_df: pd.DataFrame,
        bars_df: pd.DataFrame,
    ) -> None:
        """Learn calibration ratio from real snapshot OI vs cumulative volume.

        Parameters
        ----------
        snapshot_df:
            Snapshot data with ``option_ticker``, ``open_interest``.
        bars_df:
            Bar data with ``option_ticker``, ``volume``, ``date``.
        """
        if snapshot_df.empty or bars_df.empty:
            logger.info(
                "No calibration data available, using default ratio %.3f",
                self.DEFAULT_RATIO,
            )
            self.calibration_ratio = self.DEFAULT_RATIO
            return

        # Compute cumulative volume per contract from bars
        cum_vol = (
            bars_df.groupby("option_ticker")["volume"]
            .sum()
            .rename("cum_volume")
        )

        # Join with snapshot OI
        snap_oi = snapshot_df.set_index("option_ticker")["open_interest"]
        joined = pd.concat([snap_oi, cum_vol], axis=1).dropna()
        joined = joined[joined["cum_volume"] > 0]

        if joined.empty:
            self.calibration_ratio = self.DEFAULT_RATIO
            return

        ratios = joined["open_interest"] / joined["cum_volume"]
        self.calibration_ratio = float(ratios.mean())
        logger.info("Calibrated OI ratio: %.4f from %d contracts", self.calibration_ratio, len(joined))

    def estimate(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Add estimated open_interest to bars DataFrame.

        Computes: ``oi = cumsum(volume * exp(-decay * days_since)) * ratio``

        Parameters
        ----------
        bars_df:
            Must have ``option_ticker``, ``date``, ``volume``, ``open_interest``.

        Returns
        -------
        Copy of bars_df with ``open_interest`` replaced by proxy values.
        """
        df = bars_df.copy()

        if df.empty:
            return df

        results: list[pd.DataFrame] = []

        for ticker, group in df.groupby("option_ticker"):
            g = group.sort_values("date").copy()
            dates = pd.to_datetime(g["date"])
            first_date = dates.iloc[0]
            days_elapsed = (dates - first_date).dt.days.values.astype(float)

            volumes = g["volume"].values.astype(float)

            # Compute decayed cumulative volume
            n = len(volumes)
            oi_values = np.zeros(n)

            for i in range(n):
                # Sum all prior volume with decay from their age
                decayed_sum = 0.0
                for j in range(i + 1):
                    age = days_elapsed[i] - days_elapsed[j]
                    decayed_sum += volumes[j] * math.exp(-self.decay_rate * age)
                oi_values[i] = decayed_sum * self.calibration_ratio

            g["open_interest"] = oi_values.astype(int)
            results.append(g)

        return pd.concat(results, ignore_index=True)

    def save_calibration(self, path: Path) -> None:
        """Save calibration parameters to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "calibration_ratio": self.calibration_ratio,
                    "decay_rate": self.decay_rate,
                },
                f,
                indent=2,
            )

    @classmethod
    def load_calibration(cls, path: Path) -> OIProxyEstimator:
        """Load calibration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            calibration_ratio=data["calibration_ratio"],
            decay_rate=data.get("decay_rate", 0.03),
        )


class OIEnricher:
    """Merges real and proxy OI into monthly bar Parquet files.

    Priority cascade:
    1. Snapshot OI (from ``snapshots/`` directory)
    2. Proxy OI (from ``OIProxyEstimator``)

    Parameters
    ----------
    data_dir:
        Root data directory containing ``monthly/`` and ``snapshots/``.
    estimator:
        Calibrated OI proxy estimator.
    """

    def __init__(
        self,
        data_dir: Path,
        estimator: OIProxyEstimator,
    ) -> None:
        self.data_dir = data_dir
        self.estimator = estimator

    def enrich_month(self, month_path: Path) -> pd.DataFrame:
        """Enrich a single monthly Parquet with OI data.

        Parameters
        ----------
        month_path:
            Path to a monthly Parquet file (e.g. ``monthly/2023-01.parquet``).

        Returns
        -------
        Enriched DataFrame with ``open_interest`` and ``oi_source`` columns.
        """
        df = pd.read_parquet(month_path)

        if df.empty:
            df["oi_source"] = pd.Series(dtype=str)
            return df

        # Step 1: Apply proxy OI to all rows
        df = self.estimator.estimate(df)
        df["oi_source"] = "proxy"

        # Step 2: Overlay real snapshot OI where available
        snapshots_dir = self.data_dir / "snapshots"
        if snapshots_dir.exists():
            for snap_file in snapshots_dir.glob("*.parquet"):
                snap_df = pd.read_parquet(snap_file)
                snap_date = snap_file.stem  # YYYY-MM-DD

                if snap_df.empty:
                    continue

                # Build lookup: (date, option_ticker) -> open_interest
                snap_lookup = {}
                for _, row in snap_df.iterrows():
                    key = (str(row.get("date", snap_date))[:10], row["option_ticker"])
                    snap_lookup[key] = row["open_interest"]

                # Overlay
                for idx, row in df.iterrows():
                    key = (str(row["date"])[:10], row["option_ticker"])
                    if key in snap_lookup:
                        df.at[idx, "open_interest"] = snap_lookup[key]
                        df.at[idx, "oi_source"] = "snapshot"

        return df

    def enrich_all(self) -> int:
        """Enrich all monthly Parquet files in place.

        Returns the number of files enriched.
        """
        monthly_dir = self.data_dir / "monthly"
        if not monthly_dir.exists():
            logger.warning("No monthly directory at %s", monthly_dir)
            return 0

        count = 0
        for pf in sorted(monthly_dir.glob("*.parquet")):
            logger.info("Enriching %s", pf.name)
            enriched = self.enrich_month(pf)
            enriched.to_parquet(pf, index=False)
            count += 1

            proxy_count = (enriched["oi_source"] == "proxy").sum()
            snap_count = (enriched["oi_source"] == "snapshot").sum()
            logger.info(
                "  %s: %d rows (proxy=%d, snapshot=%d)",
                pf.name, len(enriched), proxy_count, snap_count,
            )

        return count
