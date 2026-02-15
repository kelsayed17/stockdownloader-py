"""Writes IntradayPriceData to CSV files in Yahoo Finance 5-minute bar format.

Produces CSV files compatible with :class:`IntradayCsvLoader` using the
format: ``Datetime,Open,High,Low,Close,Volume``.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from stockdownloader.model.intraday_price_data import IntradayPriceData

logger = logging.getLogger(__name__)

_HEADER = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


def write_to_file(
    data: list[IntradayPriceData],
    filepath: str | Path,
) -> int:
    """Write intraday bars to *filepath* in CSV format.

    Parameters
    ----------
    data:
        Bars to write, in chronological order.
    filepath:
        Destination CSV file path.  Parent directories are created if
        they do not exist.

    Returns
    -------
    int
        Number of rows written (excluding header).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)

        for bar in data:
            writer.writerow([
                bar.date,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ])

    logger.info("Wrote %d intraday bars to %s", len(data), filepath)
    return len(data)
