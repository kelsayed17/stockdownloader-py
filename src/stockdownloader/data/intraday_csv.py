"""Reads and writes IntradayPriceData CSV files (5-minute bar format).

Merges the former ``intraday_csv_loader`` and ``intraday_csv_writer`` modules
into a single I/O module.  Also exports :func:`normalize_tz` for use by
:mod:`intraday_data_accumulator`.

Expected CSV columns: Datetime, Open, High, Low, Close, Volume
(No Adj Close column -- ``adj_close`` defaults to ``close``.)
"""
from __future__ import annotations

import csv
import io
import logging
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

from stockdownloader.core.models.price import IntradayPriceData

logger = logging.getLogger(__name__)

_HEADER = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


# ======================================================================
# Public API — reading
# ======================================================================


class IntradayCsvLoader:
    """Loads :class:`IntradayPriceData` from 5-minute bar CSV files.

    All methods are static; the class is not intended to be instantiated.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("IntradayCsvLoader should not be instantiated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_file(filename: str | Path) -> list[IntradayPriceData]:
        """Load intraday price data from *filename*.

        Returns an empty list if the file cannot be read.
        """
        try:
            with open(filename, newline="", encoding="utf-8") as fh:
                return _parse_intraday_records(fh)
        except (OSError, csv.Error, InvalidOperation, ValueError) as exc:
            logger.warning("Error loading intraday CSV file %s: %s", filename, exc)
            return []

    @staticmethod
    def load_from_stream(stream: BinaryIO) -> list[IntradayPriceData]:
        """Load intraday price data from a binary *stream*.

        Returns an empty list if the stream cannot be parsed.
        """
        try:
            text_stream = io.TextIOWrapper(stream, encoding="utf-8")
            return _parse_intraday_records(text_stream)
        except (OSError, csv.Error, InvalidOperation, ValueError) as exc:
            logger.warning("Error loading intraday CSV from stream: %s", exc)
            return []


# ======================================================================
# Public API — writing
# ======================================================================


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


# ======================================================================
# Shared timezone helper
# ======================================================================


def normalize_tz(dt_str: str) -> str:
    """Ensure the timezone offset contains a colon (ISO-8601).

    Yahoo's ``strftime("%z")`` produces ``-0500``; Polygon uses ``-05:00``.
    Normalising here ensures consistent datetime keys regardless of source.
    """
    m = re.search(r'([+-])(\d{2})(\d{2})$', dt_str)
    if m and ':' not in dt_str[-6:]:
        return dt_str[:-4] + m.group(2) + ':' + m.group(3)
    return dt_str


# ======================================================================
# Internal helpers
# ======================================================================


def _parse_intraday_records(
    text_io: io.TextIOBase | io.TextIOWrapper,
) -> list[IntradayPriceData]:
    """Parse CSV rows into :class:`IntradayPriceData` instances.

    Supports two column layouts:

    * **5-column** (Datetime, Open, High, Low, Close, Volume) --
      ``adj_close`` defaults to ``close``.
    * **7-column** (Datetime, Open, High, Low, Close, Adj Close, Volume) --
      fully explicit.

    The first line is assumed to be a header and is skipped.
    """
    reader = csv.reader(text_io)
    next(reader, None)  # skip header

    data: list[IntradayPriceData] = []
    for line in reader:
        try:
            dt_str = normalize_tz(line[0].strip())
            open_ = Decimal(line[1])
            high = Decimal(line[2])
            low = Decimal(line[3])
            close = Decimal(line[4])

            # Determine layout by checking if column 5 looks like a price
            # or an integer volume.
            if len(line) >= 7:
                adj_close = Decimal(line[5])
                volume = int(Decimal(line[6]).to_integral_value())
            elif len(line) >= 6:
                # Could be Adj Close or Volume -- heuristic: if it contains
                # a decimal point it's a price, else volume.
                col5 = line[5].strip()
                if "." in col5:
                    # Likely Adj Close with no Volume
                    adj_close = Decimal(col5)
                    volume = 0
                else:
                    adj_close = close
                    volume = int(Decimal(col5).to_integral_value())
            else:
                adj_close = close
                volume = 0

            data.append(
                IntradayPriceData(
                    date=dt_str,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    adj_close=adj_close,
                    volume=volume,
                )
            )
        except (InvalidOperation, ValueError, IndexError) as exc:
            logger.debug("Skipping invalid intraday CSV row: %s (%s)", line, exc)
            continue

    return data
