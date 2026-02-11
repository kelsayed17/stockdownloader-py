"""Loads PriceData from CSV files or streams in Yahoo Finance historical format.

Expected columns: Date, Open, High, Low, Close, Adj Close, Volume
"""
from __future__ import annotations

import csv
import io
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

from stockdownloader.model import PriceData

logger = logging.getLogger(__name__)


class CsvPriceDataLoader:
    """Utility class for loading :class:`PriceData` from CSV files or streams.

    All methods are static; the class is not intended to be instantiated.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("CsvPriceDataLoader should not be instantiated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_file(filename: str | Path) -> list[PriceData]:
        """Load price data from *filename* (path to a CSV file).

        Returns an empty list if the file cannot be read.
        """
        try:
            with open(filename, newline="", encoding="utf-8") as fh:
                return _parse_records(fh)
        except Exception as exc:
            logger.warning("Error loading CSV file %s: %s", filename, exc)
            return []

    @staticmethod
    def load_from_stream(stream: BinaryIO) -> list[PriceData]:
        """Load price data from a binary *stream* (e.g. an HTTP response body).

        Returns an empty list if the stream cannot be parsed.
        """
        try:
            text_stream = io.TextIOWrapper(stream, encoding="utf-8")
            return _parse_records(text_stream)
        except Exception as exc:
            logger.warning("Error loading CSV from stream: %s", exc)
            return []


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_records(text_io: io.TextIOBase | io.TextIOWrapper) -> list[PriceData]:
    """Parse CSV rows into :class:`PriceData` instances.

    The first line is assumed to be a header and is skipped.
    """
    reader = csv.reader(text_io)
    next(reader, None)  # skip header

    data: list[PriceData] = []
    for line in reader:
        try:
            date = line[0]
            open_ = Decimal(line[1])
            high = Decimal(line[2])
            low = Decimal(line[3])
            close = Decimal(line[4])
            adj_close = Decimal(line[5]) if len(line) > 5 else close
            volume = int(line[6]) if len(line) > 6 else 0

            data.append(
                PriceData(
                    date=date,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    adj_close=adj_close,
                    volume=volume,
                )
            )
        except (InvalidOperation, ValueError, IndexError):
            # Skip lines with invalid data (e.g. "null" values)
            continue

    return data
