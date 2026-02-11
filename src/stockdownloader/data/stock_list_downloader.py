"""Downloads stock ticker lists from Nasdaq API, Zacks, and Yahoo Earnings.

NASDAQ FTP (ftp://ftp.nasdaqtrader.com/) has been replaced with:
- Nasdaq screener API: https://api.nasdaq.com/api/screener/stocks
- HTTP fallback: http://ftp.nasdaqtrader.com/dynamic/SymDir/ for symbol
  directory files
"""
from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Set

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INCOMPLETE_FILE = "incomplete.txt"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class StockListDownloader:
    """Downloads and manages stock ticker lists from multiple sources."""

    def __init__(self) -> None:
        self._nasdaq_list: set[str] = set()
        self._others_list: set[str] = set()
        self._mfunds_list: set[str] = set()
        self._zacks_list: set[str] = set()
        self._earnings_list: set[str] = set()
        self._incomplete_list: set[str] = set()

    # ------------------------------------------------------------------
    # Download methods
    # ------------------------------------------------------------------

    def download_nasdaq(self) -> None:
        """Download NASDAQ-listed tickers (API first, pipe-delimited fallback)."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                self._nasdaq_list = self._download_from_nasdaq_api("nasdaq")
                if self._nasdaq_list:
                    return
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.debug("Retrying Nasdaq stock list download, attempt %d", attempt + 1)
                else:
                    logger.warning("Failed Nasdaq stock list download after %d retries: %s", _MAX_RETRIES, exc)

        # Fallback: HTTP symbol directory
        if not self._nasdaq_list:
            url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
            filename = "nasdaqlisted.txt"
            _download_file(url, filename)
            self._nasdaq_list = _read_pipe_delimited_file(filename)

    def download_others(self) -> None:
        """Download NYSE and AMEX tickers (API first, pipe-delimited fallback)."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                nyse = self._download_from_nasdaq_api("nyse")
                amex = self._download_from_nasdaq_api("amex")
                self._others_list.update(nyse)
                self._others_list.update(amex)
                if self._others_list:
                    return
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.debug("Retrying Other exchanges stock list download, attempt %d", attempt + 1)
                else:
                    logger.warning("Failed Other exchanges stock list download after %d retries: %s", _MAX_RETRIES, exc)

        # Fallback: HTTP symbol directory
        if not self._others_list:
            url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
            filename = "otherslisted.txt"
            _download_file(url, filename)
            self._others_list = _read_pipe_delimited_file(filename)

    def download_mutual_funds(self) -> None:
        """Download mutual fund symbols from NASDAQ FTP (HTTP)."""
        url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/mfundslist.txt"
        filename = "mfundslist.txt"
        _download_file(url, filename)
        self._mfunds_list = _read_pipe_delimited_file(filename)

    def download_zacks(self) -> None:
        """Download Zacks Rank-1 stock list."""
        url = "https://www.zacks.com/portfolios/rank/rank_excel.php?rank=1&reference_id=all"
        filename = "rank_1.txt"
        _download_file(url, filename)

        try:
            with open(filename, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Skip first 2 lines
                next(reader, None)
                next(reader, None)
                for row in reader:
                    if row:
                        self._zacks_list.add(row[0])
        except Exception as exc:
            logger.warning("Error reading Zacks file: %s", exc)

    def download_yahoo_earnings(self, market_date: date) -> None:
        """Download Yahoo Finance earnings calendar for *market_date*."""
        date_str = market_date.isoformat()
        url = f"https://finance.yahoo.com/calendar/earnings?day={date_str}"
        yahoo_earnings: dict[str, list[str]] = defaultdict(list)

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=15,
            )
            soup = BeautifulSoup(resp.text, "html.parser")

            table_body = soup.select_one("table tbody")
            if table_body is not None:
                for tr in table_body.find_all("tr"):
                    cells = tr.find_all("td")
                    if len(cells) >= 4:
                        ticker = cells[1].get_text(strip=True)
                        time = cells[3].get_text(strip=True)
                        yahoo_earnings[time].append(ticker)

            for time_key, tickers in yahoo_earnings.items():
                self._earnings_list.update(tickers)
        except Exception as exc:
            logger.warning(
                "Error downloading Yahoo earnings for %s: %s", date_str, exc
            )

        for time_key, tickers in yahoo_earnings.items():
            print(f"{date_str}\t{time_key}:\t{tickers}")

    # ------------------------------------------------------------------
    # Incomplete list management
    # ------------------------------------------------------------------

    def read_incomplete(self) -> None:
        """Read the incomplete ticker list from disk."""
        self._incomplete_list = _read_lines(_INCOMPLETE_FILE)

    def add_incomplete(self, ticker: str) -> None:
        """Add *ticker* to the in-memory incomplete list."""
        self._incomplete_list.add(ticker)

    def append_incomplete(self, ticker: str) -> None:
        """Append *ticker* to the incomplete file on disk."""
        _append_line(ticker, _INCOMPLETE_FILE)

    def write_incomplete(self) -> None:
        """Write the full incomplete list to disk."""
        _write_lines(self._incomplete_list, _INCOMPLETE_FILE)

    # ------------------------------------------------------------------
    # Properties (read-only copies)
    # ------------------------------------------------------------------

    @property
    def nasdaq_list(self) -> Set[str]:
        return frozenset(self._nasdaq_list)

    @property
    def others_list(self) -> Set[str]:
        return frozenset(self._others_list)

    @property
    def mfunds_list(self) -> Set[str]:
        return frozenset(self._mfunds_list)

    @property
    def zacks_list(self) -> Set[str]:
        return frozenset(self._zacks_list)

    @property
    def earnings_list(self) -> Set[str]:
        return frozenset(self._earnings_list)

    @property
    def incomplete_list(self) -> Set[str]:
        return frozenset(self._incomplete_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_from_nasdaq_api(exchange: str) -> set[str]:
        """Download stock tickers from the Nasdaq screener API."""
        tickers: set[str] = set()
        url = (
            "https://api.nasdaq.com/api/screener/stocks"
            f"?tableonly=true&limit=10000&exchange={exchange}"
        )
        resp = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
            timeout=30,
        )
        body = resp.json()

        data = body.get("data")
        if data is None:
            return tickers

        rows = data.get("rows")
        if rows is None:
            return tickers

        for row in rows:
            symbol = row.get("symbol")
            if symbol:
                ticker = symbol.strip()
                if ticker:
                    tickers.add(ticker)

        return tickers


# ------------------------------------------------------------------
# Module-level file I/O helpers
# ------------------------------------------------------------------


def _download_file(url: str, filename: str) -> None:
    """Download *url* and save to *filename* with retry logic."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            Path(filename).write_bytes(resp.content)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                logger.debug("Retrying download %s, attempt %d", filename, attempt + 1)
            else:
                logger.warning(
                    "Failed download %s after %d retries: %s",
                    filename,
                    _MAX_RETRIES,
                    last_exc,
                )


def _read_pipe_delimited_file(filename: str) -> set[str]:
    """Read a pipe-delimited symbol directory file.  Skips the header and
    the last line (which is typically a file-creation timestamp).
    """
    tickers: set[str] = set()
    try:
        lines = Path(filename).read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            return tickers

        # Skip header (index 0) and last line
        for line in lines[1:-1]:
            pipe_idx = line.find("|")
            if pipe_idx > 0:
                tickers.add(line[:pipe_idx])
    except Exception as exc:
        logger.warning("Error reading file %s: %s", filename, exc)

    return tickers


def _read_lines(filename: str) -> set[str]:
    """Read non-blank lines from *filename* into a set."""
    lines: set[str] = set()
    path = Path(filename)
    if not path.exists():
        return lines
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                lines.add(stripped)
    except Exception as exc:
        logger.warning("Error reading file %s: %s", filename, exc)
    return lines


def _write_lines(lines: set[str], filename: str) -> None:
    """Write a set of lines to *filename*, overwriting any existing content."""
    try:
        Path(filename).write_text(
            "\n".join(sorted(lines)), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Error writing file %s: %s", filename, exc)


def _append_line(line: str, filename: str) -> None:
    """Append a single *line* to *filename*."""
    try:
        with open(filename, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.warning("Error appending to %s: %s", filename, exc)
