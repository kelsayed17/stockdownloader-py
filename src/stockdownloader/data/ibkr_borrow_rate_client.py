"""Fetches real stock borrow rates from Interactive Brokers.

IBKR publishes a daily file of all shortable securities with availability
and fee rates via FTP at ``ftp://shortstock:@ftp3.interactivebrokers.com/usa.txt``.

No IBKR account is required — the FTP file uses anonymous-style access
(username ``shortstock``, blank password).

Usage::

    client = IbkrBorrowRateClient()
    rate = client.fetch_borrow_rate("GME")
    # rate = IbkrBorrowRate(symbol='GME', fee_rate=34.5, available=150000, ...)

    # Or fetch and cache for squeeze predictor
    client.fetch_and_cache("GME")
    # Writes to data/cache/borrow_rate/GME_ibkr.json
"""

from __future__ import annotations

import ftplib
import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_FTP_HOST = "ftp3.interactivebrokers.com"
_FTP_USER = "shortstock"
_FTP_PASS = ""
_FTP_FILE = "usa.txt"

# Alternative HTTP URL (sometimes more reliable than FTP)
_HTTP_URL = (
    "https://www.interactivebrokers.com/en/index.php"
    "?f=4587&ns=T&action=downloadFile&file=usa"
)


@dataclass(frozen=True, slots=True)
class IbkrBorrowRate:
    """Parsed borrow rate record from IBKR shortable securities file."""

    symbol: str
    currency: str
    fee_rate: float      # Annual borrow fee % (e.g. 34.5 means 34.5%)
    available: int       # Number of shares available to short
    timestamp: str       # When data was fetched (ISO format)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")


class IbkrBorrowRateClient:
    """Fetches real borrow rate data from Interactive Brokers.

    Primary: FTP download of usa.txt shortable securities file.
    Fallback: HTTP download from IBKR website.

    The file is pipe-delimited with columns:
    ``BOL|SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|...``

    Parameters
    ----------
    cache_dir:
        Directory for JSON cache files.
    """

    def __init__(self, cache_dir: str = "data/cache/borrow_rate") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._parsed_data: dict[str, IbkrBorrowRate] | None = None
        self._last_fetch_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_borrow_rate(self, symbol: str) -> IbkrBorrowRate | None:
        """Fetch the current borrow rate for a single symbol.

        Downloads the full IBKR file (if not already cached in memory),
        parses it, and returns the record for *symbol*.

        Returns ``None`` if the symbol is not in the shortable list
        (which may mean it's not shortable or very hard to borrow).
        """
        symbol_upper = symbol.upper()

        # Use in-memory cache if fresh (< 30 min old)
        if self._parsed_data and (time.time() - self._last_fetch_time) < 1800:
            return self._parsed_data.get(symbol_upper)

        # Try FTP first, then HTTP fallback
        raw_lines = self._download_ftp()
        if raw_lines is None:
            raw_lines = self._download_http()

        if raw_lines is None:
            logger.warning(
                "Could not download IBKR shortable file — "
                "falling back to cache"
            )
            return self._load_cached_rate(symbol_upper)

        self._parsed_data = self._parse_file(raw_lines)
        self._last_fetch_time = time.time()

        rate = self._parsed_data.get(symbol_upper)
        if rate is not None:
            self._save_cached_rate(symbol_upper, rate)

        return rate

    def fetch_and_cache(self, symbol: str) -> IbkrBorrowRate | None:
        """Fetch borrow rate and persist to cache file.

        Returns the :class:`IbkrBorrowRate` or ``None``.
        """
        rate = self.fetch_borrow_rate(symbol)
        if rate is not None:
            self._save_cached_rate(symbol.upper(), rate)
        return rate

    def fetch_multiple(
        self, symbols: list[str]
    ) -> dict[str, IbkrBorrowRate]:
        """Fetch borrow rates for multiple symbols in one download.

        Returns a dict mapping symbol -> rate.  Symbols not found in
        the shortable list are omitted from the result.
        """
        # Force a fresh download
        raw_lines = self._download_ftp()
        if raw_lines is None:
            raw_lines = self._download_http()

        if raw_lines is None:
            # Fall back to individual caches
            result: dict[str, IbkrBorrowRate] = {}
            for sym in symbols:
                cached = self._load_cached_rate(sym.upper())
                if cached is not None:
                    result[sym.upper()] = cached
            return result

        self._parsed_data = self._parse_file(raw_lines)
        self._last_fetch_time = time.time()

        result = {}
        for sym in symbols:
            sym_upper = sym.upper()
            if sym_upper in self._parsed_data:
                rate = self._parsed_data[sym_upper]
                result[sym_upper] = rate
                self._save_cached_rate(sym_upper, rate)

        return result

    # ------------------------------------------------------------------
    # Download methods
    # ------------------------------------------------------------------

    def _download_ftp(self) -> list[str] | None:
        """Download the shortable securities file via FTP.

        Returns list of lines, or ``None`` on failure.
        """
        ftp = None
        try:
            logger.info("Downloading IBKR shortable file via FTP...")
            buf = io.BytesIO()

            ftp = ftplib.FTP(timeout=30)
            ftp.connect(_FTP_HOST)
            ftp.login(_FTP_USER, _FTP_PASS)
            ftp.retrbinary(f"RETR {_FTP_FILE}", buf.write)
            ftp.quit()
            ftp = None  # Mark as closed

            buf.seek(0)
            text = buf.read().decode("utf-8", errors="replace")
            lines = text.splitlines()

            logger.info(
                "IBKR FTP: downloaded %d lines (%d KB)",
                len(lines), len(text) // 1024,
            )
            return lines

        except (OSError, EOFError, ConnectionError, TimeoutError) as exc:
            logger.warning("IBKR FTP download failed: %s", exc)
            return None
        except Exception as exc:
            # Catch-all for ftplib errors and other unexpected issues
            logger.warning("IBKR FTP download failed (unexpected): %s", exc)
            return None
        finally:
            # Ensure FTP connection is cleaned up
            if ftp is not None:
                try:
                    ftp.quit()
                except Exception:
                    try:
                        ftp.close()
                    except Exception:
                        pass

    def _download_http(self) -> list[str] | None:
        """Fallback: download via HTTP from IBKR website."""
        try:
            import requests

            logger.info("Downloading IBKR shortable file via HTTP (fallback)...")
            resp = requests.get(
                _HTTP_URL,
                headers={"User-Agent": "StockDownloader/1.0"},
                timeout=60,
            )
            if resp.status_code == 200:
                lines = resp.text.splitlines()
                logger.info(
                    "IBKR HTTP: downloaded %d lines", len(lines),
                )
                return lines

            logger.warning(
                "IBKR HTTP download returned %d", resp.status_code,
            )
        except Exception as exc:
            logger.warning("IBKR HTTP download failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_file(lines: list[str]) -> dict[str, IbkrBorrowRate]:
        """Parse the pipe-delimited IBKR shortable securities file.

        Expected format (header line + data lines)::

            BOL|SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|...

        FEERATE is the annual borrow fee as a percentage.
        AVAILABLE is the number of shares available to short.
        Some lines may have ``#BOL`` prefix for header/metadata.
        """
        result: dict[str, IbkrBorrowRate] = {}
        timestamp = datetime.utcnow().isoformat()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 8:
                continue

            # Skip header-like rows
            if parts[0] == "BOL" or parts[1] == "SYM":
                continue

            try:
                symbol = parts[1].strip().upper()
                currency = parts[2].strip()

                # FEERATE is column index 7 (0-based)
                fee_str = parts[7].strip()
                if not fee_str or fee_str == "NA":
                    continue
                fee_rate = float(fee_str)

                # AVAILABLE is column index 8
                avail_str = parts[8].strip() if len(parts) > 8 else "0"
                available = int(avail_str) if avail_str.isdigit() else 0

                if symbol:
                    result[symbol] = IbkrBorrowRate(
                        symbol=symbol,
                        currency=currency,
                        fee_rate=fee_rate,
                        available=available,
                        timestamp=timestamp,
                    )
            except (ValueError, IndexError):
                continue

        logger.info("IBKR: parsed %d shortable securities", len(result))
        return result

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cached_rate(self, symbol: str) -> IbkrBorrowRate | None:
        """Load a cached borrow rate for *symbol*."""
        cache_file = self._cache_dir / f"{symbol}_ibkr.json"
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return IbkrBorrowRate(**data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed to load IBKR cache for %s: %s", symbol, exc)
            return None

    def _save_cached_rate(self, symbol: str, rate: IbkrBorrowRate) -> None:
        """Persist a borrow rate to JSON cache."""
        cache_file = self._cache_dir / f"{symbol}_ibkr.json"
        try:
            cache_file.write_text(
                json.dumps(asdict(rate), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("Failed to save IBKR cache for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Historical tracking
    # ------------------------------------------------------------------

    def append_to_history(self, symbol: str, rate: IbkrBorrowRate) -> None:
        """Append a rate snapshot to the historical log.

        Maintains a JSONL (JSON Lines) file for time-series analysis.
        """
        history_file = self._cache_dir / f"{symbol}_ibkr_history.jsonl"
        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rate)) + "\n")
        except OSError as exc:
            logger.debug(
                "Failed to append IBKR history for %s: %s", symbol, exc,
            )

    def load_history(self, symbol: str) -> list[IbkrBorrowRate]:
        """Load all historical snapshots for *symbol*."""
        history_file = self._cache_dir / f"{symbol}_ibkr_history.jsonl"
        if not history_file.exists():
            return []

        rates: list[IbkrBorrowRate] = []
        try:
            with open(history_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        rates.append(IbkrBorrowRate(**data))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug(
                "Failed to load IBKR history for %s: %s", symbol, exc,
            )
        return rates
