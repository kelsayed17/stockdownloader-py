"""Exchange-specific Reg SHO threshold list source fetchers.

Each public function queries a single exchange source and returns a list
of :class:`~stockdownloader.data.regsho_threshold_client.ThresholdRecord`
instances for the requested symbol.

Sources
-------
- :func:`query_nyse` -- NYSE / NYSE Arca / NYSE American (2010+)
- :func:`query_nasdaq` -- Nasdaq-listed securities (2006+)
- :func:`query_cboe` -- CBOE BZX-listed ETFs (2015+)
- :func:`query_occ` -- OCC combined all-exchange list (recent ~4-6 weeks)

These were extracted from ``RegShoThresholdClient`` so they can be tested
and composed independently.  The orchestrating class still lives in
:mod:`stockdownloader.data.regsho_threshold_client`.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import requests

if TYPE_CHECKING:
    from stockdownloader.data.regsho_threshold_client import ThresholdRecord

try:
    from curl_cffi import requests as curl_requests  # noqa: F401

    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover -- optional dependency
    _HAS_CURL_CFFI = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- formerly module-private in regsho_threshold_client
# ---------------------------------------------------------------------------

# Retry limits
MAX_CONSECUTIVE_FAILURES = 15

# Rate-limiting delays (seconds) -- randomised to avoid pattern detection
NYSE_DELAY_RANGE = (1.5, 3.0)
NYSE_BATCH_SIZE = 25
NYSE_BATCH_PAUSE_RANGE = (30, 60)
NYSE_429_BACKOFF_RANGE = (120, 180)
OCC_RATE_LIMIT_DELAY = 0.5
CBOE_RATE_LIMIT_DELAY = 0.3

# URL templates
NYSE_URL_TEMPLATE = (
    "https://www.nyse.com/api/regulatory/threshold-securities/"
    "download?selectedDate={date}"
)
NASDAQ_URL_TEMPLATE = (
    "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date}.txt"
)
OCC_URL_TEMPLATE = (
    "https://marketdata.theocc.com/threshold-securities?reportDate={date}"
)
CBOE_URL_TEMPLATE = (
    "https://www.cboe.com/us/equities/market_statistics/"
    "reg_sho_threshold/{date}/csv/"
)

# Browser-like headers for curl_cffi requests (Sec-Fetch-* headers).
# User-Agent and sec-ch-ua are set automatically by curl_cffi impersonation.
BROWSER_HEADERS = {
    "Accept": "text/plain, text/csv, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def cffi_get(
    curl_session: object | None,
    url: str,
    timeout: int = 15,
) -> tuple[str, int]:
    """HTTP GET via ``curl_cffi`` with Chrome TLS impersonation.

    Falls back to :func:`curl_get` (subprocess) when *curl_session* is
    ``None``.

    Returns ``(body_text, http_status_code)``.
    """
    if curl_session is not None:
        try:
            resp = curl_session.get(url, timeout=timeout, headers=BROWSER_HEADERS)
            return resp.text, resp.status_code
        except Exception:
            return "", 0
    else:
        return curl_get(url, timeout=timeout)


def curl_get(url: str, timeout: int = 15) -> tuple[str, int]:
    """HTTP GET via system ``curl`` binary (fallback).

    Used when ``curl_cffi`` is not installed.  Spawns a subprocess for each
    request.

    Returns ``(body_text, http_status_code)``.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "-o", "-",
                "-w", "\n%{http_code}",
                "--max-time", str(timeout),
                url,
            ],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        output = result.stdout
        lines = output.rsplit("\n", 1)
        if len(lines) == 2:
            body = lines[0]
            try:
                status = int(lines[1].strip())
            except ValueError:
                status = 0
        else:
            body = output
            status = 0
        return body, status
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "", 0


# ---------------------------------------------------------------------------
# Rate-limiting helpers
# ---------------------------------------------------------------------------


def nyse_rate_limit(state: dict) -> None:
    """Rate limit for NYSE requests with randomised delay and batch pausing.

    *state* must contain keys ``"request_count"`` (int) and
    ``"last_request_time"`` (float, monotonic).  Both are mutated in-place.
    """
    state["request_count"] = state.get("request_count", 0) + 1

    if state["request_count"] % NYSE_BATCH_SIZE == 0:
        pause = random.uniform(*NYSE_BATCH_PAUSE_RANGE)
        logger.info(
            "NYSE batch pause: %.0fs after %d requests",
            pause, state["request_count"],
        )
        time.sleep(pause)
        state["last_request_time"] = time.monotonic()
        return

    actual_delay = random.uniform(*NYSE_DELAY_RANGE)
    now = time.monotonic()
    elapsed = now - state.get("last_request_time", 0.0)
    if elapsed < actual_delay:
        time.sleep(actual_delay - elapsed)
    state["last_request_time"] = time.monotonic()


def occ_rate_limit(state: dict) -> None:
    """Rate limit for OCC requests.

    *state* must contain key ``"last_request_time"`` (float, monotonic).
    """
    now = time.monotonic()
    elapsed = now - state.get("last_request_time", 0.0)
    if elapsed < OCC_RATE_LIMIT_DELAY:
        time.sleep(OCC_RATE_LIMIT_DELAY - elapsed)
    state["last_request_time"] = time.monotonic()


def cboe_rate_limit(state: dict) -> None:
    """Rate limit for CBOE requests.

    *state* must contain key ``"last_request_time"`` (float, monotonic).
    """
    now = time.monotonic()
    elapsed = now - state.get("last_request_time", 0.0)
    if elapsed < CBOE_RATE_LIMIT_DELAY:
        time.sleep(CBOE_RATE_LIMIT_DELAY - elapsed)
    state["last_request_time"] = time.monotonic()


# ---------------------------------------------------------------------------
# NYSE progress helpers
# ---------------------------------------------------------------------------


def save_nyse_progress(
    progress_dir: Path, queried_dates: set[str]
) -> None:
    """Persist set of NYSE dates already queried."""
    progress_file = progress_dir / "regsho_nyse.json"
    try:
        existing: set[str] = set()
        if progress_file.exists():
            existing = set(json.loads(
                progress_file.read_text(encoding="utf-8")
            ))
        combined = sorted(existing | queried_dates)
        progress_file.write_text(
            json.dumps(combined), encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Failed to save NYSE progress: %s", exc)


def load_nyse_progress(
    progress_dir: Path, symbol_dir: Path
) -> set[str]:
    """Load previously queried NYSE dates.

    Handles legacy migration from ``regsho_nyse_progress.json`` in the
    symbol directory to ``regsho_nyse.json`` in the ``.progress/``
    directory.
    """
    progress_file = progress_dir / "regsho_nyse.json"
    legacy = symbol_dir / "regsho_nyse_progress.json"
    if not progress_file.exists() and legacy.exists():
        legacy.rename(progress_file)
        logger.info("Migrated %s -> %s", legacy, progress_file)
    if not progress_file.exists():
        return set()
    try:
        return set(json.loads(progress_file.read_text(encoding="utf-8")))
    except Exception:
        return set()


def incremental_nyse_save(
    symbol: str,
    new_records: list[ThresholdRecord],
    cached_dates: set[str],
    queried_dates: set[str],
    load_cache_fn: Callable[[str], list[ThresholdRecord] | None],
    save_cache_fn: Callable[[str, list[ThresholdRecord]], None],
    save_progress_fn: Callable[[set[str]], None],
) -> None:
    """Merge *new_records* with existing cache and save.

    Called periodically during long NYSE backfills so that progress is
    preserved if the process is interrupted.
    """
    try:
        cached = load_cache_fn(symbol) or []
        by_key: dict[tuple[str, str], ThresholdRecord] = {}
        for r in cached:
            by_key[(r.date, r.market)] = r
        for r in new_records:
            by_key[(r.date, r.market)] = r
        merged = sorted(by_key.values(), key=lambda r: r.date)
        save_cache_fn(symbol, merged)
        for r in new_records:
            cached_dates.add(r.date)
        save_progress_fn(queried_dates)
        logger.info(
            "NYSE incremental save: %d threshold records, "
            "%d dates queried for %s",
            len(merged), len(queried_dates), symbol,
        )
    except Exception as exc:
        logger.warning(
            "NYSE incremental save failed for %s: %s", symbol, exc,
        )


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------


def query_nyse(
    symbol: str,
    lookback_days: int = 5500,
    *,
    cached_dates: set[str] | None = None,
    curl_session: object | None = None,
    session: requests.Session | None = None,
    load_cache_fn: Callable[[str], list[ThresholdRecord] | None] | None = None,
    save_cache_fn: Callable[[str, list[ThresholdRecord]], None] | None = None,
    progress_dir_fn: Callable[[str], Path] | None = None,
    symbol_dir_fn: Callable[[str], Path] | None = None,
) -> list[ThresholdRecord]:
    """Query NYSE daily threshold list API.

    The NYSE API returns pipe-delimited text covering NYSE, NYSE Arca,
    and NYSE American listed securities.  Data is available from ~2010
    onward.  Format::

        Symbol|Security Name|Market Category|Reg SHO Threshold Flag||

    Uses ``curl_cffi`` with Chrome TLS impersonation to bypass Cloudflare
    JA3/JA4 TLS fingerprint blocking.  Randomised delays and batch pausing
    avoid triggering rate limits.

    Saves incrementally after every batch to preserve progress if the
    process is interrupted.
    """
    from stockdownloader.data.regsho_threshold_client import ThresholdRecord

    if cached_dates is None:
        cached_dates = set()

    # Resolve progress directory
    _progress_dir: Path | None = None
    _symbol_dir: Path | None = None
    if progress_dir_fn is not None:
        _progress_dir = progress_dir_fn(symbol)
    if symbol_dir_fn is not None:
        _symbol_dir = symbol_dir_fn(symbol)

    # Load dates already queried in prior (interrupted) runs
    queried_dates: set[str] = set()
    if _progress_dir is not None and _symbol_dir is not None:
        queried_dates = load_nyse_progress(_progress_dir, _symbol_dir)
        if queried_dates:
            logger.info(
                "NYSE: resuming -- %d dates already queried in prior runs",
                len(queried_dates),
            )

    # Build save_progress closure for incremental_nyse_save
    def _save_progress(qd: set[str]) -> None:
        if _progress_dir is not None:
            save_nyse_progress(_progress_dir, qd)

    records: list[ThresholdRecord] = []
    today = date.today()
    consecutive_failures = 0
    rate_state: dict = {"request_count": 0, "last_request_time": 0.0}

    for offset in range(lookback_days):
        check_date = today - timedelta(days=offset)
        if check_date.weekday() >= 5:
            continue

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Too many consecutive NYSE failures (%d), stopping at %s",
                consecutive_failures, check_date.isoformat(),
            )
            break

        date_str = check_date.strftime("%Y-%m-%d")

        if date_str in cached_dates or date_str in queried_dates:
            consecutive_failures = 0
            continue

        url = NYSE_URL_TEMPLATE.format(date=date_str)

        try:
            nyse_rate_limit(rate_state)

            if (
                rate_state["request_count"] > 0
                and rate_state["request_count"] % NYSE_BATCH_SIZE == 0
                and load_cache_fn is not None
                and save_cache_fn is not None
            ):
                incremental_nyse_save(
                    symbol, records, cached_dates, queried_dates,
                    load_cache_fn, save_cache_fn, _save_progress,
                )

            text, status = cffi_get(curl_session, url)

            if status == 200 and text:
                queried_dates.add(date_str)
                lines = text.strip().splitlines()
                if len(lines) <= 2:
                    consecutive_failures = 0
                    continue
                consecutive_failures = 0
                for line in lines:
                    parts = line.split("|")
                    if len(parts) >= 4 and parts[0].strip() == symbol:
                        market = parts[2].strip() or "NYSE"
                        records.append(ThresholdRecord(
                            date=check_date.isoformat(),
                            symbol=symbol,
                            market=market,
                            threshold_shares=0,
                            consecutive_days=0,
                        ))
                        break
            elif status in (404, 204):
                queried_dates.add(date_str)
                consecutive_failures = 0
            elif status in (403, 429):
                consecutive_failures += 1
                if consecutive_failures in (5, 10):
                    wait_secs = random.uniform(*NYSE_429_BACKOFF_RANGE)
                    logger.info(
                        "NYSE rate limited -- pausing %.0fs before retry "
                        "(failure %d)",
                        wait_secs, consecutive_failures,
                    )
                    time.sleep(wait_secs)
                    text2, status2 = cffi_get(curl_session, url)
                    if status2 == 200:
                        queried_dates.add(date_str)
                        consecutive_failures = 0
                        lines = text2.strip().splitlines()
                        for line in lines:
                            parts = line.split("|")
                            if (
                                len(parts) >= 4
                                and parts[0].strip() == symbol
                            ):
                                market = parts[2].strip() or "NYSE"
                                records.append(ThresholdRecord(
                                    date=check_date.isoformat(),
                                    symbol=symbol,
                                    market=market,
                                    threshold_shares=0,
                                    consecutive_days=0,
                                ))
                                break
                        continue
                logger.info(
                    "NYSE %d on %s (failures: %d)",
                    status, date_str, consecutive_failures,
                )
            else:
                consecutive_failures += 1
        except Exception as exc:
            consecutive_failures += 1
            logger.debug("NYSE request error for %s: %s", date_str, exc)

    if records:
        logger.info(
            "NYSE Reg SHO: %d threshold dates for %s", len(records), symbol,
        )
    if queried_dates and load_cache_fn is not None and save_cache_fn is not None:
        incremental_nyse_save(
            symbol, records, cached_dates, queried_dates,
            load_cache_fn, save_cache_fn, _save_progress,
        )
    return records


def query_occ(
    symbol: str,
    lookback_days: int = 60,
    session: requests.Session | None = None,
) -> list[ThresholdRecord]:
    """Query the OCC combined threshold list.

    The OCC (Options Clearing Corporation) publishes a combined threshold
    list covering NYSE, NASDAQ, NYSE Arca, NYSE American, and other
    exchanges.  Pipe-delimited format.

    .. note::
        OCC only retains approximately 4-6 weeks of history.
        The lookback is capped at 60 days by default.
    """
    from stockdownloader.data.regsho_threshold_client import ThresholdRecord

    if session is None:
        session = requests.Session()

    effective_lookback = min(lookback_days, 60)
    records: list[ThresholdRecord] = []
    today = date.today()
    consecutive_failures = 0
    rate_state: dict = {"last_request_time": 0.0}

    for offset in range(effective_lookback):
        check_date = today - timedelta(days=offset)
        if check_date.weekday() >= 5:
            continue

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.debug(
                "OCC: %d consecutive failures, stopping at %s",
                consecutive_failures, check_date.isoformat(),
            )
            break

        date_str = check_date.strftime("%Y%m%d")
        url = OCC_URL_TEMPLATE.format(date=date_str)

        try:
            occ_rate_limit(rate_state)
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                text = resp.text.strip()
                if "does not exist" in text.lower():
                    consecutive_failures += 1
                    continue
                consecutive_failures = 0
                for line in text.splitlines():
                    parts = line.split("|")
                    if len(parts) >= 4 and parts[0].strip() == symbol:
                        market = parts[2].strip() or "OCC"
                        records.append(ThresholdRecord(
                            date=check_date.isoformat(),
                            symbol=symbol,
                            market=market,
                            threshold_shares=0,
                            consecutive_days=0,
                        ))
                        break
            elif resp.status_code in (404, 204):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except requests.RequestException:
            consecutive_failures += 1

    if records:
        logger.info(
            "OCC Reg SHO: %d threshold dates for %s",
            len(records), symbol,
        )
    return records


def query_nasdaq(
    symbol: str,
    lookback_days: int = 5500,
    session: requests.Session | None = None,
    rate_limit_fn: Callable[[], None] | None = None,
) -> list[ThresholdRecord]:
    """Check Nasdaq daily threshold list text files.

    Nasdaq publishes pipe-delimited threshold lists at a predictable URL
    pattern.  Available from ~2006 onward but only covers Nasdaq-listed
    securities (market categories Q, G, S).
    """
    from stockdownloader.data.regsho_threshold_client import ThresholdRecord

    if session is None:
        session = requests.Session()

    records: list[ThresholdRecord] = []
    today = date.today()
    consecutive_failures = 0

    for offset in range(lookback_days):
        check_date = today - timedelta(days=offset)
        if check_date.weekday() >= 5:
            continue

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Too many consecutive Nasdaq failures, stopping at %s",
                check_date.isoformat(),
            )
            break

        date_str = check_date.strftime("%Y%m%d")
        url = NASDAQ_URL_TEMPLATE.format(date=date_str)

        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=10)
            if resp.status_code == 200 and "text" in resp.headers.get(
                "Content-Type", ""
            ):
                consecutive_failures = 0
                for line in resp.text.splitlines():
                    parts = line.split("|")
                    if len(parts) >= 2 and parts[0].strip() == symbol:
                        market = "NASDAQ"
                        if len(parts) >= 3:
                            cat = parts[2].strip()
                            if cat in ("Q", "G", "S"):
                                market = f"NASDAQ ({cat})"
                        records.append(ThresholdRecord(
                            date=check_date.isoformat(),
                            symbol=symbol,
                            market=market,
                            threshold_shares=0,
                            consecutive_days=0,
                        ))
                        break
            elif resp.status_code in (302, 404):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except requests.RequestException:
            consecutive_failures += 1

    if records:
        logger.info(
            "Nasdaq Reg SHO: %d threshold dates for %s",
            len(records), symbol,
        )
    return records


def query_cboe(
    symbol: str,
    lookback_days: int = 5500,
    session: requests.Session | None = None,
) -> list[ThresholdRecord]:
    """Query CBOE BZX daily threshold list.

    CBOE publishes pipe-delimited threshold lists for BZX-listed
    securities (mostly leveraged/buffer ETFs).  Available from ~2015
    onward.
    """
    from stockdownloader.data.regsho_threshold_client import ThresholdRecord

    if session is None:
        session = requests.Session()

    records: list[ThresholdRecord] = []
    today = date.today()
    consecutive_failures = 0
    rate_state: dict = {"last_request_time": 0.0}

    earliest = date(2015, 1, 1)

    for offset in range(lookback_days):
        check_date = today - timedelta(days=offset)
        if check_date < earliest:
            break
        if check_date.weekday() >= 5:
            continue

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.debug(
                "CBOE: %d consecutive failures, stopping at %s",
                consecutive_failures, check_date.isoformat(),
            )
            break

        date_str = check_date.strftime("%Y-%m-%d")
        url = CBOE_URL_TEMPLATE.format(date=date_str)

        try:
            cboe_rate_limit(rate_state)
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                consecutive_failures = 0
                for line in resp.text.splitlines():
                    parts = line.split("|")
                    if len(parts) >= 2 and parts[0].strip() == symbol:
                        records.append(ThresholdRecord(
                            date=check_date.isoformat(),
                            symbol=symbol,
                            market="CBOE BZX",
                            threshold_shares=0,
                            consecutive_days=0,
                        ))
                        break
            elif resp.status_code in (404, 204):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except requests.RequestException:
            consecutive_failures += 1

    if records:
        logger.info(
            "CBOE Reg SHO: %d threshold dates for %s",
            len(records), symbol,
        )
    return records
