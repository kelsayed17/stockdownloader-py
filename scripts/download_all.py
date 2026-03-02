#!/usr/bin/env python3
"""Download all data for GME and GMEWS, filling gaps.

Runs each client sequentially within API-source groups to respect rate
limits, but different API sources could be parallelized in the future.

Usage::

    python3 scripts/download_all.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import asdict, fields as dc_fields
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.core.models.symbol import get_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_all")

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")

# Polygon API key
POLYGON_API_KEY = "jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu"

# FINRA credentials
FINRA_CLIENT_ID = "a876a67c64314aae9afd"
FINRA_CLIENT_SECRET = "GA&NzO0eZFdn@TB"


def _save_records_csv(symbol: str, filename: str, records: list) -> None:
    """Save a list of dataclass records to CSV under data/{SYMBOL}/."""
    if not records:
        return
    sym_dir = Path(DATA_DIR) / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sym_dir / filename
    fieldnames = [f.name for f in dc_fields(records[0])]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info("  -> saved %d records to %s", len(records), csv_path)


def _fetch_with_aliases(
    symbol: str,
    fetch_fn,
    label: str,
    dedup_key=None,
    *,
    try_all: bool = True,
) -> list:
    """Try canonical ticker first, then aliases.

    Args:
        symbol: Canonical ticker (e.g. "GMEWS").
        fetch_fn: Callable(ticker) -> list of records.
        label: Human label for logging.
        dedup_key: Optional callable(record) -> hashable for deduplication.
            If None, no dedup — results are concatenated.
        try_all: If True (default), try every alias and merge/dedup all
            results.  If False, stop after the first alias that returns
            data (useful for slow sources where aliases are equivalent).

    Returns:
        Merged, deduplicated list of records from aliases that returned data.
    """
    tickers = get_all_tickers(symbol)
    all_records = []
    seen = set()
    for ticker in tickers:
        try:
            records = fetch_fn(ticker)
            if records:
                logger.info("  %s: %d records from ticker %r", label, len(records), ticker)
                for r in records:
                    if dedup_key is not None:
                        key = dedup_key(r)
                        if key in seen:
                            continue
                        seen.add(key)
                    all_records.append(r)
                if not try_all:
                    break
        except Exception as e:
            logger.warning("  %s: ticker %r failed: %s", label, ticker, e)
    return all_records


def run_finra(symbols: list[str]) -> None:
    """FINRA clients: short volume, short interest, dark pool."""
    from stockdownloader.data.finra.short_volume_client import FinraShortVolumeClient
    from stockdownloader.data.finra.short_interest_client import FinraShortInterestClient
    from stockdownloader.data.finra.dark_pool_client import FinraDarkPoolClient

    sv = FinraShortVolumeClient(data_dir=DATA_DIR)
    si = FinraShortInterestClient(data_dir=DATA_DIR)
    dp = FinraDarkPoolClient(data_dir=DATA_DIR)

    for sym in symbols:
        # Short volume CDN scan downloads ~1700 daily files (~8 min) and
        # filters by ticker.  Only worth running once per symbol — aliases
        # would be identical lines in the same files.  Use canonical only.
        logger.info("=== FINRA Short Volume: %s ===", sym)
        try:
            records = sv.fetch_short_volume(sym)
            if records:
                logger.info("  ShortVol: %d records from ticker %r", len(records), sym)
                _save_records_csv(sym, "short_volume.csv", records)
        except Exception as e:
            records = []
            logger.warning("  ShortVol: ticker %r failed: %s", sym, e)
        logger.info("  -> %d short volume records", len(records))

        # Short interest uses API only — fast, try all aliases and merge.
        logger.info("=== FINRA Short Interest: %s ===", sym)
        records = _fetch_with_aliases(
            sym, si.fetch_short_interest, "ShortInt",
            dedup_key=lambda r: r.settlement_date,
        )
        if records:
            _save_records_csv(sym, "short_interest.csv", records)
        logger.info("  -> %d short interest records", len(records))

        # Dark pool uses API — try all aliases and merge.
        logger.info("=== FINRA Dark Pool: %s ===", sym)
        records = _fetch_with_aliases(
            sym, dp.fetch_dark_pool_volume, "DarkPool",
            dedup_key=lambda r: r.week_ending,
        )
        if records:
            _save_records_csv(sym, "dark_pool.csv", records)
        logger.info("  -> %d dark pool records", len(records))


def run_sec(symbols: list[str]) -> None:
    """SEC clients: FTD, insider, ownership, filings."""
    from stockdownloader.data.sec.ftd_client import SecFtdClient
    from stockdownloader.data.sec.insider_client import SecInsiderClient
    from stockdownloader.data.sec.ownership_client import SecOwnershipClient
    from stockdownloader.data.sec.edgar_client import SecEdgarClient

    ftd = SecFtdClient(cache_dir=str(Path(DATA_DIR) / "cache" / "ftd"))
    insider = SecInsiderClient(data_dir=DATA_DIR)
    ownership = SecOwnershipClient(data_dir=DATA_DIR)
    edgar = SecEdgarClient()

    for sym in symbols:
        logger.info("=== SEC FTD: %s ===", sym)
        try:
            records = _fetch_with_aliases(
                sym, ftd.fetch_ftd_data, "FTD",
                dedup_key=lambda r: (r.settlement_date, r.quantity),
            )
            logger.info("  -> %d FTD records", len(records))
            if records:
                _save_records_csv(sym, "ftd_data.csv", records)
        except Exception as e:
            logger.error("  FAILED: %s", e)

        logger.info("=== SEC Insider Transactions: %s ===", sym)
        try:
            records = insider.fetch_insider_transactions(sym)
            logger.info("  -> %d insider transaction records", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)

        logger.info("=== SEC Beneficial Owners: %s ===", sym)
        try:
            records = insider.fetch_beneficial_owners(sym)
            logger.info("  -> %d beneficial owner records", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)

        logger.info("=== SEC Ownership (13F): %s ===", sym)
        try:
            records = ownership.fetch_ownership_snapshots(sym)
            logger.info("  -> %d ownership snapshots", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)

        logger.info("=== SEC EDGAR Filings: %s ===", sym)
        try:
            records = edgar.fetch_gme_filings() if sym == "GME" else edgar.fetch_filings(sym)
            logger.info("  -> %d filings", len(records))
            # SecEdgarClient returns records in-memory — save to CSV
            _save_records_csv(sym, "sec_filings.csv", records)
        except Exception as e:
            logger.error("  FAILED: %s", e)


def run_sec_ftd_only(symbols: list[str]) -> None:
    """Just FTD data for additional symbols (e.g. GMEWS)."""
    from stockdownloader.data.sec.ftd_client import SecFtdClient

    ftd = SecFtdClient(cache_dir=str(Path(DATA_DIR) / "cache" / "ftd"))
    for sym in symbols:
        logger.info("=== SEC FTD: %s ===", sym)
        try:
            records = _fetch_with_aliases(
                sym, ftd.fetch_ftd_data, "FTD",
                dedup_key=lambda r: (r.settlement_date, r.quantity),
            )
            logger.info("  -> %d FTD records", len(records))
            if records:
                _save_records_csv(sym, "ftd_data.csv", records)
        except Exception as e:
            logger.error("  FAILED: %s", e)


def run_occ(symbols: list[str]) -> None:
    """OCC open interest client."""
    from stockdownloader.data.market.occ_client import OccOptionsClient

    occ = OccOptionsClient(data_dir=DATA_DIR)
    for sym in symbols:
        logger.info("=== OCC Open Interest: %s ===", sym)
        try:
            records = occ.fetch_open_interest(sym)
            logger.info("  -> %d OCC OI records", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)


def run_regsho(symbols: list[str]) -> None:
    """RegSHO threshold client."""
    from stockdownloader.data.regsho.threshold_client import RegShoThresholdClient

    regsho = RegShoThresholdClient(data_dir=DATA_DIR)
    for sym in symbols:
        logger.info("=== RegSHO Threshold: %s ===", sym)
        try:
            records = regsho.fetch_threshold_status(sym)
            logger.info("  -> %d threshold records", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)


def _move_alias_data(alias: str, canonical: str) -> None:
    """Move data files created under an alias dir to the canonical dir."""
    alias_dir = Path(DATA_DIR) / alias.upper()
    canonical_dir = Path(DATA_DIR) / canonical.upper()
    if not alias_dir.exists() or alias_dir == canonical_dir:
        return
    canonical_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in alias_dir.iterdir():
        dest = canonical_dir / f.name
        shutil.move(str(f), str(dest))
        logger.info("  Moved %s -> %s", f, dest)
    alias_dir.rmdir()
    logger.info("  Removed alias dir %s", alias_dir)


def run_polygon(symbols: list[str]) -> None:
    """Polygon daily + intraday price data."""
    from stockdownloader.data.history_fetcher import FullHistoryFetcher

    fetcher = FullHistoryFetcher(cache_dir=str(Path(DATA_DIR) / "cache"))
    for sym in symbols:
        logger.info("=== Daily Bars (Yahoo/Polygon): %s ===", sym)
        try:
            records = fetcher.fetch_full_daily_history(sym)
            if not records:
                for alias in get_all_tickers(sym)[1:]:
                    logger.info("  Trying alias %r for daily bars", alias)
                    records = fetcher.fetch_full_daily_history(alias)
                    if records:
                        _move_alias_data(alias, sym)
                        break
            logger.info("  -> %d daily bars", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)

        logger.info("=== 5m Bars (Polygon): %s ===", sym)
        try:
            records = fetcher.fetch_intraday_history(sym)
            if not records:
                for alias in get_all_tickers(sym)[1:]:
                    logger.info("  Trying alias %r for 5m bars", alias)
                    records = fetcher.fetch_intraday_history(alias)
                    if records:
                        _move_alias_data(alias, sym)
                        break
            logger.info("  -> %d 5m bars", len(records))
        except Exception as e:
            logger.error("  FAILED: %s", e)


def main() -> None:
    start = time.time()

    os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY
    os.environ["FINRA_CLIENT_ID"] = FINRA_CLIENT_ID
    os.environ["FINRA_CLIENT_SECRET"] = FINRA_CLIENT_SECRET

    # GME is the primary symbol; GMEWS is a warrant variant
    gme_only = ["GME"]
    all_symbols = ["GME", "GMEWS"]

    logger.info("=" * 60)
    logger.info("STARTING FULL DATA DOWNLOAD")
    logger.info("=" * 60)

    # --- FINRA (GME + GMEWS) ---
    logger.info("\n>>> FINRA DATA (GME + GMEWS)")
    run_finra(all_symbols)

    # --- SEC (GME for all; GMEWS for FTD only) ---
    logger.info("\n>>> SEC DATA (GME full suite)")
    run_sec(gme_only)
    logger.info("\n>>> SEC FTD (GMEWS only)")
    run_sec_ftd_only(["GMEWS"])

    # --- OCC (GME only — warrants don't have listed options) ---
    logger.info("\n>>> OCC OPEN INTEREST (GME only)")
    run_occ(gme_only)

    # --- RegSHO (GME only) ---
    logger.info("\n>>> REGSHO THRESHOLD (GME only)")
    run_regsho(gme_only)

    # --- Price data (GME + GMEWS) ---
    logger.info("\n>>> PRICE DATA (GME + GMEWS)")
    run_polygon(all_symbols)

    # --- Cleanup: remove any stale alias directories created by clients ---
    from stockdownloader.core.models.symbol import get_symbol_info
    data_path = Path(DATA_DIR)
    for sym in all_symbols:
        info = get_symbol_info(sym)
        if info is not None:
            for alias in info.aliases:
                alias_dir = data_path / alias.upper()
                if alias_dir.exists() and alias_dir.is_dir():
                    import shutil
                    # Move any files to canonical, then remove
                    canonical_dir = data_path / sym.upper()
                    for f in alias_dir.iterdir():
                        dest = canonical_dir / f.name
                        if not dest.exists():
                            shutil.move(str(f), str(dest))
                    try:
                        shutil.rmtree(str(alias_dir))
                        logger.info("Cleaned up stale alias dir: %s", alias_dir)
                    except OSError:
                        pass

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("ALL DOWNLOADS COMPLETE in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
