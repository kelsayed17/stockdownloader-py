"""Parses 13F institutional ownership from SEC EDGAR.

**Primary approach**: Downloads SEC Form 13F bulk data sets (TSV) from
``sec.gov/files/structureddata/data/form-13f-data-sets/``.  These contain
every 13F-HR filing for an entire quarter in flat TSV format, with one row
per holding.  Filtering by CUSIP gives complete institutional ownership.

**Fallback** (for the most recent quarter whose bulk file may not yet
exist): Uses the EDGAR full-text search API (``efts.sec.gov``) with
per-quarter date windowing to find individual 13F-HR filings, then
downloads and parses each filing's XML.

Rate-limited to 10 req/s (SEC fair-use policy).  Results are cached
locally as JSON.

Usage::

    client = SecOwnershipClient()
    snapshots = client.fetch_ownership_snapshots("GME", cusip="36467W109")
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

from stockdownloader.data.base_client import BaseDataClient
from stockdownloader.data.sec import common as sec_common
from stockdownloader.data.sec import ownership_parser as parsers
from stockdownloader.data.sec.common import SplitAdjustment, splits_for_symbol
from stockdownloader.core.models.regulatory import (
    InstitutionalHolding,
    OwnershipSnapshot,
)

logger = logging.getLogger(__name__)

# EDGAR full-text search index (fallback for current quarter)
_EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Direct archive access for filing documents (EFTS fallback)
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


class SecOwnershipClient(BaseDataClient):
    """Parses 13F institutional ownership from SEC EDGAR.

    Uses SEC bulk 13F data sets (TSV) as the primary, authoritative source.
    Falls back to EFTS full-text search for the most recent quarter if the
    bulk file is not yet available.
    """

    def __init__(
        self,
        user_agent: str = "StockDownloader admin@example.com",
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            rate_limit_delay=0.11,
            max_retries=3,
            data_dir=data_dir,
            default_headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        # Bulk ZIP files are multi-ticker, kept under cache/
        self._bulk_dir = self._data_dir / "cache" / "bulk_13f"
        self._bulk_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ownership_snapshots(
        self,
        symbol: str,
        cusip: str = parsers.GME_CUSIP,
        num_quarters: int = 100,
        *,
        force_refresh: bool = False,
        extra_splits: list[SplitAdjustment] | None = None,
    ) -> list[OwnershipSnapshot]:
        """Fetch institutional ownership snapshots for *symbol*.

        Downloads SEC bulk 13F data sets for each quarter, filters by
        CUSIP, and aggregates into quarterly :class:`OwnershipSnapshot`
        objects.

        Parameters
        ----------
        symbol:
            Ticker symbol (used for caching and labeling).
        cusip:
            CUSIP identifier to filter 13F holdings.
        num_quarters:
            Maximum number of quarterly snapshots to return.
        force_refresh:
            If ``True``, ignore the JSON cache and re-download.
        extra_splits:
            Additional stock splits to apply.  Well-known splits
            (see :data:`_KNOWN_SPLITS` in ``sec_common``) are
            applied automatically.

        Returns
        -------
        List of :class:`OwnershipSnapshot` sorted by ``quarter_end``
        ascending.  Share counts are split-adjusted so that pre-split
        quarters are comparable to post-split quarters.
        """
        symbol_upper = symbol.upper()

        # Auto-resolve CUSIP from the symbol registry when the caller
        # used the default GME CUSIP.  This fixes a bug where non-GME
        # symbols silently queried with GME's CUSIP.
        from stockdownloader.core.models.symbol import get_symbol_info

        _info = get_symbol_info(symbol_upper)
        if cusip == parsers.GME_CUSIP and symbol_upper != "GME":
            if _info is not None and _info.cusip:
                cusip = _info.cusip
                logger.info(
                    "Auto-resolved CUSIP for %s: %s",
                    symbol_upper, cusip,
                )
            else:
                logger.warning(
                    "No CUSIP found for %s in symbol registry; "
                    "using default GME CUSIP — results will be "
                    "incorrect.  Pass cusip= explicitly.",
                    symbol_upper,
                )

        # Try cache first (unless force refresh)
        if not force_refresh:
            cached = self._load_cache(symbol_upper)
            if cached is not None:
                logger.info(
                    "Using cached ownership data for %s (%d snapshots)",
                    symbol_upper, len(cached),
                )
                return cached[-num_quarters:]
        else:
            # Delete only the merged ownership_13f.json; preserve per-quarter snapshots
            merged = self._symbol_dir(symbol_upper) / "ownership_13f.json"
            if merged.exists():
                merged.unlink()
                logger.info(
                    "force_refresh: deleted merged cache %s", merged,
                )

        min_y, min_q = sec_common.ipo_quarter_floor(symbol_upper)

        snapshots: list[OwnershipSnapshot] = []

        for year, quarter in sec_common.quarter_iterator(
            num_quarters, min_year=min_y, min_quarter=min_q,
        ):
            quarter_end_str = f"{year}-{parsers.QUARTER_ENDS[quarter]}"

            # Check per-quarter cache first (covers both bulk and EFTS)
            snap = self._load_quarter_snapshot(symbol_upper, year, quarter)
            if snap is not None:
                logger.info(
                    "Using cached snapshot for %s Q%d %d",
                    symbol_upper, quarter, year,
                )
                snapshots.append(snap)
                continue

            logger.info(
                "Fetching 13F data for %s Q%d %d (%s)",
                symbol_upper, quarter, year, quarter_end_str,
            )

            # Try bulk data set first (primary approach)
            snap = self._fetch_from_bulk(
                symbol_upper, cusip, year, quarter, quarter_end_str,
            )

            # Fallback to EFTS search for recent quarters
            if snap is None:
                logger.info(
                    "Bulk data not available for Q%d %d, "
                    "trying EFTS fallback...", quarter, year,
                )
                snap = self._fetch_from_efts(
                    symbol_upper, cusip, year, quarter, quarter_end_str,
                )

            if snap is not None:
                snapshots.append(snap)
                # Save per-quarter snapshot so both bulk and EFTS results
                # are cached individually in the symbol subdir.
                self._save_quarter_snapshot(
                    symbol_upper, year, quarter, snap,
                )

        # Apply split adjustments so pre-split share counts are
        # comparable to post-split counts.  A stock may have multiple
        # historical splits (e.g. TSLA 5:1 in 2020 and 3:1 in 2022).
        symbol_splits = splits_for_symbol(symbol_upper, extra_splits)
        for split in symbol_splits:
            snapshots = [
                parsers.apply_split_to_snapshot(snap, split)
                for snap in snapshots
            ]

        snapshots.sort(key=lambda s: s.quarter_end)

        if snapshots:
            self._save_cache(symbol_upper, snapshots)

        return snapshots[-num_quarters:]

    # ------------------------------------------------------------------
    # Primary: SEC Bulk 13F Data Sets
    # ------------------------------------------------------------------

    def _fetch_from_bulk(
        self,
        symbol: str,
        cusip: str,
        year: int,
        quarter: int,
        quarter_end: str,
    ) -> OwnershipSnapshot | None:
        """Download and parse a SEC bulk 13F data set for one quarter.

        Downloads the ZIP file, extracts INFOTABLE.tsv and SUBMISSION.tsv,
        filters by CUSIP, and aggregates into an OwnershipSnapshot.
        """
        zip_path = self._bulk_dir / f"{year}Q{quarter}_form13f.zip"

        # Download if not cached
        if not zip_path.exists():
            urls = parsers.bulk_zip_urls(year, quarter)
            zip_data = None
            for url in urls:
                logger.info("Trying bulk 13F URL: %s", url)
                zip_data = sec_common.download_with_retry(
                    self._session, url, rate_limit_fn=self._rate_limit,
                )
                if zip_data is not None:
                    break
            if zip_data is None:
                return None
            try:
                zip_path.write_bytes(zip_data)
            except OSError as exc:
                logger.warning("Failed to cache ZIP: %s", exc)
                # Continue with in-memory data
                return parsers.parse_bulk_zip(
                    io.BytesIO(zip_data), symbol, cusip, quarter_end,
                )

        # Parse the cached ZIP
        try:
            return parsers.parse_bulk_zip(
                zip_path, symbol, cusip, quarter_end,
            )
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning(
                "Failed to parse bulk ZIP %s: %s", zip_path, exc,
            )
            # Delete corrupted file and retry next time
            zip_path.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Fallback: EDGAR Full-Text Search (EFTS) for current quarter
    # ------------------------------------------------------------------

    def _fetch_from_efts(
        self,
        symbol: str,
        cusip: str,
        year: int,
        quarter: int,
        quarter_end: str,
    ) -> OwnershipSnapshot | None:
        """Fetch 13F data via EFTS for a single quarter (fallback).

        Uses date-range windowing to stay under the 10,000-hit limit.
        """
        # Check per-quarter cache first
        cached = self._load_quarter_snapshot(symbol, year, quarter)
        if cached is not None:
            logger.info(
                "Using cached EFTS quarter snapshot for %s %dQ%d",
                symbol, year, quarter,
            )
            return cached

        # 13F filings for a quarter are filed 0-45 days after quarter end.
        _, q_end = parsers.quarter_range(year, quarter)
        search_start = q_end + timedelta(days=1)
        search_end = q_end + timedelta(days=60)

        start_str = search_start.strftime("%Y-%m-%d")
        end_str = search_end.strftime("%Y-%m-%d")

        base_url = (
            f"{_EFTS_SEARCH_URL}?q=%22{cusip}%22&forms=13F-HR"
            f"&startdt={start_str}&enddt={end_str}"
        )

        all_hits = sec_common.fetch_all_efts_hits(
            self._session, base_url, rate_limit_fn=self._rate_limit,
        )

        if not all_hits:
            return None

        logger.info(
            "EFTS returned %d hits for CUSIP %s Q%d %d",
            len(all_hits), cusip, quarter, year,
        )

        # Download and parse each filing's infotable (XML or text)
        holdings: list[InstitutionalHolding] = []

        for hit in all_hits:
            source = hit.get("_source", {})
            ciks = source.get("ciks", [])
            cik = ciks[0].lstrip("0") if ciks else ""
            accession = source.get("adsh", "")
            display_names = source.get("display_names", [])
            manager_name = display_names[0] if display_names else ""

            hit_id = hit.get("_id", "")
            xml_filename = ""
            if ":" in hit_id:
                _, xml_filename = hit_id.split(":", 1)

            if not accession or not cik:
                continue

            acc_nodash = accession.replace("-", "")
            xml_url = ""
            if xml_filename:
                xml_url = (
                    f"{_ARCHIVE_BASE}/{cik}/{acc_nodash}/{xml_filename}"
                )

            if not xml_url:
                # Try index.json fallback
                index_url = f"{_ARCHIVE_BASE}/{cik}/{acc_nodash}/"
                xml_url = parsers.find_infotable_url(
                    self._session, index_url,
                    rate_limit_fn=self._rate_limit,
                ) or ""

            if not xml_url:
                continue

            doc_content = sec_common.fetch_url_text(
                self._session, xml_url, rate_limit_fn=self._rate_limit,
            )
            if doc_content is None:
                continue

            # Try XML first (post-2013), then text fallback (pre-2013).
            parsed = parsers.parse_13f_xml(doc_content, cusip)
            if not parsed:
                parsed = parsers.parse_13f_text(doc_content, cusip)
            for h in parsed:
                # Override manager name with display name from EFTS
                try:
                    holdings.append(InstitutionalHolding(
                        filing_date=quarter_end,
                        manager_name=manager_name or h.manager_name,
                        manager_cik=cik,
                        shares=h.shares,
                        value_usd=h.value_usd,
                        share_class=h.share_class,
                    ))
                except ValueError:
                    continue

        if not holdings:
            return None

        # Aggregate
        total_shares = sum(h.shares for h in holdings)
        sorted_h = sorted(holdings, key=lambda h: h.shares, reverse=True)
        top10 = sum(h.shares for h in sorted_h[:10])
        top10_conc = top10 / total_shares if total_shares > 0 else 0.0

        try:
            snapshot = OwnershipSnapshot(
                quarter_end=quarter_end,
                symbol=symbol,
                total_institutional_shares=total_shares,
                num_institutions=len(holdings),
                top_10_concentration=top10_conc,
                holdings=tuple(sorted_h),
            )
            self._save_quarter_snapshot(symbol, year, quarter, snapshot)
            return snapshot
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self, symbol: str) -> list[OwnershipSnapshot] | None:
        """Load cached ownership snapshots for *symbol*."""
        sym_dir = self._symbol_dir(symbol)
        cache_file = sym_dir / "ownership_13f.json"

        # Legacy migration from old cache paths
        if not cache_file.exists():
            for legacy_path in (
                self._data_dir / "cache" / "ownership" / symbol.upper() / "13f.json",
                self._data_dir / "cache" / "ownership" / f"{symbol}_13f.json",
                self._data_dir / "cache" / "ownership" / f"{symbol}_ownership.json",
                sym_dir / "13f.json",
            ):
                if legacy_path.exists():
                    legacy_path.rename(cache_file)
                    logger.info("Migrated %s -> %s", legacy_path, cache_file)
                    break

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            snapshots: list[OwnershipSnapshot] = []
            for snap in data:
                holdings = tuple(
                    InstitutionalHolding(
                        filing_date=h["filing_date"],
                        manager_name=h["manager_name"],
                        manager_cik=h["manager_cik"],
                        shares=h["shares"],
                        value_usd=h["value_usd"],
                        share_class=h["share_class"],
                    )
                    for h in snap.get("holdings", [])
                )
                snapshots.append(OwnershipSnapshot(
                    quarter_end=snap["quarter_end"],
                    symbol=snap["symbol"],
                    total_institutional_shares=snap[
                        "total_institutional_shares"
                    ],
                    num_institutions=snap["num_institutions"],
                    top_10_concentration=snap["top_10_concentration"],
                    holdings=holdings,
                ))
            snapshots.sort(key=lambda s: s.quarter_end)
            return snapshots
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load ownership cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self,
        symbol: str,
        snapshots: list[OwnershipSnapshot],
    ) -> None:
        """Persist ownership snapshots to JSON cache."""
        cache_file = self._symbol_dir(symbol) / "ownership_13f.json"
        data = [
            {
                "quarter_end": s.quarter_end,
                "symbol": s.symbol,
                "total_institutional_shares": s.total_institutional_shares,
                "num_institutions": s.num_institutions,
                "top_10_concentration": s.top_10_concentration,
                "holdings": [
                    {
                        "filing_date": h.filing_date,
                        "manager_name": h.manager_name,
                        "manager_cik": h.manager_cik,
                        "shares": h.shares,
                        "value_usd": h.value_usd,
                        "share_class": h.share_class,
                    }
                    for h in s.holdings
                ],
            }
            for s in snapshots
        ]
        try:
            cache_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save ownership cache for %s: %s", symbol, exc,
            )

    def _save_quarter_snapshot(
        self,
        symbol: str,
        year: int,
        quarter: int,
        snapshot: OwnershipSnapshot,
    ) -> None:
        """Persist a single quarter snapshot as ``{SYMBOL}/.progress/ownership/{YYYY}Q{Q}.json``."""
        quarter_dir = self._progress_dir(symbol) / "ownership"
        quarter_dir.mkdir(parents=True, exist_ok=True)
        cache_file = quarter_dir / f"{year}Q{quarter}.json"
        data = {
            "quarter_end": snapshot.quarter_end,
            "symbol": snapshot.symbol,
            "total_institutional_shares": snapshot.total_institutional_shares,
            "num_institutions": snapshot.num_institutions,
            "top_10_concentration": snapshot.top_10_concentration,
            "holdings": [
                {
                    "filing_date": h.filing_date,
                    "manager_name": h.manager_name,
                    "manager_cik": h.manager_cik,
                    "shares": h.shares,
                    "value_usd": h.value_usd,
                    "share_class": h.share_class,
                }
                for h in snapshot.holdings
            ],
        }
        try:
            cache_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save quarter snapshot %dQ%d for %s: %s",
                year, quarter, symbol, exc,
            )

    def _load_quarter_snapshot(
        self,
        symbol: str,
        year: int,
        quarter: int,
    ) -> OwnershipSnapshot | None:
        """Load a single quarter snapshot from ``{SYMBOL}/.progress/ownership/{YYYY}Q{Q}.json``."""
        quarter_dir = self._progress_dir(symbol) / "ownership"
        cache_file = quarter_dir / f"{year}Q{quarter}.json"

        # Legacy migration: move from old locations
        if not cache_file.exists():
            for legacy in (
                self._symbol_dir(symbol) / "ownership" / f"{year}Q{quarter}.json",
                self._data_dir / "cache" / "ownership" / symbol.upper() / f"{year}Q{quarter}.json",
            ):
                if legacy.exists():
                    quarter_dir.mkdir(parents=True, exist_ok=True)
                    legacy.rename(cache_file)
                    logger.info("Migrated %s -> %s", legacy, cache_file)
                    break

        if not cache_file.exists():
            return None
        try:
            snap = json.loads(cache_file.read_text(encoding="utf-8"))
            holdings = tuple(
                InstitutionalHolding(
                    filing_date=h["filing_date"],
                    manager_name=h["manager_name"],
                    manager_cik=h["manager_cik"],
                    shares=h["shares"],
                    value_usd=h["value_usd"],
                    share_class=h["share_class"],
                )
                for h in snap.get("holdings", [])
            )
            return OwnershipSnapshot(
                quarter_end=snap["quarter_end"],
                symbol=snap["symbol"],
                total_institutional_shares=snap[
                    "total_institutional_shares"
                ],
                num_institutions=snap["num_institutions"],
                top_10_concentration=snap["top_10_concentration"],
                holdings=holdings,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load quarter snapshot %dQ%d for %s: %s",
                year, quarter, symbol, exc,
            )
            return None
