#!/usr/bin/env python3
"""FTD-targeted RegSHO threshold backfill for GME.

Instead of scanning every weekday from 2010-present (~4,200 dates),
this script uses FTD data to identify periods where GME likely
qualified for the threshold list (FTDs >= 10,000 for 5+ consecutive
settlement days) and queries only those dates.

This reduces the number of NYSE API calls from ~4,200 to ~1,500,
and with progress tracking, only queries dates not yet checked.

Usage:
    python3 scripts/regsho_targeted_backfill.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Set up logging BEFORE imports so background tasks show output
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.data.regsho_threshold_client import RegShoThresholdClient


def load_ftd_dates(data_dir: Path, symbol: str, min_ftd: int = 10_000, min_streak: int = 5) -> list[str]:
    """Load FTD data and find dates within qualifying threshold periods.

    A qualifying period = min_streak consecutive settlement days with FTDs >= min_ftd.
    Returns all dates within those periods, plus a 5-day buffer on each side
    (threshold list appearance lags FTD reporting by T+2).
    """
    ftd_file = data_dir / symbol / "ftd_data.json"
    if not ftd_file.exists():
        logger.error("FTD data not found: %s", ftd_file)
        return []

    ftd_data = json.loads(ftd_file.read_text(encoding="utf-8"))
    logger.info("Loaded %d FTD records for %s", len(ftd_data), symbol)

    # Build a set of dates with high FTDs
    # FTD records use "settlement_date" as key
    high_ftd_dates: set[str] = set()
    for record in ftd_data:
        qty = record.get("quantity", 0)
        date_key = record.get("settlement_date") or record.get("date", "")
        if qty >= min_ftd and date_key:
            high_ftd_dates.add(date_key)

    logger.info("Dates with FTDs >= %d: %d", min_ftd, len(high_ftd_dates))

    # Sort dates and find streaks of min_streak+ consecutive settlement days
    sorted_dates = sorted(high_ftd_dates)
    qualifying_dates: set[str] = set()

    # Group into streaks (consecutive business days)
    streaks: list[list[str]] = []
    current_streak: list[str] = []

    for date_str in sorted_dates:
        if not current_streak:
            current_streak = [date_str]
            continue

        # Check if this date is the next business day after the last
        prev = datetime.strptime(current_streak[-1], "%Y-%m-%d").date()
        curr = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Allow up to 4 calendar days gap (Fri->Mon = 3 days, Thu->Mon with holiday = 4)
        gap = (curr - prev).days
        if gap <= 4:
            current_streak.append(date_str)
        else:
            if len(current_streak) >= min_streak:
                streaks.append(current_streak)
            current_streak = [date_str]

    if len(current_streak) >= min_streak:
        streaks.append(current_streak)

    logger.info("Found %d qualifying streaks of %d+ consecutive days", len(streaks), min_streak)

    # Collect all dates within qualifying streaks, plus a buffer
    # Buffer accounts for T+2 settlement lag and threshold list publication delay
    buffer_days = 7  # calendar days

    for streak in streaks:
        start = datetime.strptime(streak[0], "%Y-%m-%d").date() - timedelta(days=buffer_days)
        end = datetime.strptime(streak[-1], "%Y-%m-%d").date() + timedelta(days=buffer_days)

        current = start
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                # NYSE data starts from ~2010
                if current >= date(2010, 1, 1):
                    qualifying_dates.add(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

    result = sorted(qualifying_dates)
    logger.info(
        "Total target dates (with buffer): %d across %d qualifying periods",
        len(result), len(streaks),
    )
    return result


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    symbol = "GME"

    # Step 1: Generate target dates from FTD data
    logger.info("=" * 60)
    logger.info("FTD-Targeted RegSHO Threshold Backfill for %s", symbol)
    logger.info("=" * 60)

    target_dates = load_ftd_dates(data_dir, symbol)
    if not target_dates:
        logger.error("No target dates generated. Check FTD data.")
        return

    # Step 2: Check how many are already queried
    progress_file = data_dir / symbol / ".progress" / "regsho_nyse.json"
    legacy_progress = data_dir / symbol / "regsho_nyse_progress.json"
    already_queried: set[str] = set()
    if progress_file.exists():
        already_queried = set(json.loads(progress_file.read_text(encoding="utf-8")))
    elif legacy_progress.exists():
        already_queried = set(json.loads(legacy_progress.read_text(encoding="utf-8")))

    remaining = [d for d in target_dates if d not in already_queried]
    logger.info(
        "Target dates: %d total, %d already queried, %d remaining",
        len(target_dates), len(target_dates) - len(remaining), len(remaining),
    )

    if not remaining:
        logger.info("All target dates already queried!")
        # Load and display results
        threshold_csv = data_dir / symbol / "regsho_threshold.csv"
        threshold_json = data_dir / symbol / "regsho_threshold.json"
        if threshold_csv.exists():
            with threshold_csv.open(encoding="utf-8") as fh:
                records = list(csv.DictReader(fh))
            logger.info("Total threshold records: %d", len(records))
        elif threshold_json.exists():
            records = json.loads(threshold_json.read_text(encoding="utf-8"))
            logger.info("Total threshold records: %d", len(records))
        return

    # Estimate time
    # At ~2.5s per request + batch pauses every 25 requests
    est_minutes = (len(remaining) * 2.5 + (len(remaining) // 25) * 45) / 60
    logger.info("Estimated time: %.0f minutes for %d queries", est_minutes, len(remaining))

    # Step 3: Run targeted backfill
    client = RegShoThresholdClient(data_dir=str(data_dir))
    records = client.fetch_threshold_targeted(symbol, target_dates)

    # Step 4: Report results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("Total threshold records for %s: %d", symbol, len(records))

    if records:
        dates = [r.date if hasattr(r, 'date') else r['date'] for r in records]
        logger.info("Date range: %s to %s", min(dates), max(dates))


if __name__ == "__main__":
    main()
