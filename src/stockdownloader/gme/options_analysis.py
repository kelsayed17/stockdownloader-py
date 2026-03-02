#!/usr/bin/env python3
"""GME options open interest analysis.

Loads OCC open interest data and cross-references with FTD and short
volume data to detect synthetic short position patterns:

1. Total OI (in shares) vs float and outstanding
2. Daily OI changes and spikes
3. Expirations with anomalous OI concentration
4. Correlation with FTD and short volume spikes

Usage:
    python3 -m stockdownloader.gme.options_analysis
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
SHARES_OUTSTANDING = 69_750_000  # SEC filing, Jan 30, 2021
FREE_FLOAT = 50_650_000  # Consistent with SEC "~140%" statement
CONTRACT_MULTIPLIER = 100  # Each options contract = 100 shares


def _load_csv_records(path: Path, int_fields: tuple = (), float_fields: tuple = ()) -> list[dict]:
    """Load records from a CSV file, converting numeric fields."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k in int_fields:
                if k in row:
                    row[k] = int(row[k]) if row[k] else 0
            for k in float_fields:
                if k in row:
                    row[k] = float(row[k]) if row[k] else 0.0
            records.append(row)
    return records


def load_occ_data(symbol: str = "GME") -> list[dict]:
    """Load OCC open interest records."""
    csv_file = DATA_DIR / symbol / "occ_open_interest.csv"
    json_file = DATA_DIR / symbol / "occ_open_interest.json"
    if csv_file.exists():
        return _load_csv_records(
            csv_file,
            int_fields=("volume", "exercised", "open_interest"),
        )
    if json_file.exists():
        return json.loads(json_file.read_text(encoding="utf-8"))
    logger.error("OCC data not found: %s", csv_file)
    logger.info(
        "Run the OCC backfill first:\n"
        "  python3 -c \"\n"
        "from stockdownloader.data.market.occ_client import OccOptionsClient\n"
        "c = OccOptionsClient()\n"
        "records = c.fetch_open_interest('GME')\n"
        "print(f'Records: {len(records)}')\n"
        "\""
    )
    return []


def load_ftd_data(symbol: str = "GME") -> list[dict]:
    """Load FTD data."""
    ftd_file = DATA_DIR / symbol / "ftd_data.json"
    if not ftd_file.exists():
        return []
    return json.loads(ftd_file.read_text(encoding="utf-8"))


def load_short_volume(symbol: str = "GME") -> list[dict]:
    """Load short volume data."""
    csv_file = DATA_DIR / symbol / "short_volume.csv"
    json_file = DATA_DIR / symbol / "short_volume.json"
    if csv_file.exists():
        return _load_csv_records(
            csv_file,
            int_fields=("short_volume", "total_volume", "short_exempt_volume"),
            float_fields=("short_volume_ratio",),
        )
    if json_file.exists():
        return json.loads(json_file.read_text(encoding="utf-8"))
    return []


def analyze_daily_oi(records: list[dict]) -> dict[str, dict]:
    """Aggregate daily OI across all exchanges and expirations.

    Returns {date: {total_oi_contracts, total_oi_shares, total_volume,
                    num_expirations, num_exchanges}}.
    """
    daily: dict[str, dict] = defaultdict(lambda: {
        "total_oi_contracts": 0,
        "total_oi_shares": 0,
        "total_volume": 0,
        "total_exercised": 0,
        "num_expirations": 0,
        "num_exchanges": 0,
        "expirations": set(),
        "exchanges": set(),
    })

    for r in records:
        d = daily[r["date"]]
        d["total_oi_contracts"] += r["open_interest"]
        d["total_oi_shares"] += r["open_interest"] * CONTRACT_MULTIPLIER
        d["total_volume"] += r["volume"]
        d["total_exercised"] += r["exercised"]
        d["expirations"].add(r["expiration"])
        d["exchanges"].add(r["exchange"])

    # Finalize counts
    for d in daily.values():
        d["num_expirations"] = len(d["expirations"])
        d["num_exchanges"] = len(d["exchanges"])
        del d["expirations"]
        del d["exchanges"]

    return dict(sorted(daily.items()))


def print_summary(daily_oi: dict[str, dict]) -> None:
    """Print summary statistics."""
    if not daily_oi:
        logger.info("No OI data to analyze.")
        return

    dates = sorted(daily_oi.keys())
    logger.info("=" * 70)
    logger.info("GME OPTIONS OPEN INTEREST ANALYSIS")
    logger.info("=" * 70)
    logger.info("Date range: %s to %s (%d trading days)", dates[0], dates[-1], len(dates))
    logger.info("")

    # Overall stats
    oi_shares = [d["total_oi_shares"] for d in daily_oi.values()]
    avg_oi = sum(oi_shares) / len(oi_shares)
    max_oi = max(oi_shares)
    max_oi_date = dates[oi_shares.index(max_oi)]
    latest_oi = oi_shares[-1]

    logger.info("--- OPEN INTEREST (in equivalent shares) ---")
    logger.info("Average daily OI:    %12s shares (%5.1f%% of float)",
                f"{avg_oi:,.0f}", avg_oi / FREE_FLOAT * 100)
    logger.info("Peak daily OI:       %12s shares (%5.1f%% of float) on %s",
                f"{max_oi:,.0f}", max_oi / FREE_FLOAT * 100, max_oi_date)
    logger.info("Latest daily OI:     %12s shares (%5.1f%% of float)",
                f"{latest_oi:,.0f}", latest_oi / FREE_FLOAT * 100)
    logger.info("Shares outstanding:  %12s", f"{SHARES_OUTSTANDING:,}")
    logger.info("Free float:          %12s", f"{FREE_FLOAT:,}")
    logger.info("")

    # Top 10 peak OI days
    ranked = sorted(daily_oi.items(), key=lambda x: x[1]["total_oi_shares"], reverse=True)
    logger.info("--- TOP 10 PEAK OI DAYS ---")
    logger.info("%-12s %15s %10s %10s", "Date", "OI (shares)", "% Float", "Volume")
    for dt, d in ranked[:10]:
        logger.info(
            "%-12s %15s %9.1f%% %10s",
            dt,
            f"{d['total_oi_shares']:,}",
            d["total_oi_shares"] / FREE_FLOAT * 100,
            f"{d['total_volume']:,}",
        )
    logger.info("")

    # Detect large day-over-day spikes
    logger.info("--- LARGEST DAILY OI CHANGES ---")
    changes = []
    for i in range(1, len(dates)):
        prev_oi = daily_oi[dates[i - 1]]["total_oi_shares"]
        curr_oi = daily_oi[dates[i]]["total_oi_shares"]
        change = curr_oi - prev_oi
        pct_change = change / prev_oi * 100 if prev_oi > 0 else 0
        changes.append((dates[i], change, pct_change, curr_oi))

    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    logger.info("%-12s %15s %10s %15s", "Date", "Change", "% Change", "New OI")
    for dt, chg, pct, new_oi in changes[:10]:
        logger.info(
            "%-12s %+14s %+9.1f%% %15s",
            dt, f"{chg:,}", pct, f"{new_oi:,}",
        )


def main() -> None:
    records = load_occ_data()
    if not records:
        return

    logger.info("Loaded %d OCC open interest records for GME", len(records))

    daily_oi = analyze_daily_oi(records)
    print_summary(daily_oi)


if __name__ == "__main__":
    main()
