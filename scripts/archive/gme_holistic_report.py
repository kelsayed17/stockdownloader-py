#!/usr/bin/env python3
"""
GME Holistic Data Report
========================
Fetches & aligns ALL GME alternative data sources on a common timeline:
- Price action (daily from Yahoo, 2002-present)
- SEC FTD data (2017-present, from cached zip files)
- FINRA Short Interest (bi-monthly, from cache)
- FINRA Dark Pool / ATS volume (weekly, from cache)
- SEC 13F Institutional Ownership (quarterly, from cache)
- Estimated Borrow Rates (derived from short interest)

Outputs:
1. Console summary with key statistics
2. CSV: data/gme_holistic_aligned.csv (all data on daily timeline)
3. JSON: data/gme_report_data.json (for TradingView indicator)
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockdownloader.data.sec_ftd_client import SecFtdClient
from stockdownloader.data.borrow_rate_proxy import BorrowRateProxy

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


def load_price_data():
    """Load GME daily price data from CSV."""
    csv_path = DATA_DIR / "gme_daily_bars.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run full_history_fetcher first.")
        sys.exit(1)

    prices = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get("Date", row.get("date", ""))
            # Normalize date to YYYY-MM-DD
            if "/" in date_str:
                dt = datetime.strptime(date_str, "%m/%d/%Y")
                date_str = dt.strftime("%Y-%m-%d")
            prices.append({
                "date": date_str,
                "open": float(row.get("Open", row.get("open", 0))),
                "high": float(row.get("High", row.get("high", 0))),
                "low": float(row.get("Low", row.get("low", 0))),
                "close": float(row.get("Close", row.get("close", 0))),
                "adj_close": float(row.get("Adj Close", row.get("adj_close", row.get("close", 0)))),
                "volume": int(float(row.get("Volume", row.get("volume", 0)))),
            })
    prices.sort(key=lambda x: x["date"])
    return prices


def load_ftd_data():
    """Load FTD data from cached SEC zip files."""
    ftd_client = SecFtdClient(cache_dir=str(CACHE_DIR / "ftd"))
    records = ftd_client.fetch_ftd_data("GME", start_year=2017)
    print(f"  FTD records loaded: {len(records)}")

    # Build daily FTD map (sum if multiple entries per day)
    ftd_by_date = defaultdict(lambda: {"quantity": 0, "value": 0.0})
    for r in records:
        d = r.settlement_date
        ftd_by_date[d]["quantity"] += r.quantity
        ftd_by_date[d]["value"] += float(r.quantity) * float(r.price)
    return dict(ftd_by_date)


def load_short_interest():
    """Load short interest from FINRA cache."""
    cache_path = CACHE_DIR / "short_interest" / "GME_si.json"
    if not cache_path.exists():
        print("  WARNING: No short interest cache found")
        return {}

    with open(cache_path) as f:
        records = json.load(f)

    si_by_date = {}
    for r in records:
        si_by_date[r["settlement_date"]] = {
            "short_interest": r["short_interest"],
            "avg_daily_volume": r["avg_daily_volume"],
            "days_to_cover": r["days_to_cover"],
            "short_interest_pct": r.get("short_interest_pct", 0.0),
        }
    print(f"  Short interest records loaded: {len(si_by_date)}")
    return si_by_date


def load_dark_pool():
    """Load dark pool / ATS data from FINRA cache."""
    cache_path = CACHE_DIR / "dark_pool" / "GME_dp.json"
    if not cache_path.exists():
        print("  WARNING: No dark pool cache found")
        return {}

    with open(cache_path) as f:
        records = json.load(f)

    dp_by_week = {}
    for r in records:
        dp_by_week[r["week_ending"]] = {
            "total_weekly_volume": r["total_weekly_volume"],
            "ats_volume": r["ats_volume"],
            "otc_volume": r["otc_volume"],
            "ats_pct": r["ats_pct"],
        }
    print(f"  Dark pool weekly records loaded: {len(dp_by_week)}")
    return dp_by_week


def load_ownership():
    """Load 13F institutional ownership from SEC cache."""
    # Check new filename first, then legacy
    cache_path = CACHE_DIR / "ownership" / "GME_13f.json"
    if not cache_path.exists():
        cache_path = CACHE_DIR / "ownership" / "GME_ownership.json"
    if not cache_path.exists():
        print("  WARNING: No ownership cache found")
        return {}

    print(f"  Loading ownership from: {cache_path.name}")
    with open(cache_path) as f:
        records = json.load(f)

    own_by_quarter = {}
    for r in records:
        own_by_quarter[r["quarter_end"]] = {
            "total_institutional_shares": r["total_institutional_shares"],
            "num_institutions": r["num_institutions"],
            "top_10_concentration": r["top_10_concentration"],
            "holdings": r.get("holdings", []),
        }
    print(f"  Ownership quarterly records loaded: {len(own_by_quarter)}")
    return own_by_quarter


def estimate_borrow_rates(si_by_date):
    """Estimate borrow rates from short interest data using DTC proxy."""
    # GME shares outstanding ~ 305M post-split (as of 2024)
    SHARES_OUTSTANDING = 305_000_000

    rates = {}
    for date, si in si_by_date.items():
        dtc = si["days_to_cover"]
        # Piecewise linear DTC → fee mapping
        if dtc < 1.0:
            fee = 0.25 + (dtc / 1.0) * 0.75
        elif dtc < 3.0:
            fee = 1.0 + ((dtc - 1.0) / 2.0) * 4.0
        elif dtc < 7.0:
            fee = 5.0 + ((dtc - 3.0) / 4.0) * 15.0
        elif dtc < 15.0:
            fee = 20.0 + ((dtc - 7.0) / 8.0) * 30.0
        else:
            fee = min(50.0 + (dtc - 15.0) * 3.0, 100.0)

        utilization = si["short_interest"] / SHARES_OUTSTANDING * 100.0
        rates[date] = {
            "estimated_fee_pct": round(fee, 2),
            "days_to_cover": dtc,
            "utilization_pct": round(utilization, 2),
        }
    print(f"  Borrow rate estimates computed: {len(rates)}")
    return rates


def align_all_data(prices, ftd_by_date, si_by_date, dp_by_week, own_by_quarter, borrow_rates):
    """
    Align all data sources on the daily price timeline with forward-fill.
    """
    # Sort dark pool weeks and ownership quarters for forward-fill
    dp_dates = sorted(dp_by_week.keys())
    own_dates = sorted(own_by_quarter.keys())
    si_dates = sorted(si_by_date.keys())

    # Current forward-filled values
    current_si = None
    current_dp = None
    current_own = None
    current_borrow = None

    si_idx = 0
    dp_idx = 0
    own_idx = 0

    aligned = []

    for price in prices:
        d = price["date"]

        # Forward-fill short interest
        while si_idx < len(si_dates) and si_dates[si_idx] <= d:
            current_si = si_by_date[si_dates[si_idx]]
            current_borrow = borrow_rates.get(si_dates[si_idx])
            si_idx += 1

        # Forward-fill dark pool
        while dp_idx < len(dp_dates) and dp_dates[dp_idx] <= d:
            current_dp = dp_by_week[dp_dates[dp_idx]]
            dp_idx += 1

        # Forward-fill ownership
        while own_idx < len(own_dates) and own_dates[own_idx] <= d:
            current_own = own_by_quarter[own_dates[own_idx]]
            own_idx += 1

        # FTD is daily — no forward fill, just lookup
        ftd = ftd_by_date.get(d, {"quantity": 0, "value": 0.0})

        row = {
            "date": d,
            # Price
            "open": price["open"],
            "high": price["high"],
            "low": price["low"],
            "close": price["close"],
            "volume": price["volume"],
            # FTD (daily, 0 if no failures)
            "ftd_quantity": ftd["quantity"],
            "ftd_value": round(ftd["value"], 2),
            # Short Interest (forward-filled)
            "short_interest": current_si["short_interest"] if current_si else 0,
            "si_days_to_cover": current_si["days_to_cover"] if current_si else 0,
            "si_avg_daily_vol": current_si["avg_daily_volume"] if current_si else 0,
            # Dark Pool (forward-filled)
            "dp_total_weekly_vol": current_dp["total_weekly_volume"] if current_dp else 0,
            "dp_ats_volume": current_dp["ats_volume"] if current_dp else 0,
            "dp_ats_pct": round(current_dp["ats_pct"] * 100, 2) if current_dp else 0,
            # Ownership (forward-filled)
            "institutional_shares": current_own["total_institutional_shares"] if current_own else 0,
            "num_institutions": current_own["num_institutions"] if current_own else 0,
            "top10_concentration": round(current_own["top_10_concentration"] * 100, 2) if current_own else 0,
            # Borrow Rate (forward-filled)
            "borrow_rate_est": current_borrow["estimated_fee_pct"] if current_borrow else 0,
            "borrow_utilization": current_borrow["utilization_pct"] if current_borrow else 0,
        }
        aligned.append(row)

    return aligned


def compute_derived_metrics(aligned):
    """Add rolling averages, z-scores, and regime indicators."""
    n = len(aligned)

    for i in range(n):
        row = aligned[i]

        # === Daily return ===
        if i > 0:
            prev_close = aligned[i - 1]["close"]
            row["daily_return_pct"] = round(
                (row["close"] - prev_close) / prev_close * 100, 4
            ) if prev_close > 0 else 0
        else:
            row["daily_return_pct"] = 0

        # === FTD rolling 20-day average ===
        window = [aligned[j]["ftd_quantity"] for j in range(max(0, i - 19), i + 1)]
        row["ftd_20d_avg"] = round(sum(window) / len(window))

        # === FTD rolling 5-day sum (T+2 settlement window) ===
        window5 = [aligned[j]["ftd_quantity"] for j in range(max(0, i - 4), i + 1)]
        row["ftd_5d_sum"] = sum(window5)

        # === Volume 20-day average ===
        vol_window = [aligned[j]["volume"] for j in range(max(0, i - 19), i + 1)]
        row["volume_20d_avg"] = round(sum(vol_window) / len(vol_window))

        # === Relative volume ===
        row["relative_volume"] = round(
            row["volume"] / row["volume_20d_avg"], 2
        ) if row["volume_20d_avg"] > 0 else 1.0

        # === FTD as % of daily volume ===
        row["ftd_pct_of_volume"] = round(
            row["ftd_quantity"] / row["volume"] * 100, 4
        ) if row["volume"] > 0 else 0

        # === 20-day realized volatility ===
        if i >= 19:
            returns = [aligned[j]["daily_return_pct"] for j in range(i - 19, i + 1)]
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            row["volatility_20d"] = round(var ** 0.5, 4)
        else:
            row["volatility_20d"] = 0

        # === Price moving averages ===
        if i >= 49:
            row["sma_50"] = round(
                sum(aligned[j]["close"] for j in range(i - 49, i + 1)) / 50, 4
            )
        else:
            row["sma_50"] = row["close"]

        if i >= 199:
            row["sma_200"] = round(
                sum(aligned[j]["close"] for j in range(i - 199, i + 1)) / 200, 4
            )
        else:
            row["sma_200"] = row["close"]

    return aligned


def print_summary_report(aligned, ftd_by_date, si_by_date, dp_by_week, own_by_quarter):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("  GME HOLISTIC DATA REPORT")
    print("=" * 80)

    # Date range
    first = aligned[0]["date"]
    last = aligned[-1]["date"]
    print(f"\n📊 Date Range: {first} → {last} ({len(aligned)} trading days)")

    # === PRICE ACTION SUMMARY ===
    print("\n" + "─" * 60)
    print("  PRICE ACTION SUMMARY")
    print("─" * 60)

    # Key price milestones
    closes = [(r["date"], r["close"]) for r in aligned]
    all_time_high = max(closes, key=lambda x: x[1])
    all_time_low = min(closes, key=lambda x: x[1])
    current = closes[-1]

    print(f"  Current Price:     ${current[1]:.2f} ({current[0]})")
    print(f"  All-Time High:     ${all_time_high[1]:.2f} ({all_time_high[0]})")
    print(f"  All-Time Low:      ${all_time_low[1]:.2f} ({all_time_low[0]})")

    # Key periods
    # Pre-sneeze (2020)
    pre_2020 = [r for r in aligned if "2020-01-01" <= r["date"] <= "2020-12-31"]
    if pre_2020:
        avg_2020 = sum(r["close"] for r in pre_2020) / len(pre_2020)
        print(f"  Avg Price 2020:    ${avg_2020:.2f}")

    # The Sneeze (Jan 2021)
    sneeze = [r for r in aligned if "2021-01-01" <= r["date"] <= "2021-02-05"]
    if sneeze:
        sneeze_high = max(sneeze, key=lambda x: x["high"])
        sneeze_vol = sum(r["volume"] for r in sneeze)
        print(f"  Sneeze Peak:       ${sneeze_high['high']:.2f} ({sneeze_high['date']})")
        print(f"  Sneeze Volume:     {sneeze_vol:,.0f} shares")

    # 2024 DFV return
    dfv_return = [r for r in aligned if "2024-05-10" <= r["date"] <= "2024-06-30"]
    if dfv_return:
        dfv_high = max(dfv_return, key=lambda x: x["high"])
        print(f"  DFV Return Peak:   ${dfv_high['high']:.2f} ({dfv_high['date']})")

    # === FTD ANALYSIS ===
    print("\n" + "─" * 60)
    print("  FAILURE-TO-DELIVER (FTD) ANALYSIS")
    print("─" * 60)

    ftd_rows = [r for r in aligned if r["ftd_quantity"] > 0]
    print(f"  Total FTD Days:    {len(ftd_rows)} / {len(aligned)} trading days")
    if ftd_rows:
        total_ftd = sum(r["ftd_quantity"] for r in ftd_rows)
        avg_ftd = total_ftd / len(ftd_rows)
        max_ftd = max(ftd_rows, key=lambda x: x["ftd_quantity"])
        print(f"  Total FTDs:        {total_ftd:,.0f} shares")
        print(f"  Avg FTD (when >0): {avg_ftd:,.0f} shares")
        print(f"  Peak FTD Day:      {max_ftd['ftd_quantity']:,.0f} ({max_ftd['date']}, close=${max_ftd['close']:.2f})")

        # Top 20 FTD days
        top_ftd = sorted(ftd_rows, key=lambda x: x["ftd_quantity"], reverse=True)[:20]
        print("\n  Top 20 FTD Days:")
        print(f"  {'Date':<14} {'FTDs':>12} {'Close':>10} {'Volume':>14} {'FTD%Vol':>8}")
        for r in top_ftd:
            ftd_pct = r["ftd_quantity"] / r["volume"] * 100 if r["volume"] > 0 else 0
            print(f"  {r['date']:<14} {r['ftd_quantity']:>12,} ${r['close']:>8.2f} {r['volume']:>14,} {ftd_pct:>7.2f}%")

    # === SHORT INTEREST ANALYSIS ===
    print("\n" + "─" * 60)
    print("  SHORT INTEREST ANALYSIS")
    print("─" * 60)

    si_sorted = sorted(si_by_date.items(), key=lambda x: x[0])
    if si_sorted:
        print(f"  Date Range:        {si_sorted[0][0]} → {si_sorted[-1][0]}")
        print(f"  Reports:           {len(si_sorted)}")
        latest_si = si_sorted[-1]
        print(f"  Latest SI:         {latest_si[1]['short_interest']:,} shares ({latest_si[0]})")
        print(f"  Latest DTC:        {latest_si[1]['days_to_cover']:.2f} days")

        max_si = max(si_sorted, key=lambda x: x[1]["short_interest"])
        print(f"  Peak SI:           {max_si[1]['short_interest']:,} shares ({max_si[0]})")
        max_dtc = max(si_sorted, key=lambda x: x[1]["days_to_cover"])
        print(f"  Peak DTC:          {max_dtc[1]['days_to_cover']:.2f} days ({max_dtc[0]})")

        print("\n  Short Interest Timeline:")
        print(f"  {'Date':<14} {'Short Interest':>16} {'DTC':>8} {'Avg Daily Vol':>14}")
        for date, si in si_sorted:
            print(f"  {date:<14} {si['short_interest']:>16,} {si['days_to_cover']:>8.2f} {si['avg_daily_volume']:>14,}")

    # === DARK POOL ANALYSIS ===
    print("\n" + "─" * 60)
    print("  DARK POOL / ATS ANALYSIS")
    print("─" * 60)

    dp_sorted = sorted(dp_by_week.items(), key=lambda x: x[0])
    if dp_sorted:
        print(f"  Date Range:        {dp_sorted[0][0]} → {dp_sorted[-1][0]}")
        print(f"  Weekly Reports:    {len(dp_sorted)}")

        total_dp_vol = sum(d[1]["total_weekly_volume"] for d in dp_sorted)
        total_ats = sum(d[1]["ats_volume"] for d in dp_sorted)
        avg_ats_pct = sum(d[1]["ats_pct"] for d in dp_sorted) / len(dp_sorted) * 100

        print(f"  Total OTC+ATS Vol: {total_dp_vol:,}")
        print(f"  Total ATS Vol:     {total_ats:,}")
        print(f"  Avg ATS %:         {avg_ats_pct:.1f}%")

        max_dp = max(dp_sorted, key=lambda x: x[1]["total_weekly_volume"])
        print(f"  Peak Week Volume:  {max_dp[1]['total_weekly_volume']:,} ({max_dp[0]})")
        max_ats = max(dp_sorted, key=lambda x: x[1]["ats_pct"])
        print(f"  Peak ATS %:        {max_ats[1]['ats_pct'] * 100:.1f}% ({max_ats[0]})")

        # Show most recent 20 weeks
        print("\n  Recent Dark Pool Activity (last 20 weeks):")
        print(f"  {'Week Ending':<14} {'Total Vol':>12} {'ATS Vol':>12} {'OTC Vol':>12} {'ATS%':>7}")
        for date, dp in dp_sorted[-20:]:
            print(f"  {date:<14} {dp['total_weekly_volume']:>12,} {dp['ats_volume']:>12,} {dp['otc_volume']:>12,} {dp['ats_pct'] * 100:>6.1f}%")

    # === OWNERSHIP ANALYSIS ===
    print("\n" + "─" * 60)
    print("  INSTITUTIONAL OWNERSHIP (13F)")
    print("─" * 60)

    own_sorted = sorted(own_by_quarter.items(), key=lambda x: x[0])
    if own_sorted:
        print(f"  Quarters:          {len(own_sorted)}")
        print(f"  Date Range:        {own_sorted[0][0]} → {own_sorted[-1][0]}")

        print(f"\n  {'Quarter':<14} {'Inst. Shares':>14} {'# Inst':>8} {'Top10 Conc':>12}")
        for date, own in own_sorted:
            print(f"  {date:<14} {own['total_institutional_shares']:>14,} {own['num_institutions']:>8} {own['top_10_concentration'] * 100:>11.1f}%")

        # Show holdings detail for latest quarter
        latest_own = own_sorted[-1]
        print(f"\n  Latest Quarter ({latest_own[0]}) Top Holdings:")
        holdings = sorted(latest_own[1]["holdings"], key=lambda h: h["shares"], reverse=True)
        for h in holdings[:15]:
            val = h["value_usd"] * 1000 if h["value_usd"] else 0
            print(f"    {h['manager_name']:<40} {h['shares']:>12,} shares  ${val:>14,}")

    # === BORROW RATE ESTIMATES ===
    print("\n" + "─" * 60)
    print("  ESTIMATED BORROW RATES")
    print("─" * 60)

    borrow_rows = [r for r in aligned if r.get("borrow_rate_est", 0) > 0]
    if borrow_rows:
        current_br = borrow_rows[-1]
        max_br = max(borrow_rows, key=lambda x: x["borrow_rate_est"])
        print(f"  Current Est Rate:  {current_br['borrow_rate_est']:.2f}%")
        print(f"  Peak Est Rate:     {max_br['borrow_rate_est']:.2f}% ({max_br['date']})")
        print(f"  Current Util:      {current_br['borrow_utilization']:.2f}%")


def print_key_events_analysis(aligned):
    """Analyze correlations around key GME events."""
    print("\n" + "=" * 80)
    print("  KEY EVENT ANALYSIS")
    print("=" * 80)

    events = [
        ("2021-01-22", "2021-02-05", "The Sneeze / Short Squeeze"),
        ("2021-02-24", "2021-03-12", "Feb 2021 Recovery Rally"),
        ("2021-05-25", "2021-06-15", "June 2021 Run-up"),
        ("2021-08-20", "2021-09-10", "Aug/Sep 2021 Cycle"),
        ("2021-11-01", "2021-11-30", "Nov 2021 Run-up"),
        ("2022-03-14", "2022-04-05", "Mar 2022 Rally (RC Buys)"),
        ("2022-05-16", "2022-06-10", "May 2022 Rally"),
        ("2022-07-22", "2022-08-22", "Post-Split Period"),
        ("2024-05-10", "2024-06-20", "DFV Return / Roaring Kitty"),
        ("2024-12-01", "2025-01-31", "Recent Activity"),
    ]

    for start, end, label in events:
        period = [r for r in aligned if start <= r["date"] <= end]
        if not period:
            continue

        open_price = period[0]["open"]
        close_price = period[-1]["close"]
        high_price = max(r["high"] for r in period)
        low_price = min(r["low"] for r in period)
        total_vol = sum(r["volume"] for r in period)
        avg_vol = total_vol / len(period)

        # FTD metrics
        total_ftd = sum(r["ftd_quantity"] for r in period)
        max_ftd = max(period, key=lambda x: x["ftd_quantity"])
        avg_ftd = total_ftd / len(period)

        # SI at start and end
        si_start = period[0]["short_interest"]
        si_end = period[-1]["short_interest"]

        # Dark pool at start and end
        dp_start = period[0]["dp_ats_pct"]
        dp_end = period[-1]["dp_ats_pct"]

        pct_change = (close_price - open_price) / open_price * 100 if open_price > 0 else 0

        print(f"\n  📌 {label} ({start} → {end})")
        print(f"     Price: ${open_price:.2f} → ${close_price:.2f} (high ${high_price:.2f}) [{pct_change:+.1f}%]")
        print(f"     Volume: {total_vol:,.0f} total, {avg_vol:,.0f}/day avg")
        print(f"     FTDs: {total_ftd:,.0f} total, {avg_ftd:,.0f}/day avg, peak {max_ftd['ftd_quantity']:,.0f} ({max_ftd['date']})")
        print(f"     Short Interest: {si_start:,} → {si_end:,}")
        print(f"     Dark Pool ATS%: {dp_start:.1f}% → {dp_end:.1f}%")
        print(f"     Borrow Rate Est: {period[0]['borrow_rate_est']:.1f}% → {period[-1]['borrow_rate_est']:.1f}%")


def export_csv(aligned, filepath):
    """Export aligned data to CSV."""
    if not aligned:
        return
    fieldnames = list(aligned[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aligned)
    print(f"\n✅ CSV exported: {filepath} ({len(aligned)} rows)")


def export_tradingview_json(aligned, si_by_date, dp_by_week, own_by_quarter, filepath):
    """
    Export JSON optimized for TradingView Pine Script indicator.
    Only includes dates where alt data actually changes (sparse format).
    """
    tv_data = {
        "meta": {
            "symbol": "GME",
            "generated": datetime.now().isoformat(),
            "date_range": f"{aligned[0]['date']} to {aligned[-1]['date']}",
            "total_trading_days": len(aligned),
        },
        "ftd_daily": {},
        "short_interest": {},
        "dark_pool_weekly": {},
        "ownership_quarterly": {},
        "borrow_rates": {},
        "key_events": {},
    }

    # FTD — only dates with FTDs > 0
    for row in aligned:
        if row["ftd_quantity"] > 0:
            tv_data["ftd_daily"][row["date"]] = {
                "qty": row["ftd_quantity"],
                "val": row["ftd_value"],
                "avg20": row["ftd_20d_avg"],
                "sum5": row["ftd_5d_sum"],
                "pct_vol": row["ftd_pct_of_volume"],
            }

    # Short interest (original report dates only)
    for date, si in si_by_date.items():
        tv_data["short_interest"][date] = si

    # Dark pool (original report dates only)
    for date, dp in dp_by_week.items():
        tv_data["dark_pool_weekly"][date] = dp

    # Ownership (original report dates only)
    for date, own in own_by_quarter.items():
        tv_data["ownership_quarterly"][date] = {
            "total_shares": own["total_institutional_shares"],
            "num_institutions": own["num_institutions"],
            "top10_concentration": own["top_10_concentration"],
        }

    # Key events
    tv_data["key_events"] = {
        "2021-01-28": "Sneeze / Buy Button Removed",
        "2021-06-09": "Shareholder Meeting / ATM Offering",
        "2022-03-22": "RC Increases Stake",
        "2022-07-22": "4:1 Stock Split",
        "2024-05-13": "DFV Returns (Roaring Kitty)",
        "2024-06-07": "DFV Reveals 9M Share Position",
        "2024-06-14": "GME ATM Offering (75M shares)",
    }

    with open(filepath, "w") as f:
        json.dump(tv_data, f, indent=2, default=str)
    print(f"✅ TradingView JSON exported: {filepath}")


def print_interpretation(aligned, si_by_date, dp_by_week, ftd_by_date):
    """Print analytical interpretation of the data."""
    print("\n" + "=" * 80)
    print("  DATA INTERPRETATION & ANALYSIS")
    print("=" * 80)

    print("""
  1. FTD PATTERN ANALYSIS
  ─────────────────────────────────────────────────────

  Failure-to-Deliver data reveals the structural mechanics behind GME's price
  movements. FTDs represent shares that were sold but never delivered — a
  hallmark of naked short selling or operational failures in the clearing system.
""")

    # Find FTD spikes that preceded price moves
    ftd_spikes = []
    for i in range(20, len(aligned)):
        row = aligned[i]
        if row["ftd_quantity"] > 0 and row["ftd_20d_avg"] > 0:
            spike_ratio = row["ftd_quantity"] / row["ftd_20d_avg"]
            if spike_ratio > 3.0 and row["ftd_quantity"] > 100000:
                # Check price move in next 5-10 days
                future_prices = [aligned[j]["close"] for j in range(i + 1, min(i + 11, len(aligned)))]
                if future_prices:
                    max_future = max(future_prices)
                    min_future = min(future_prices)
                    pct_up = (max_future - row["close"]) / row["close"] * 100
                    pct_down = (min_future - row["close"]) / row["close"] * 100
                    ftd_spikes.append({
                        "date": row["date"],
                        "ftd": row["ftd_quantity"],
                        "avg": row["ftd_20d_avg"],
                        "spike_ratio": spike_ratio,
                        "close": row["close"],
                        "max_up_10d": pct_up,
                        "max_down_10d": pct_down,
                    })

    if ftd_spikes:
        up_after = sum(1 for s in ftd_spikes if s["max_up_10d"] > 5)
        down_after = sum(1 for s in ftd_spikes if s["max_down_10d"] < -5)
        total_spikes = len(ftd_spikes)

        print(f"  Found {total_spikes} FTD spikes (>3x 20-day avg, >100K shares)")
        print(f"  - {up_after} ({up_after/total_spikes*100:.0f}%) followed by >5% price increase within 10 days")
        print(f"  - {down_after} ({down_after/total_spikes*100:.0f}%) followed by >5% price decrease within 10 days")

        print(f"\n  Notable FTD Spike → Price Move Correlations:")
        notable = sorted(ftd_spikes, key=lambda x: x["spike_ratio"], reverse=True)[:15]
        print(f"  {'Date':<14} {'FTDs':>10} {'Spike':>6} {'Close':>8} {'10d Max Up':>10} {'10d Max Dn':>10}")
        for s in notable:
            print(f"  {s['date']:<14} {s['ftd']:>10,} {s['spike_ratio']:>5.1f}x ${s['close']:>7.2f} {s['max_up_10d']:>+9.1f}% {s['max_down_10d']:>+9.1f}%")

    print("""
  2. SHORT INTEREST vs PRICE ACTION
  ─────────────────────────────────────────────────────

  Short interest data shows the aggregate bearish positioning against GME.
  Days-to-cover (DTC) is particularly telling — it measures how many days
  of average volume it would take to close all short positions.
""")

    si_sorted = sorted(si_by_date.items(), key=lambda x: x[0])
    if si_sorted:
        # Find SI changes correlated with price moves
        for i in range(1, len(si_sorted)):
            prev_date, prev_si = si_sorted[i - 1]
            curr_date, curr_si = si_sorted[i]
            si_change = curr_si["short_interest"] - prev_si["short_interest"]
            si_change_pct = si_change / prev_si["short_interest"] * 100 if prev_si["short_interest"] > 0 else 0

            # Find price change over same period
            prices_in_range = [r for r in aligned if prev_date <= r["date"] <= curr_date]
            if len(prices_in_range) >= 2:
                price_change = (prices_in_range[-1]["close"] - prices_in_range[0]["close"]) / prices_in_range[0]["close"] * 100

        # Summary stats
        dtc_values = [si[1]["days_to_cover"] for si in si_sorted]
        avg_dtc = sum(dtc_values) / len(dtc_values)
        max_dtc = max(dtc_values)
        print(f"  Average Days-to-Cover: {avg_dtc:.2f}")
        print(f"  Maximum Days-to-Cover: {max_dtc:.2f}")
        print(f"  A DTC > 5 is considered elevated; GME has historically shown spikes")
        print(f"  coinciding with or preceding significant price volatility.")

    print("""
  3. DARK POOL ACTIVITY INTERPRETATION
  ─────────────────────────────────────────────────────

  Dark pool (ATS) trading represents orders executed off-exchange, often by
  institutional market makers. A high ATS percentage can indicate:
  - Institutional accumulation (buying off-exchange to avoid price impact)
  - Market maker internalization of retail orders
  - Reduced price discovery on lit exchanges
""")

    dp_sorted = sorted(dp_by_week.items(), key=lambda x: x[0])
    if dp_sorted:
        ats_pcts = [d[1]["ats_pct"] * 100 for d in dp_sorted]
        avg_ats = sum(ats_pcts) / len(ats_pcts)
        max_ats = max(ats_pcts)
        min_ats = min(ats_pcts)
        print(f"  Average ATS %: {avg_ats:.1f}%")
        print(f"  Range: {min_ats:.1f}% - {max_ats:.1f}%")
        print(f"  Typical large-cap ATS%: 15-25%. GME's levels suggest")
        print(f"  significant off-exchange routing of order flow.")

    print("""
  4. FTD SETTLEMENT CYCLES (T+35 / T+2+35)
  ─────────────────────────────────────────────────────

  RegSHO requires FTDs to be closed out within specific timeframes:
  - T+2: Standard settlement
  - T+35: Maximum allowed for market makers (Reg SHO Rule 204)
  - C+35: Calendar days from FTD date (actual enforcement)

  Large FTD spikes often correspond to forced buying 35 calendar days later,
  creating predictable price pressure cycles.
""")

    # Calculate T+35 from top FTD days and check if price increased
    ftd_sorted = sorted(
        [(r["date"], r["ftd_quantity"]) for r in aligned if r["ftd_quantity"] > 200000],
        key=lambda x: x[1],
        reverse=True,
    )[:30]

    date_to_idx = {r["date"]: i for i, r in enumerate(aligned)}

    print(f"  T+35 Analysis for Top FTD Days (>200K shares):")
    print(f"  {'FTD Date':<14} {'FTDs':>10} {'T+35 Date':<14} {'T+35 Close':>10} {'T+35 Chg':>10}")

    t35_hits = 0
    t35_total = 0
    for ftd_date, ftd_qty in ftd_sorted:
        try:
            t35_date = (datetime.strptime(ftd_date, "%Y-%m-%d") + timedelta(days=35)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        # Find closest trading day to T+35
        ftd_idx = date_to_idx.get(ftd_date)
        if ftd_idx is None:
            continue

        ftd_close = aligned[ftd_idx]["close"]

        # Find T+35 date (nearest trading day)
        t35_idx = None
        for offset in range(0, 5):
            check = (datetime.strptime(ftd_date, "%Y-%m-%d") + timedelta(days=35 + offset)).strftime("%Y-%m-%d")
            if check in date_to_idx:
                t35_idx = date_to_idx[check]
                t35_date = check
                break

        if t35_idx and t35_idx < len(aligned):
            t35_close = aligned[t35_idx]["close"]
            t35_change = (t35_close - ftd_close) / ftd_close * 100
            t35_total += 1
            if t35_change > 0:
                t35_hits += 1
            print(f"  {ftd_date:<14} {ftd_qty:>10,} {t35_date:<14} ${t35_close:>8.2f} {t35_change:>+9.1f}%")

    if t35_total > 0:
        print(f"\n  T+35 Win Rate: {t35_hits}/{t35_total} ({t35_hits/t35_total*100:.0f}%) of large FTD days")
        print(f"  showed positive price change at T+35")

    print("""
  5. OVERALL THESIS
  ─────────────────────────────────────────────────────

  The data paints a picture of a stock with unusual market microstructure:

  • Persistent FTDs indicate ongoing settlement failures that exceed normal
    levels for a stock of GME's market cap and liquidity.

  • Short interest remains elevated relative to the available float,
    creating vulnerability to forced buying when FTDs must be settled.

  • Dark pool routing absorbs significant order flow off-exchange,
    potentially suppressing price discovery on lit markets.

  • The T+35 settlement cycle creates periodic forced-buying pressure,
    which has historically coincided with price spikes.

  • Institutional ownership snapshots show the evolving composition of
    large holders, with concentration levels indicating either conviction
    or passive index inclusion effects.
""")


def main():
    print("Loading GME data from all sources...")
    print()

    # 1. Price data
    print("📈 Loading price data...")
    prices = load_price_data()
    print(f"  Daily bars loaded: {len(prices)} ({prices[0]['date']} → {prices[-1]['date']})")

    # 2. FTD data (from cached SEC zips)
    print("📋 Loading FTD data...")
    ftd_by_date = load_ftd_data()

    # 3. Short Interest (from FINRA cache)
    print("📊 Loading short interest...")
    si_by_date = load_short_interest()

    # 4. Dark Pool (from FINRA cache)
    print("🌑 Loading dark pool data...")
    dp_by_week = load_dark_pool()

    # 5. Ownership (from SEC cache)
    print("🏛️  Loading institutional ownership...")
    own_by_quarter = load_ownership()

    # 6. Estimate borrow rates
    print("💰 Computing borrow rate estimates...")
    borrow_rates = estimate_borrow_rates(si_by_date)

    # 7. Align everything
    print("\n⚡ Aligning all data on daily timeline...")
    aligned = align_all_data(prices, ftd_by_date, si_by_date, dp_by_week, own_by_quarter, borrow_rates)
    print(f"  Aligned dataset: {len(aligned)} rows")

    # 8. Compute derived metrics
    print("📐 Computing derived metrics (rolling averages, volatility, etc.)...")
    aligned = compute_derived_metrics(aligned)

    # 9. Print reports
    print_summary_report(aligned, ftd_by_date, si_by_date, dp_by_week, own_by_quarter)
    print_key_events_analysis(aligned)
    print_interpretation(aligned, si_by_date, dp_by_week, ftd_by_date)

    # 10. Export
    csv_path = DATA_DIR / "gme_holistic_aligned.csv"
    json_path = DATA_DIR / "gme_report_data.json"
    export_csv(aligned, csv_path)
    export_tradingview_json(aligned, si_by_date, dp_by_week, own_by_quarter, json_path)


if __name__ == "__main__":
    main()
