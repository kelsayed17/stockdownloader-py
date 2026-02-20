#!/usr/bin/env python3
"""Comprehensive GME data deep-dive analysis."""

import csv
import json
from datetime import datetime, timedelta
from collections import defaultdict
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'gme_holistic_aligned.csv')
JSON_PATH = os.path.join(DATA_DIR, 'gme_report_data.json')
OWNERSHIP_PATH = os.path.join(DATA_DIR, 'cache', 'ownership', 'GME_13f.json')

# Read CSV
rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Index by date for quick lookup
date_idx = {}
for i, r in enumerate(rows):
    date_idx[r['date']] = i

print(f'Total rows: {len(rows)}')
print(f'Date range: {rows[0]["date"]} to {rows[-1]["date"]}')

# ============================================
# A) ALL DAYS WITH >10% PRICE MOVE
# ============================================
print('\n' + '='*100)
print('SECTION A: ALL SINGLE-DAY MOVES > 10%')
print('='*100)

big_moves = []
for i, r in enumerate(rows):
    ret = float(r['daily_return_pct'])
    if abs(ret) > 10:
        big_moves.append((i, r))

print(f'Total days with >10% move: {len(big_moves)}\n')

for idx, r in big_moves:
    date = r['date']
    ret = float(r['daily_return_pct'])
    close = float(r['close'])
    vol = int(float(r['volume']))
    ftd = int(float(r['ftd_quantity']))
    si = int(float(r['short_interest']))
    dp_pct = float(r['dp_ats_pct'])
    inst = int(float(r['institutional_shares']))
    rvol = float(r['relative_volume'])
    v20d = float(r['volatility_20d'])

    context_ftds = []
    context_prices = []
    for offset in range(-3, 4):
        ci = idx + offset
        if 0 <= ci < len(rows):
            cr = rows[ci]
            context_ftds.append((cr['date'], int(float(cr['ftd_quantity']))))
            context_prices.append((cr['date'], float(cr['daily_return_pct']), float(cr['close'])))

    ftd_sum_7d = sum(cf[1] for cf in context_ftds)

    print(f'DATE: {date} | Return: {ret:+.2f}% | Close: ${close:.2f} | Vol: {vol:,} | RVOL: {rvol:.2f}x')
    print(f'  FTD: {ftd:,} | SI: {si:,} | DP ATS%: {dp_pct:.1f}% | Inst Shares: {inst:,} | Vol20d: {v20d:.2f}%')
    print(f'  Surrounding prices: ', end='')
    for cd, cr_ret, cr_close in context_prices:
        marker = ' <<<' if cd == date else ''
        print(f'{cd}({cr_ret:+.1f}%,${cr_close:.2f}){marker}  ', end='')
    print()
    print(f'  Surrounding FTDs (7d sum={ftd_sum_7d:,}): ', end='')
    for cd, cf in context_ftds:
        print(f'{cd}:{cf:,}  ', end='')
    print('\n')


# ============================================
# B) T+35 FTD SETTLEMENT PATTERN
# ============================================
print('\n' + '='*100)
print('SECTION B: T+35 FTD SETTLEMENT PATTERN')
print('='*100)

# Find all FTD spikes >1M shares
ftd_spikes = []
for i, r in enumerate(rows):
    ftd = int(float(r['ftd_quantity']))
    if ftd > 1000000:
        ftd_spikes.append((i, r, ftd))

print(f'Total FTD days >1M shares: {len(ftd_spikes)}\n')

hits = 0
misses = 0
results = []

for idx, r, ftd_qty in ftd_spikes:
    date = r['date']
    ftd_date = datetime.strptime(date, '%Y-%m-%d')
    t35_date = ftd_date + timedelta(days=35)

    # Find price at T+35 +/- 3 days
    best_match = None
    best_dist = 99
    for offset_days in range(-3, 4):
        check_date = t35_date + timedelta(days=offset_days)
        check_str = check_date.strftime('%Y-%m-%d')
        if check_str in date_idx:
            ci = date_idx[check_str]
            if best_match is None or abs(offset_days) < best_dist:
                best_match = (ci, rows[ci], offset_days)
                best_dist = abs(offset_days)

    if best_match:
        settle_idx, settle_row, settle_offset = best_match
        settle_date = settle_row['date']
        settle_close = float(settle_row['close'])
        ftd_close = float(r['close'])

        # Check price change in window around T+35 (+/-3 days)
        max_price = 0
        min_price = 999999
        for offset_days in range(-3, 4):
            check_date = t35_date + timedelta(days=offset_days)
            check_str = check_date.strftime('%Y-%m-%d')
            if check_str in date_idx:
                ci = date_idx[check_str]
                p = float(rows[ci]['close'])
                max_price = max(max_price, p)
                min_price = min(min_price, p)

        price_change_pct = ((settle_close - ftd_close) / ftd_close) * 100
        max_gain_pct = ((max_price - ftd_close) / ftd_close) * 100

        # "Hit" = price went up >5% at any point in the T+35 window
        spike = max_gain_pct > 5
        if spike:
            hits += 1
        else:
            misses += 1

        results.append({
            'ftd_date': date,
            'ftd_qty': ftd_qty,
            'ftd_close': ftd_close,
            'settle_date': settle_date,
            'settle_close': settle_close,
            'price_change': price_change_pct,
            'max_gain': max_gain_pct,
            'hit': spike
        })

        hit_marker = 'HIT' if spike else 'MISS'
        print(f'FTD: {date} ({ftd_qty:,} shares, close=${ftd_close:.2f}) -> T+35: {settle_date} (close=${settle_close:.2f}) | Change: {price_change_pct:+.1f}% | Max gain: {max_gain_pct:+.1f}% | {hit_marker}')
    else:
        print(f'FTD: {date} ({ftd_qty:,} shares) -> T+35: No data available')

total = hits + misses
print(f'\nT+35 SUMMARY:')
print(f'  Total FTD spikes >1M analyzed: {total}')
print(f'  Hits (>5% gain in T+35 window): {hits} ({hits/total*100:.1f}%)')
print(f'  Misses: {misses} ({misses/total*100:.1f}%)')

# Also check >10% threshold
hits_10 = sum(1 for r in results if r['max_gain'] > 10)
print(f'  Hits (>10% gain in T+35 window): {hits_10} ({hits_10/total*100:.1f}%)')


# ============================================
# C) DARK POOL % SUDDEN INCREASES >5pp
# ============================================
print('\n' + '='*100)
print('SECTION C: DARK POOL ATS% SUDDEN INCREASES >5 PERCENTAGE POINTS')
print('='*100)

# Track changes in dp_ats_pct - need to find transitions where it jumps
prev_dp = 0.0
dp_changes = []
for i, r in enumerate(rows):
    dp = float(r['dp_ats_pct'])
    if dp > 0 and prev_dp > 0:
        change = dp - prev_dp
        if abs(change) > 5:
            dp_changes.append((i, r, prev_dp, dp, change))
    prev_dp = dp

# DP data is weekly, so look at week-over-week changes
# Actually, the data is daily but DP values are carried forward from weekly reports
# Let's look at distinct DP value transitions
distinct_dp = []
prev_dp_val = -1
for i, r in enumerate(rows):
    dp = float(r['dp_ats_pct'])
    if dp != prev_dp_val and dp > 0:
        distinct_dp.append((i, r, dp, prev_dp_val if prev_dp_val > 0 else None))
        prev_dp_val = dp

print(f'Total distinct DP ATS% values: {len(distinct_dp)}')
print(f'\nDP transitions where ATS% changed by >5 percentage points:\n')

dp_jumps = []
for j in range(1, len(distinct_dp)):
    curr_i, curr_r, curr_dp, _ = distinct_dp[j]
    prev_i, prev_r, prev_dp, _ = distinct_dp[j-1]
    change = curr_dp - prev_dp
    if abs(change) > 5:
        dp_jumps.append((curr_i, curr_r, prev_dp, curr_dp, change, prev_r['date']))

print(f'Total ATS% jumps >5pp: {len(dp_jumps)}\n')

for idx, r, old_dp, new_dp, change, prev_date in dp_jumps:
    date = r['date']
    close = float(r['close'])

    # Look at price performance 5 days before and 10 days after
    pre_prices = []
    post_prices = []
    for offset in range(-5, 0):
        ci = idx + offset
        if 0 <= ci < len(rows):
            pre_prices.append(float(rows[ci]['close']))
    for offset in range(0, 11):
        ci = idx + offset
        if 0 <= ci < len(rows):
            post_prices.append(float(rows[ci]['close']))

    pre_avg = sum(pre_prices)/len(pre_prices) if pre_prices else close
    post_max = max(post_prices) if post_prices else close
    post_min = min(post_prices) if post_prices else close
    post_end = post_prices[-1] if post_prices else close

    price_change_10d = ((post_end - close) / close) * 100 if close > 0 else 0

    direction = 'UP' if change > 0 else 'DOWN'

    print(f'{date}: ATS% {old_dp:.1f}% -> {new_dp:.1f}% ({direction} {abs(change):.1f}pp) | Price: ${close:.2f} | 10d price change: {price_change_10d:+.1f}%')
    print(f'  Previous change date: {prev_date}')
    print(f'  Next 10d price range: ${post_min:.2f} - ${post_max:.2f}')
    print()


# ============================================
# D) SHORT INTEREST CHANGES >10%
# ============================================
print('\n' + '='*100)
print('SECTION D: SHORT INTEREST CHANGES >10% BETWEEN REPORTING PERIODS')
print('='*100)

# Find distinct SI values (SI is reported bi-monthly)
distinct_si = []
prev_si_val = -1
for i, r in enumerate(rows):
    si = int(float(r['short_interest']))
    if si != prev_si_val and si > 0:
        distinct_si.append((i, r, si))
        prev_si_val = si

print(f'Total distinct SI reports: {len(distinct_si)}\n')

si_jumps = []
for j in range(1, len(distinct_si)):
    curr_i, curr_r, curr_si = distinct_si[j]
    prev_i, prev_r, prev_si = distinct_si[j-1]
    if prev_si > 0:
        pct_change = ((curr_si - prev_si) / prev_si) * 100
        if abs(pct_change) > 10:
            si_jumps.append((curr_i, curr_r, prev_si, curr_si, pct_change, prev_r['date']))

print(f'SI changes >10%: {len(si_jumps)}\n')

for idx, r, old_si, new_si, pct_change, prev_date in si_jumps:
    date = r['date']
    close = float(r['close'])

    # Look at price 30 days after
    price_30d = None
    max_price_30d = close
    min_price_30d = close
    for offset in range(1, 31):
        ci = idx + offset
        if 0 <= ci < len(rows):
            p = float(rows[ci]['close'])
            max_price_30d = max(max_price_30d, p)
            min_price_30d = min(min_price_30d, p)
            if offset >= 28:
                price_30d = p
                break

    if price_30d is None:
        for offset in range(30, 0, -1):
            ci = idx + offset
            if 0 <= ci < len(rows):
                price_30d = float(rows[ci]['close'])
                break

    change_30d = ((price_30d - close) / close) * 100 if price_30d and close > 0 else 0
    max_30d = ((max_price_30d - close) / close) * 100
    min_30d = ((min_price_30d - close) / close) * 100

    direction = 'UP' if pct_change > 0 else 'DOWN'

    print(f'{date}: SI {old_si:,} -> {new_si:,} ({direction} {abs(pct_change):.1f}%) from {prev_date}')
    print(f'  Price at SI report: ${close:.2f}')
    print(f'  30-day price change: {change_30d:+.1f}% (to ${price_30d:.2f})')
    print(f'  30-day range: {min_30d:+.1f}% to {max_30d:+.1f}% (${min_price_30d:.2f} - ${max_price_30d:.2f})')
    print()

