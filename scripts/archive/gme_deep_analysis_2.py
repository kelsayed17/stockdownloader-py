#!/usr/bin/env python3
"""GME deep-dive analysis - Parts E, F, G."""

import csv
import json
from datetime import datetime, timedelta
from collections import defaultdict
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'gme_holistic_aligned.csv')
OWNERSHIP_PATH = os.path.join(DATA_DIR, 'cache', 'ownership', 'GME_13f.json')

# Read CSV
rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

date_idx = {}
for i, r in enumerate(rows):
    date_idx[r['date']] = i

# Read institutional ownership JSON
with open(OWNERSHIP_PATH, 'r') as f:
    ownership_data = json.load(f)

# ============================================
# E) INSTITUTIONAL OWNERSHIP TRANSITIONS
# ============================================
print('='*100)
print('SECTION E: INSTITUTIONAL OWNERSHIP TRANSITIONS')
print('='*100)

# Get distinct quarterly ownership values from CSV
inst_quarters = []
prev_inst = -1
for i, r in enumerate(rows):
    inst = int(float(r['institutional_shares']))
    num_inst = int(float(r['num_institutions']))
    if inst != prev_inst and inst > 0:
        inst_quarters.append((i, r, inst, num_inst))
        prev_inst = inst

print(f'Total distinct institutional ownership levels: {len(inst_quarters)}\n')

# Calculate quarter-over-quarter changes
for j in range(1, len(inst_quarters)):
    curr_i, curr_r, curr_inst, curr_num = inst_quarters[j]
    prev_i, prev_r, prev_inst, prev_num = inst_quarters[j-1]

    change = curr_inst - prev_inst
    pct_change = (change / prev_inst) * 100 if prev_inst > 0 else 0

    # Get price range during transition
    prices = []
    for idx in range(prev_i, min(curr_i + 1, len(rows))):
        prices.append(float(rows[idx]['close']))

    start_price = float(prev_r['close'])
    end_price = float(curr_r['close'])
    price_change = ((end_price - start_price) / start_price) * 100

    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0

    print(f'{prev_r["date"]} -> {curr_r["date"]}: Inst. shares {prev_inst:,} -> {curr_inst:,} ({pct_change:+.1f}%) | # Institutions: {prev_num} -> {curr_num}')
    print(f'  Price: ${start_price:.2f} -> ${end_price:.2f} ({price_change:+.1f}%) | Period range: ${min_price:.2f} - ${max_price:.2f}')
    print()

# Also analyze the 13F JSON for top holders
print('\n--- TOP HOLDER ANALYSIS FROM 13F DATA ---\n')
if isinstance(ownership_data, list):
    quarterly_data = ownership_data
elif isinstance(ownership_data, dict):
    quarterly_data = ownership_data.get('data', ownership_data.get('quarters', []))

# Process each quarter from the JSON
if quarterly_data:
    for q in quarterly_data:
        quarter = q.get('quarter', q.get('period', 'unknown'))
        total_shares = q.get('total_shares', 0)
        num_holders = q.get('num_holders', q.get('count', 0))
        top_holders = q.get('top_holders', q.get('holders', []))

        print(f'Quarter: {quarter} | Total shares: {total_shares:,} | Holders: {num_holders}')
        if top_holders and len(top_holders) > 0:
            for i, h in enumerate(top_holders[:5]):
                name = h.get('name', h.get('holder', 'Unknown'))
                shares = h.get('shares', h.get('value', 0))
                print(f'  {i+1}. {name}: {shares:,} shares')
        print()


# ============================================
# F) "IMPOSSIBLE" DATA - SI + IO + INSIDERS > OUTSTANDING
# ============================================
print('\n' + '='*100)
print('SECTION F: "IMPOSSIBLE" DATA - SI + IO + INSIDERS > SHARES OUTSTANDING')
print('='*100)

# GME shares outstanding history (split-adjusted):
# Pre-split (before July 22, 2022): ~76.35M shares -> post-split equivalent ~305.4M
# Post-split: 305.4M -> then ATM offerings increased it
# As of 2024: ~446.5M shares outstanding
# RC Ventures (Ryan Cohen insider): ~75.2M shares (16.78%)

# Approximate shares outstanding timeline (split-adjusted)
shares_outstanding = {
    '2020': 280_000_000,  # ~70M pre-split * 4
    '2021': 305_000_000,  # increased slightly with ATM offerings
    '2022_pre_split': 305_000_000,
    '2022_post_split': 305_000_000,
    '2023': 305_000_000,
    '2024': 446_500_000,  # After ATM offerings
    '2025': 447_000_000,
    '2026': 447_000_000,
}

RC_SHARES = 75_200_000  # Ryan Cohen's insider holding

print(f'\nRC Ventures (insider) holds ~{RC_SHARES:,} shares\n')

# For each date where we have both SI and IO data
prev_si = 0
prev_inst = 0
impossible_dates = []

# Get distinct quarterly periods
for j in range(len(inst_quarters)):
    idx, r, inst_shares, num_inst = inst_quarters[j]
    date = r['date']
    year = date[:4]
    si = int(float(r['short_interest']))

    # Determine shares outstanding
    if year in ('2020', '2021'):
        outstanding = shares_outstanding[year]
    elif year == '2022':
        outstanding = shares_outstanding['2022_post_split']
    elif year == '2023':
        outstanding = shares_outstanding['2023']
    elif year == '2024':
        outstanding = shares_outstanding['2024']
    elif year == '2025':
        outstanding = shares_outstanding['2025']
    else:
        outstanding = shares_outstanding['2026']

    # Need SI data too - find nearest SI
    if si == 0:
        # Look forward for nearest SI
        for offset in range(0, 30):
            ci = idx + offset
            if ci < len(rows):
                test_si = int(float(rows[ci]['short_interest']))
                if test_si > 0:
                    si = test_si
                    break

    if si > 0 and inst_shares > 0:
        total_claimed = si + inst_shares + RC_SHARES
        ratio = total_claimed / outstanding
        excess = total_claimed - outstanding
        excess_pct = (excess / outstanding) * 100

        is_impossible = total_claimed > outstanding

        if is_impossible:
            impossible_dates.append(date)

        marker = '>>> IMPOSSIBLE <<<' if is_impossible else ''
        print(f'{date}: SI={si:,} + IO={inst_shares:,} + RC={RC_SHARES:,} = {total_claimed:,} vs Outstanding={outstanding:,} | Ratio: {ratio:.2%} | Excess: {excess:,} ({excess_pct:+.1f}%) {marker}')

print(f'\nTotal quarters with "impossible" data: {len(impossible_dates)}')
if impossible_dates:
    print(f'Dates: {", ".join(impossible_dates)}')


# ============================================
# G) FTD CLUSTERING PATTERNS
# ============================================
print('\n' + '='*100)
print('SECTION G: FTD CLUSTERING PATTERNS')
print('='*100)

# Analyze day-of-week and day-of-month patterns
dow_ftds = defaultdict(list)  # day of week -> list of FTD quantities
dom_ftds = defaultdict(list)  # day of month -> list of FTD quantities
month_ftds = defaultdict(list)  # month -> list of FTD quantities

# Only use rows where FTD data exists
for r in rows:
    ftd = int(float(r['ftd_quantity']))
    if ftd > 0:
        date = datetime.strptime(r['date'], '%Y-%m-%d')
        dow_ftds[date.strftime('%A')].append(ftd)
        dom_ftds[date.day].append(ftd)
        month_ftds[date.month].append(ftd)

print('\n--- DAY OF WEEK ANALYSIS ---\n')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
for day in day_order:
    ftds = dow_ftds.get(day, [])
    if ftds:
        avg = sum(ftds) / len(ftds)
        total = sum(ftds)
        count = len(ftds)
        big_days = sum(1 for f in ftds if f > 1_000_000)
        print(f'{day:10s}: Count={count:4d} | Avg FTD={avg:>12,.0f} | Total={total:>15,} | Days>1M: {big_days}')

print('\n--- DAY OF MONTH ANALYSIS ---\n')
print('Day | Count | Avg FTD      | Total FTDs       | Days>1M')
print('-' * 75)
for day in range(1, 32):
    ftds = dom_ftds.get(day, [])
    if ftds:
        avg = sum(ftds) / len(ftds)
        total = sum(ftds)
        count = len(ftds)
        big_days = sum(1 for f in ftds if f > 1_000_000)
        marker = ' ***' if avg > 1_000_000 else ''
        print(f' {day:2d}  | {count:4d}  | {avg:>12,.0f} | {total:>15,} | {big_days:4d}{marker}')

print('\n--- MONTHLY ANALYSIS ---\n')
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month in range(1, 13):
    ftds = month_ftds.get(month, [])
    if ftds:
        avg = sum(ftds) / len(ftds)
        total = sum(ftds)
        count = len(ftds)
        big_days = sum(1 for f in ftds if f > 1_000_000)
        print(f'{month_names[month-1]:3s}: Count={count:4d} | Avg FTD={avg:>12,.0f} | Total={total:>15,} | Days>1M: {big_days}')

# Look for cyclical patterns - FTD spikes every N days
print('\n--- CYCLICAL PATTERN ANALYSIS ---\n')
print('Looking for periodic FTD spikes (>500K shares)...\n')

spike_dates = []
for r in rows:
    ftd = int(float(r['ftd_quantity']))
    if ftd > 500_000:
        spike_dates.append(datetime.strptime(r['date'], '%Y-%m-%d'))

# Calculate gaps between spikes
if len(spike_dates) > 1:
    gaps = []
    for i in range(1, len(spike_dates)):
        gap = (spike_dates[i] - spike_dates[i-1]).days
        if gap > 0:
            gaps.append(gap)

    # Frequency distribution of gaps
    gap_buckets = defaultdict(int)
    for g in gaps:
        bucket = (g // 7) * 7  # weekly buckets
        gap_buckets[bucket] += 1

    print(f'Total FTD spikes >500K: {len(spike_dates)}')
    print(f'Average gap between spikes: {sum(gaps)/len(gaps):.1f} days')
    print(f'Median gap: {sorted(gaps)[len(gaps)//2]} days')
    print(f'\nGap distribution (calendar day buckets):')
    for bucket in sorted(gap_buckets.keys()):
        count = gap_buckets[bucket]
        bar = '#' * count
        print(f'  {bucket:3d}-{bucket+6:3d} days: {count:4d} {bar}')

    # Check specific cycle lengths
    print('\n--- TESTING SPECIFIC CYCLE LENGTHS ---\n')
    for cycle in [21, 28, 35, 42, 63, 90]:
        hits = sum(1 for g in gaps if abs(g - cycle) <= 3)
        rate = hits / len(gaps) * 100
        print(f'  T+{cycle} cycle (+/- 3 days): {hits} hits out of {len(gaps)} gaps ({rate:.1f}%)')

