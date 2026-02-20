#!/usr/bin/env python3
"""GME Squeeze Predictor & Next Price Movement Analysis.

Reads cached data from the fraud investigation and optionally fetches
live data from IBKR (borrow rates), FINRA (short volume, threshold list),
and Yahoo (options gamma/max pain) to produce:

1. A forward-looking catalyst timeline with convergence zones
2. A short squeeze probability scorecard with historical comparison

Core dependencies: stdlib only (csv, json, datetime, math, os)
Optional live data: stockdownloader package (auto-detected)
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'gme_holistic_aligned.csv')
CACHE_PATH = os.path.join(DATA_DIR, 'cache', 'gme_cross_reference_analysis.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'cache', 'gme_squeeze_assessment.json')

SHARES_OUTSTANDING = 446_800_000
INSIDERS_EST = 53_616_000  # RC Ventures + other insiders
FLOAT_EST = SHARES_OUTSTANDING - INSIDERS_EST  # ~393.2M

TODAY = date.today()
CATALYST_HORIZON_MONTHS = 4  # Look ahead 4 months

# Quarterly OPEX months (quad-witch)
QUAD_WITCH_MONTHS = {3, 6, 9, 12}

# Historical reference events for comparison
HISTORICAL_EVENTS = {
    'jan_2021_sneeze': {
        'label': 'Jan 2021 Sneeze',
        'start': '2021-01-11',
        'peak': '2021-01-27',
        # Pre-split values (multiply by 4 for post-split equivalent)
        'si_known': 71_200_000 * 4,       # ~284.8M post-split equiv
        'dtc_known': 8.0,
        'borrow_known': 80.0,             # Public reports: 80%+
        'float_known': 51_000_000 * 4,    # ~204M post-split equiv
    },
    'jun_2024_dfv': {
        'label': 'Jun 2024 DFV Return',
        'start': '2024-05-10',
        'peak': '2024-05-14',
    },
    'mar_2025_spike': {
        'label': 'Mar 2025 Spike',
        'start': '2025-02-24',
        'peak': '2025-03-26',
    },
}

# Scoring weights for squeeze probability (base 8 factors)
# When live data adds new factors, weights are normalized automatically.
SQUEEZE_WEIGHTS = {
    'si_pct_float':       0.12,
    'days_to_cover':      0.17,  # Highest weight — most direct squeeze metric
    'cost_to_borrow':     0.13,
    'ftd_accumulation':   0.08,
    'inst_ownership':     0.08,
    'dark_pool_trend':    0.08,
    'sma_position':       0.07,
    'historical_sim':     0.07,
    # New factors (from live data — only scored if data available)
    'options_gamma':      0.10,  # Gamma exposure / max pain
    'short_volume':       0.05,  # Daily short volume ratio
    'threshold_list':     0.05,  # Reg SHO threshold status
}

# ------------------------------------------------------------------
# Live Data Integration (optional — gracefully degrades)
# ------------------------------------------------------------------

# Add parent src/ to path so we can import stockdownloader
_src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
if os.path.isdir(_src_dir):
    sys.path.insert(0, _src_dir)

LIVE_DATA = {}  # Populated by fetch_live_data()


def fetch_live_data(symbol='GME'):
    """Attempt to fetch real-time data from all available sources.

    Each source is independent — failure of one does not affect others.
    Results are stored in the global LIVE_DATA dict.
    """
    global LIVE_DATA

    # 1. IBKR Borrow Rate (real, not estimated)
    try:
        from stockdownloader.data.ibkr_borrow_rate_client import IbkrBorrowRateClient
        client = IbkrBorrowRateClient()
        rate = client.fetch_borrow_rate(symbol)
        if rate is not None:
            LIVE_DATA['ibkr_borrow'] = {
                'fee_rate': rate.fee_rate,
                'available': rate.available,
                'timestamp': rate.timestamp,
                'source': 'IBKR FTP (real)',
            }
            client.append_to_history(symbol, rate)
            print('  [LIVE] IBKR borrow rate: {:.2f}% ({:,} shares available)'.format(
                rate.fee_rate, rate.available,
            ))
        else:
            print('  [LIVE] IBKR: {} not found in shortable list'.format(symbol))
    except Exception as exc:
        print('  [LIVE] IBKR borrow rate: unavailable ({})'.format(exc))

    # 2. Options Gamma / Max Pain — Fallback chain:
    #    Tradier (best Greeks) → Yahoo + Black-Scholes self-compute → Yahoo degraded
    chain = None
    opts_source = None

    # 2a. Try Tradier first (best Greeks — real ORATS-computed)
    try:
        from stockdownloader.data.tradier_options_client import TradierOptionsClient
        tradier = TradierOptionsClient()
        chain = tradier.download(symbol)
        if chain and (chain.total_volume > 0 or chain.total_call_open_interest > 0):
            opts_source = 'Tradier Sandbox (ORATS Greeks)'
            tradier.save_chain_cache(chain)
            print('  [LIVE] Tradier options: {} calls + {} puts, {} expirations'.format(
                len(chain.all_calls), len(chain.all_puts),
                len(chain.expiration_dates),
            ))
        else:
            chain = None
            print('  [LIVE] Tradier: no data for {} — falling back to Yahoo'.format(symbol))
    except Exception as exc:
        print('  [LIVE] Tradier options: unavailable ({}) — falling back to Yahoo'.format(exc))

    # 2b. Fallback: Yahoo with Black-Scholes self-compute
    if chain is None:
        try:
            from stockdownloader.data.yahoo_options_client import YahooOptionsClient
            # compute_greeks=True enables automatic BS Greek computation
            opts_client = YahooOptionsClient(compute_greeks=True)
            chain = opts_client.download(symbol)
            if chain and chain.total_volume > 0:
                opts_source = 'Yahoo Finance + Black-Scholes Greeks'
                print('  [LIVE] Yahoo options: {} calls + {} puts'.format(
                    len(chain.all_calls), len(chain.all_puts),
                ))
            else:
                chain = None
                print('  [LIVE] Yahoo options: no data available for {}'.format(symbol))
        except Exception as exc:
            print('  [LIVE] Yahoo options: unavailable ({})'.format(exc))

    # 2c. Analyze the chain (whichever source won)
    if chain is not None:
        try:
            from stockdownloader.analysis.options_gamma_analyzer import OptionsGammaAnalyzer
            analyzer = OptionsGammaAnalyzer()
            report = analyzer.analyze_and_cache(chain)
            LIVE_DATA['options'] = {
                'max_pain': report.max_pain,
                'max_pain_distance_pct': report.max_pain_distance_pct,
                'net_gex': report.net_gex,
                'pcr_volume': report.pcr_volume,
                'pcr_oi': report.pcr_oi,
                'gamma_wall_call': report.gamma_wall_call,
                'gamma_wall_put': report.gamma_wall_put,
                'gex_signal': report.gex_signal,
                'max_pain_signal': report.max_pain_signal,
                'pcr_signal': report.pcr_signal,
                'overall_signal': report.overall_signal,
                'total_call_oi': report.total_call_oi,
                'total_put_oi': report.total_put_oi,
                'total_call_volume': report.total_call_volume,
                'total_put_volume': report.total_put_volume,
                'unusual_count': len(report.unusual_activity),
                'expirations': report.expirations_analyzed,
                'has_oi': report.has_oi,
                'has_greeks': report.has_greeks,
                'max_pain_source': report.max_pain_source,
                'data_quality': report.data_quality,
                'source': opts_source,
            }
            quality_note = ''
            if report.data_quality != 'full':
                quality_note = ' [{}]'.format(report.data_quality.upper())
                if not report.has_oi:
                    quality_note += ' (no OI — max pain from volume)'
                if not report.has_greeks:
                    quality_note += ' (no Greeks — GEX N/A)'
            print('  [LIVE] Options: max pain=${:.2f} ({}) | net GEX={:,.0f} | PCR={:.2f} | signal={}{}'.format(
                report.max_pain, report.max_pain_source,
                report.net_gex, report.pcr_volume, report.overall_signal,
                quality_note,
            ))
            print('  [LIVE] Options source: {}'.format(opts_source))
        except Exception as exc:
            print('  [LIVE] Options analysis: failed ({})'.format(exc))

    # 3. FINRA Daily Short Volume
    try:
        from stockdownloader.data.finra_short_volume_client import FinraShortVolumeClient
        client = FinraShortVolumeClient()
        stats = client.get_recent_stats(symbol, days=20)
        if stats and stats.get('total_days', 0) > 0:
            LIVE_DATA['short_volume'] = stats
            LIVE_DATA['short_volume']['source'] = 'FINRA Reg SHO Daily (real)'
            print('  [LIVE] Short volume: avg SVR={:.1f}% | latest={:.1f}% | trend={}'.format(
                stats['avg_svr'] * 100, stats['latest_svr'] * 100, stats['svr_trend'],
            ))
        else:
            print('  [LIVE] Short volume: no data available')
    except Exception as exc:
        print('  [LIVE] Short volume: unavailable ({})'.format(exc))

    # 4. Reg SHO Threshold List
    try:
        from stockdownloader.data.regsho_threshold_client import RegShoThresholdClient
        client = RegShoThresholdClient()
        on_list = client.is_currently_on_threshold(symbol)
        records = client.fetch_threshold_status(symbol, lookback_days=90)
        LIVE_DATA['threshold'] = {
            'currently_on_list': on_list,
            'recent_appearances': len(records),
            'dates': [r.date for r in records[-10:]],  # Last 10
            'source': 'FINRA/Nasdaq Reg SHO (real)',
        }
        if on_list:
            print('  [LIVE] Reg SHO: *** CURRENTLY ON THRESHOLD LIST *** ({} recent appearances)'.format(
                len(records),
            ))
        else:
            print('  [LIVE] Reg SHO: not on threshold list ({} appearances in 90 days)'.format(
                len(records),
            ))
    except Exception as exc:
        print('  [LIVE] Reg SHO threshold: unavailable ({})'.format(exc))

    return LIVE_DATA


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------


def load_csv_data():
    """Read aligned CSV into list of dicts + date index."""
    rows = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    date_idx = {}
    for i, r in enumerate(rows):
        date_idx[r['date']] = i

    return rows, date_idx


def load_cache_data():
    """Read cross-reference analysis JSON."""
    with open(CACHE_PATH, 'r') as f:
        return json.load(f)


def safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    """Safely convert to int."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# ------------------------------------------------------------------
# Latest conditions
# ------------------------------------------------------------------


def get_latest_conditions(rows):
    """Extract current state from last row."""
    r = rows[-1]
    return {
        'date': r['date'],
        'close': safe_float(r['close']),
        'volume': safe_int(r['volume']),
        'short_interest': safe_int(r['short_interest']),
        'si_days_to_cover': safe_float(r['si_days_to_cover']),
        'si_avg_daily_vol': safe_int(r['si_avg_daily_vol']),
        'dp_ats_pct': safe_float(r['dp_ats_pct']),
        'dp_ats_volume': safe_int(r['dp_ats_volume']),
        'institutional_shares': safe_int(r['institutional_shares']),
        'num_institutions': safe_int(r['num_institutions']),
        'top10_concentration': safe_float(r['top10_concentration']),
        'borrow_rate_est': safe_float(r['borrow_rate_est']),
        'borrow_utilization': safe_float(r['borrow_utilization']),
        'volatility_20d': safe_float(r['volatility_20d']),
        'sma_50': safe_float(r['sma_50']),
        'sma_200': safe_float(r['sma_200']),
        'ftd_20d_avg': safe_float(r['ftd_20d_avg']),
        'ftd_5d_sum': safe_int(r['ftd_5d_sum']),
        'relative_volume': safe_float(r['relative_volume']),
    }


def get_snapshot_at_date(rows, date_idx, target_date):
    """Get conditions snapshot at or near a historical date."""
    # Try exact match first, then +/- a few days
    for offset in range(0, 10):
        for d in [0, -offset, offset]:
            if d == 0 and offset > 0:
                continue
            check = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=d)).strftime('%Y-%m-%d')
            if check in date_idx:
                return get_latest_conditions([rows[date_idx[check]]])
    return None


# ------------------------------------------------------------------
# Part 1: Catalyst Identification
# ------------------------------------------------------------------


def third_friday(year, month):
    """Compute the third Friday of a given month."""
    first = date(year, month, 1)
    # Days until first Friday (Friday = 4)
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_friday)
    return first_friday + timedelta(days=14)


def find_pending_t35(rows, today):
    """Scan recent rows for FTDs whose T+35 is still future."""
    catalysts = []
    # Scan last 90 rows (roughly 4 months of trading days)
    start = max(0, len(rows) - 90)

    for i in range(start, len(rows)):
        r = rows[i]
        ftd = safe_int(r['ftd_quantity'])
        if ftd < 10000:  # Skip trivial FTDs
            continue

        ftd_date = datetime.strptime(r['date'], '%Y-%m-%d').date()
        t35_date = ftd_date + timedelta(days=35)

        if t35_date >= today:
            ftd_20d = safe_float(r['ftd_20d_avg'], 1)
            spike_ratio = ftd / max(ftd_20d, 1)

            # Score: based on magnitude relative to the 500K "strong signal" threshold
            # and the historical 55.2% hit rate
            magnitude_score = min(ftd / 500000, 1.0) * 55  # Max 55 pts for huge FTDs
            # Bonus for spike ratio (unusual relative to recent avg)
            spike_bonus = min(spike_ratio / 5, 1.0) * 15  # Max 15 pts

            catalysts.append({
                'type': 't35_settlement',
                'date': t35_date.isoformat(),
                'ftd_date': r['date'],
                'ftd_quantity': ftd,
                'ftd_20d_avg': ftd_20d,
                'spike_ratio': round(spike_ratio, 2),
                'score': round(magnitude_score + spike_bonus, 1),
                'description': 'T+35 from {:,} FTD shares on {}'.format(ftd, r['date']),
            })

    return catalysts


def compute_opex_catalysts(today, months_ahead=4):
    """Generate upcoming OPEX dates as catalysts."""
    catalysts = []
    year, month = today.year, today.month

    for _ in range(months_ahead):
        opex = third_friday(year, month)
        if opex >= today:
            is_quarterly = month in QUAD_WITCH_MONTHS
            score = 25 if is_quarterly else 15
            catalysts.append({
                'type': 'opex',
                'date': opex.isoformat(),
                'is_quarterly': is_quarterly,
                'score': score,
                'description': '{} OPEX ({})'.format(
                    'QUARTERLY' if is_quarterly else 'Monthly',
                    opex.strftime('%b %d, %Y'),
                ),
            })

        month += 1
        if month > 12:
            month = 1
            year += 1

    return catalysts


def compute_ats_trend(rows, lookback=90):
    """Determine ATS% direction and magnitude from recent distinct transitions.

    Uses two windows:
    - Short-term (last 6 distinct values, ~6 weeks) for recent momentum
    - Medium-term (last 12 distinct values, ~12 weeks) for broader trend
    Then uses the stronger signal of the two.
    """
    # Find distinct ATS% values in the last N rows
    start = max(0, len(rows) - lookback)
    distinct = []
    prev_val = -1.0

    for i in range(start, len(rows)):
        dp = safe_float(rows[i]['dp_ats_pct'])
        if dp > 0 and dp != prev_val:
            distinct.append({
                'date': rows[i]['date'],
                'ats_pct': dp,
                'idx': i,
            })
            prev_val = dp

    if len(distinct) < 2:
        return {'direction': 'neutral', 'magnitude_pp': 0.0, 'recent_values': distinct}

    # Short-term: last 6 distinct values
    short = distinct[-min(6, len(distinct)):]
    short_change = short[-1]['ats_pct'] - short[0]['ats_pct']

    # Medium-term: last 12 distinct values (or all if fewer)
    medium = distinct[-min(12, len(distinct)):]
    medium_change = medium[-1]['ats_pct'] - medium[0]['ats_pct']

    # Also compute avg of first half vs second half of medium window
    mid = len(medium) // 2
    first_half_avg = sum(d['ats_pct'] for d in medium[:mid]) / max(mid, 1)
    second_half_avg = sum(d['ats_pct'] for d in medium[mid:]) / max(len(medium) - mid, 1)
    half_avg_change = second_half_avg - first_half_avg

    # Use the strongest signal
    # Priority: if medium-term shows > 3pp change, use it; else use short-term
    if abs(medium_change) > abs(short_change):
        primary_change = medium_change
        primary_window = medium
    else:
        primary_change = short_change
        primary_window = short

    # Also factor in the half-average trend
    combined_signal = (primary_change + half_avg_change) / 2

    if combined_signal < -3:
        direction = 'bullish'
    elif combined_signal > 3:
        direction = 'bearish'
    elif combined_signal < -1.5:
        direction = 'slightly bullish'
    elif combined_signal > 1.5:
        direction = 'slightly bearish'
    else:
        direction = 'neutral'

    return {
        'direction': direction,
        'magnitude_pp': round(combined_signal, 2),
        'short_term_change_pp': round(short_change, 2),
        'medium_term_change_pp': round(medium_change, 2),
        'half_avg_change_pp': round(half_avg_change, 2),
        'first_value': round(primary_window[0]['ats_pct'], 2),
        'last_value': round(primary_window[-1]['ats_pct'], 2),
        'first_date': primary_window[0]['date'],
        'last_date': primary_window[-1]['date'],
        'recent_values': distinct[-min(8, len(distinct)):],
    }


def compute_seasonal_score(today, cache):
    """Score current month by historical FTD clustering."""
    ftd_by_month = cache.get('ftd_by_month', {})
    if not ftd_by_month:
        return 0, 'No seasonal data'

    # Month names: Jan, Feb, Mar, ...
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_str = month_names[today.month - 1]

    # Get monthly totals
    totals = {}
    for m, val in ftd_by_month.items():
        if isinstance(val, dict):
            totals[m] = val.get('total', 0)
        else:
            totals[m] = val

    if month_str not in totals:
        return 0, 'No data for {}'.format(month_str)

    current_total = totals.get(month_str, 0)
    avg_total = sum(totals.values()) / len(totals) if totals else 1
    ratio = current_total / max(avg_total, 1)

    # Rank current month
    sorted_months = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, (m, _) in enumerate(sorted_months) if m == month_str), 0)

    # Scale to 0-20 score
    score = min(ratio * 10, 20)

    # Also show upcoming high-FTD months
    next_month_idx = today.month % 12  # 0-indexed for next month
    next_month = month_names[next_month_idx]
    next_total = totals.get(next_month, 0)
    next_ratio = next_total / max(avg_total, 1)

    return round(score, 1), '{} FTD rank: #{}/12 ({:.2f}x avg), next month {} ({:.2f}x avg)'.format(
        month_str, rank, ratio, next_month, next_ratio,
    )


def compute_sma_catalyst(rows):
    """Analyze price vs moving averages for breakout potential."""
    r = rows[-1]
    close = safe_float(r['close'])
    sma_50 = safe_float(r['sma_50'])
    sma_200 = safe_float(r['sma_200'])

    result = {
        'close': close,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'above_sma_50': close > sma_50,
        'above_sma_200': close > sma_200,
        'pct_from_sma_200': round(((close - sma_200) / sma_200) * 100, 2) if sma_200 > 0 else 0,
        'sma_gap_pct': round(((sma_50 - sma_200) / sma_200) * 100, 2) if sma_200 > 0 else 0,
    }

    score = 0
    notes = []

    if close > sma_50:
        score += 10
        notes.append('Above SMA-50')
    if close > sma_200:
        score += 15
        notes.append('Above SMA-200')
    elif abs(result['pct_from_sma_200']) < 3:
        score += 10
        notes.append('Within 3% of SMA-200 ({:+.1f}%)'.format(result['pct_from_sma_200']))

    # Golden cross setup: SMA-50 approaching SMA-200 from below
    if sma_50 < sma_200 and result['sma_gap_pct'] > -5:
        score += 5
        notes.append('Golden cross potential (gap: {:.1f}%)'.format(result['sma_gap_pct']))
    elif sma_50 > sma_200:
        score += 8
        notes.append('Already golden crossed')

    result['score'] = score
    result['notes'] = notes
    return result


# ------------------------------------------------------------------
# Convergence Engine
# ------------------------------------------------------------------


def find_convergence_zones(catalysts, window_days=5):
    """Group catalysts within N calendar days into convergence zones."""
    if not catalysts:
        return []

    # Sort by date
    sorted_cats = sorted(catalysts, key=lambda c: c['date'])

    zones = []
    current_zone = [sorted_cats[0]]
    zone_end = datetime.strptime(sorted_cats[0]['date'], '%Y-%m-%d').date() + timedelta(days=window_days)

    for cat in sorted_cats[1:]:
        cat_date = datetime.strptime(cat['date'], '%Y-%m-%d').date()
        if cat_date <= zone_end:
            current_zone.append(cat)
            # Extend zone end if this catalyst is later
            new_end = cat_date + timedelta(days=window_days)
            zone_end = max(zone_end, new_end)
        else:
            zones.append(current_zone)
            current_zone = [cat]
            zone_end = cat_date + timedelta(days=window_days)

    if current_zone:
        zones.append(current_zone)

    # Score each zone
    scored_zones = []
    for zone_catalysts in zones:
        dates = [c['date'] for c in zone_catalysts]
        base_score = sum(c.get('score', 0) for c in zone_catalysts)

        # Convergence multiplier
        n = len(zone_catalysts)
        if n >= 3:
            multiplier = 2.0
        elif n >= 2:
            multiplier = 1.5
        else:
            multiplier = 1.0

        combined_score = base_score * multiplier
        types = list(set(c['type'] for c in zone_catalysts))

        scored_zones.append({
            'window_start': min(dates),
            'window_end': max(dates),
            'catalysts': zone_catalysts,
            'num_catalysts': n,
            'convergence_multiplier': multiplier,
            'base_score': round(base_score, 1),
            'combined_score': round(combined_score, 1),
            'catalyst_types': types,
        })

    # Sort by combined score descending
    scored_zones.sort(key=lambda z: z['combined_score'], reverse=True)
    return scored_zones


# ------------------------------------------------------------------
# Part 2: Squeeze Probability Factors
# ------------------------------------------------------------------


def score_si_pct_float(current):
    """Factor 1: SI as % of float."""
    si = current['short_interest']
    si_pct = (si / FLOAT_EST) * 100

    # Scoring curve: >100% = 100, >50% = 80, >20% = 50, >10% = 30
    if si_pct >= 100:
        score = 100
    elif si_pct >= 50:
        score = 80 + (si_pct - 50) / 50 * 20
    elif si_pct >= 20:
        score = 50 + (si_pct - 20) / 30 * 30
    elif si_pct >= 10:
        score = 25 + (si_pct - 10) / 10 * 25
    else:
        score = si_pct / 10 * 25

    return {
        'name': 'SI % of Float',
        'value': round(si_pct, 1),
        'unit': '%',
        'score': round(min(score, 100), 1),
        'detail': '{:,} SI / {:,} float = {:.1f}%'.format(si, FLOAT_EST, si_pct),
    }


def score_days_to_cover(current):
    """Factor 2: Days-to-cover ratio."""
    dtc = current['si_days_to_cover']

    # Scoring: DTC > 15 = 100, DTC > 10 = 80, DTC > 5 = 50, DTC > 2 = 25
    if dtc >= 15:
        score = 100
    elif dtc >= 10:
        score = 80 + (dtc - 10) / 5 * 20
    elif dtc >= 5:
        score = 50 + (dtc - 5) / 5 * 30
    elif dtc >= 2:
        score = 20 + (dtc - 2) / 3 * 30
    else:
        score = dtc / 2 * 20

    return {
        'name': 'Days-to-Cover',
        'value': round(dtc, 2),
        'unit': 'days',
        'score': round(min(score, 100), 1),
        'detail': '{:.2f} days to cover all short positions'.format(dtc),
    }


def score_cost_to_borrow(current):
    """Factor 3: Cost to borrow (uses IBKR real rate if available)."""
    rate = current.get('borrow_rate_real', current['borrow_rate_est'])
    source = current.get('borrow_source', 'estimated')

    # Scoring: >60% = 95, >40% = 80, >20% = 60, >10% = 40, >5% = 25
    if rate >= 60:
        score = 95
    elif rate >= 40:
        score = 80 + (rate - 40) / 20 * 15
    elif rate >= 20:
        score = 55 + (rate - 20) / 20 * 25
    elif rate >= 10:
        score = 35 + (rate - 10) / 10 * 20
    elif rate >= 5:
        score = 20 + (rate - 5) / 5 * 15
    else:
        score = rate / 5 * 20

    return {
        'name': 'Cost to Borrow',
        'value': round(rate, 2),
        'unit': '%',
        'score': round(min(score, 100), 1),
        'detail': '{} annual borrow fee: {:.2f}%'.format(
            'IBKR real' if 'real' in source.lower() else 'Estimated', rate,
        ),
    }


def score_ftd_accumulation(rows, current):
    """Factor 4: FTD accumulation trend."""
    # Compare recent FTD levels to historical
    ftd_20d = current['ftd_20d_avg']
    ftd_5d = current['ftd_5d_sum']

    # Compute historical percentile of current ftd_20d_avg
    all_ftd_20d = []
    for r in rows[-500:]:  # Last 2 years
        v = safe_float(r['ftd_20d_avg'])
        if v > 0:
            all_ftd_20d.append(v)

    if all_ftd_20d:
        all_ftd_20d.sort()
        percentile = sum(1 for x in all_ftd_20d if x <= ftd_20d) / len(all_ftd_20d) * 100
    else:
        percentile = 50

    # Check for recent spike (last 30 rows)
    recent_max_ftd = 0
    recent_max_date = ''
    for r in rows[-30:]:
        ftd = safe_int(r['ftd_quantity'])
        if ftd > recent_max_ftd:
            recent_max_ftd = ftd
            recent_max_date = r['date']

    # Are FTDs building or declining?
    first_half_avg = 0
    second_half_avg = 0
    half = 15
    for r in rows[-30:-half]:
        first_half_avg += safe_float(r['ftd_20d_avg'])
    for r in rows[-half:]:
        second_half_avg += safe_float(r['ftd_20d_avg'])
    first_half_avg /= max(half, 1)
    second_half_avg /= max(half, 1)

    trend = 'building' if second_half_avg > first_half_avg * 1.1 else (
        'declining' if second_half_avg < first_half_avg * 0.9 else 'stable'
    )

    # Score: based on percentile + trend bonus
    score = percentile * 0.7
    if trend == 'building':
        score += 15
    elif trend == 'declining':
        score -= 10

    # Bonus for recent spike
    if recent_max_ftd > 500000:
        score += 15
    elif recent_max_ftd > 100000:
        score += 5

    return {
        'name': 'FTD Accumulation',
        'value': round(ftd_20d, 0),
        'unit': '20d avg',
        'score': round(max(0, min(score, 100)), 1),
        'detail': '20d avg: {:,.0f} ({:.0f}th percentile), trend: {}, recent max: {:,} on {}'.format(
            ftd_20d, percentile, trend, recent_max_ftd, recent_max_date,
        ),
    }


def score_institutional_ownership(current):
    """Factor 5: Institutional ownership (inverse — lower = harder to borrow)."""
    inst = current['institutional_shares']
    inst_pct = (inst / SHARES_OUTSTANDING) * 100

    # Lower institutional ownership = fewer lendable shares = higher squeeze potential
    # Inverse score: 0% institutional = 100, 100% = 0
    score = max(0, 100 - inst_pct)

    # Bonus if concentration is high (fewer lenders to coordinate with)
    top10 = current['top10_concentration']
    if top10 > 60:
        score += 10
    elif top10 > 50:
        score += 5

    return {
        'name': 'Inst. Ownership (inverse)',
        'value': round(inst_pct, 1),
        'unit': '% of outstanding',
        'score': round(max(0, min(score, 100)), 1),
        'detail': '{:,} shares across {} institutions (top 10: {:.1f}%)'.format(
            inst, current['num_institutions'], top10,
        ),
    }


def score_dark_pool_trend(ats_trend):
    """Factor 6: Dark pool ATS% trend."""
    direction = ats_trend['direction']
    magnitude = abs(ats_trend['magnitude_pp'])

    if 'bullish' in direction:
        # Decreasing ATS% = more price discovery = bullish for squeeze
        score = 40 + min(magnitude / 10, 1.0) * 40  # 40-80
        if direction == 'slightly bullish':
            score = min(score, 60)  # Cap for "slightly"
    elif 'bearish' in direction:
        # Increasing ATS% = buy suppression = bearish
        score = max(0, 40 - min(magnitude / 10, 1.0) * 30)  # 10-40
    else:
        score = 40  # Neutral

    return {
        'name': 'Dark Pool Trend',
        'value': ats_trend['magnitude_pp'],
        'unit': 'pp change',
        'score': round(score, 1),
        'detail': 'ATS% moved {:.1f}pp ({:.1f}% -> {:.1f}%) = {}'.format(
            ats_trend['magnitude_pp'],
            ats_trend.get('first_value', 0),
            ats_trend.get('last_value', 0),
            direction.upper(),
        ),
    }


def score_sma_position(sma_data):
    """Factor 7: Price vs SMA position."""
    return {
        'name': 'Price vs SMA',
        'value': sma_data['pct_from_sma_200'],
        'unit': '% from SMA-200',
        'score': round(min(sma_data['score'] * 2.5, 100), 1),  # Scale 0-40 score to 0-100
        'detail': '${:.2f} (SMA-50: ${:.2f}, SMA-200: ${:.2f}). {}'.format(
            sma_data['close'], sma_data['sma_50'], sma_data['sma_200'],
            '; '.join(sma_data['notes']),
        ),
    }


def score_historical_similarity(current, rows, date_idx):
    """Factor 8: Compare current conditions to pre-squeeze periods."""
    comparisons = {}
    similarity_scores = []

    for event_key, event in HISTORICAL_EVENTS.items():
        snap = get_snapshot_at_date(rows, date_idx, event['start'])
        if snap is None:
            comparisons[event_key] = {'label': event['label'], 'available': False}
            continue

        comp = {'label': event['label'], 'available': True}

        # Compare each key metric
        # DTC comparison (higher DTC = more squeeze-like)
        snap_dtc = snap['si_days_to_cover']
        # For Jan 2021, use known value since CSV may be zero
        if event_key == 'jan_2021_sneeze':
            snap_dtc = event.get('dtc_known', snap_dtc)
            snap_borrow = event.get('borrow_known', snap['borrow_rate_est'])
            snap_si = event.get('si_known', snap['short_interest'])
        else:
            snap_borrow = snap['borrow_rate_est']
            snap_si = snap['short_interest']

        curr_dtc = current['si_days_to_cover']
        dtc_sim = min(curr_dtc / max(snap_dtc, 0.01), 1.5) / 1.5 * 100

        # Borrow rate comparison
        curr_borrow = current['borrow_rate_est']
        borrow_sim = min(curr_borrow / max(snap_borrow, 0.01), 1.5) / 1.5 * 100

        # SI comparison — use event-era float for historical events
        curr_si_pct = current['short_interest'] / FLOAT_EST * 100
        event_float = event.get('float_known', FLOAT_EST)
        snap_si_pct = snap_si / event_float * 100
        si_sim = min(curr_si_pct / max(snap_si_pct, 0.01), 1.5) / 1.5 * 100

        # ATS% comparison
        snap_ats = snap['dp_ats_pct']
        curr_ats = current['dp_ats_pct']
        ats_sim = 100 - abs(curr_ats - snap_ats) * 2  # Penalize divergence

        avg_sim = (dtc_sim + borrow_sim + si_sim + max(ats_sim, 0)) / 4

        comp['snapshot_date'] = event['start']
        comp['metrics'] = {
            'dtc': {'pre_event': round(snap_dtc, 2), 'current': round(curr_dtc, 2), 'similarity': round(dtc_sim, 1)},
            'borrow_rate': {'pre_event': round(snap_borrow, 2), 'current': round(curr_borrow, 2), 'similarity': round(borrow_sim, 1)},
            'si_pct_float': {'pre_event': round(snap_si_pct, 1), 'current': round(curr_si_pct, 1), 'similarity': round(si_sim, 1)},
            'ats_pct': {'pre_event': round(snap_ats, 1), 'current': round(curr_ats, 1), 'similarity': round(max(ats_sim, 0), 1)},
        }
        comp['avg_similarity'] = round(avg_sim, 1)
        comparisons[event_key] = comp
        similarity_scores.append(avg_sim)

    overall = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

    return {
        'name': 'Historical Similarity',
        'value': round(overall, 1),
        'unit': 'avg %',
        'score': round(max(0, min(overall, 100)), 1),
        'detail': 'Avg similarity to pre-squeeze conditions: {:.1f}%'.format(overall),
        'comparisons': comparisons,
    }


# ------------------------------------------------------------------
# New Factors (from live data)
# ------------------------------------------------------------------


def score_options_gamma():
    """Factor 9: Options gamma exposure, max pain, and put/call ratio.

    Only scored when live options data is available.
    Automatically adjusts scoring based on data quality:
    - Full quality: GEX (40%) + Max Pain (30%) + PCR (30%)
    - No Greeks: Max Pain (50%) + PCR (50%), GEX not scored
    - No OI: Max Pain from volume (40%) + PCR volume-only (60%)
    Returns None if data is not available.
    """
    opts = LIVE_DATA.get('options')
    if not opts:
        return None

    has_greeks = opts.get('has_greeks', True)
    has_oi = opts.get('has_oi', True)
    quality = opts.get('data_quality', 'full')
    mp_source = opts.get('max_pain_source', 'oi')

    # Score GEX (only if Greeks available)
    gex_score = 0
    gex_weight = 0.0
    if has_greeks:
        gex_weight = 0.4
        if opts['gex_signal'] == 'amplifying':
            gex_score = 80  # Gamma squeeze potential!
        elif opts['gex_signal'] == 'neutral':
            gex_score = 50
        else:
            gex_score = 25  # Dampening

    # Score Max Pain
    mp_weight = 0.3 if has_greeks else 0.5
    mp_score = 0
    mp_dist = opts['max_pain_distance_pct']
    if mp_dist > 5:
        mp_score = 80  # Max pain significantly above price
    elif mp_dist > 0:
        mp_score = 60
    elif mp_dist > -3:
        mp_score = 40
    else:
        mp_score = 20  # Max pain far below price
    # Reduce confidence for volume-based max pain
    if mp_source == 'volume':
        mp_score = mp_score * 0.7 + 50 * 0.3  # Blend toward neutral

    # Score PCR
    pcr_weight = 0.3 if has_greeks else 0.5
    pcr_score = 0
    pcr = opts['pcr_volume']
    if pcr > 1.5:
        pcr_score = 80  # Extreme put buying = squeeze fuel
    elif pcr > 1.0:
        pcr_score = 65
    elif pcr > 0.7:
        pcr_score = 50
    else:
        pcr_score = 30  # Heavy call buying

    total_weight = gex_weight + mp_weight + pcr_weight
    combined = (gex_score * gex_weight + mp_score * mp_weight + pcr_score * pcr_weight) / total_weight

    # Build detail string
    quality_tag = ''
    if quality != 'full':
        quality_tag = ' [{}]'.format(quality)

    gex_label = opts['gex_signal'].upper() if has_greeks else 'N/A (no Greeks)'
    mp_label = '${:.2f}'.format(opts['max_pain'])
    if mp_source == 'volume':
        mp_label += ' (vol-based)'

    return {
        'name': 'Options Gamma/Flow',
        'value': round(opts['net_gex'], 0),
        'unit': 'net GEX',
        'score': round(combined, 1),
        'detail': 'GEX: {} | MaxPain: {} ({:+.1f}% from price) | PCR: {:.2f} | Signal: {}{}'.format(
            gex_label, mp_label,
            opts['max_pain_distance_pct'], opts['pcr_volume'],
            opts['overall_signal'], quality_tag,
        ),
    }


def score_short_volume():
    """Factor 10: Daily short volume ratio.

    Only scored when live FINRA data is available.
    Returns None if data is not available.
    """
    sv = LIVE_DATA.get('short_volume')
    if not sv or sv.get('total_days', 0) == 0:
        return None

    avg_svr = sv['avg_svr']
    latest_svr = sv['latest_svr']
    trend = sv['svr_trend']

    # Score: SVR > 60% sustained + increasing trend = high pressure
    base_score = 0
    if latest_svr > 0.65:
        base_score = 80
    elif latest_svr > 0.55:
        base_score = 60
    elif latest_svr > 0.45:
        base_score = 40
    else:
        base_score = 25

    # Trend bonus
    if trend == 'increasing':
        base_score = min(base_score + 10, 100)
    elif trend == 'decreasing':
        base_score = max(base_score - 10, 0)

    return {
        'name': 'Daily Short Volume',
        'value': round(latest_svr * 100, 1),
        'unit': '% SVR',
        'score': round(base_score, 1),
        'detail': 'Latest SVR: {:.1f}% | 20d avg: {:.1f}% | Trend: {} | {}/{} days >50%'.format(
            latest_svr * 100, avg_svr * 100, trend,
            sv.get('days_above_50pct', 0), sv.get('total_days', 0),
        ),
    }


def score_threshold_list():
    """Factor 11: Reg SHO threshold list status.

    Being on the threshold list is a VERY strong signal — it means
    FTDs have exceeded 0.5% of shares outstanding for 5+ consecutive days.
    Returns None if data is not available.
    """
    thr = LIVE_DATA.get('threshold')
    if thr is None:
        return None

    on_list = thr['currently_on_list']
    recent = thr['recent_appearances']

    if on_list:
        score = 95  # Currently on threshold list = extremely bullish for squeeze
    elif recent >= 5:
        score = 70  # Frequently on the list recently
    elif recent >= 1:
        score = 50  # Was on the list recently
    else:
        score = 20  # Not on the list

    status = 'ON LIST' if on_list else 'Not on list'
    return {
        'name': 'Reg SHO Threshold',
        'value': 1 if on_list else 0,
        'unit': 'on list' if on_list else 'off list',
        'score': round(score, 1),
        'detail': '{} | {} appearances in last 90 days'.format(status, recent),
    }


# ------------------------------------------------------------------
# Override borrow rate with IBKR real data
# ------------------------------------------------------------------


def get_real_borrow_rate(current):
    """Override the estimated borrow rate with IBKR real data if available.

    Returns the (possibly updated) current conditions dict.
    """
    ibkr = LIVE_DATA.get('ibkr_borrow')
    if ibkr:
        old_rate = current['borrow_rate_est']
        current['borrow_rate_real'] = ibkr['fee_rate']
        current['borrow_shares_available'] = ibkr['available']
        current['borrow_source'] = 'IBKR (real)'
        print('\n  [UPDATE] Borrow rate: {:.2f}% (estimated) -> {:.2f}% (IBKR real)'.format(
            old_rate, ibkr['fee_rate'],
        ))
        if ibkr['available'] > 0:
            print('  [UPDATE] Shares available to short: {:,}'.format(ibkr['available']))
    return current


# ------------------------------------------------------------------
# Overall Squeeze Score
# ------------------------------------------------------------------


def compute_squeeze_score(factors):
    """Compute weighted average of all factors.

    Automatically normalizes weights based on which factors are present,
    so the total always sums to 1.0 regardless of which live data
    sources are available.
    """
    factor_map = {f['name']: f for f in factors}
    name_to_key = {
        'SI % of Float': 'si_pct_float',
        'Days-to-Cover': 'days_to_cover',
        'Cost to Borrow': 'cost_to_borrow',
        'FTD Accumulation': 'ftd_accumulation',
        'Inst. Ownership (inverse)': 'inst_ownership',
        'Dark Pool Trend': 'dark_pool_trend',
        'Price vs SMA': 'sma_position',
        'Historical Similarity': 'historical_sim',
        'Options Gamma/Flow': 'options_gamma',
        'Daily Short Volume': 'short_volume',
        'Reg SHO Threshold': 'threshold_list',
    }

    weighted_sum = 0
    total_weight = 0
    for factor in factors:
        key = name_to_key.get(factor['name'])
        if key:
            weight = SQUEEZE_WEIGHTS.get(key, 0)
            weighted_sum += factor['score'] * weight
            total_weight += weight

    return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0


def interpret_score(score):
    """Convert score to interpretation text."""
    if score >= 85:
        return 'EXTREME', 'Conditions are extremely conducive to a short squeeze. All major indicators are flashing red for short sellers.'
    elif score >= 70:
        return 'HIGH', 'Short squeeze probability is high. Multiple indicators suggest shorts are under significant pressure.'
    elif score >= 55:
        return 'ELEVATED', 'Conditions are elevated for a potential squeeze. Key metrics like DTC and borrow costs are notably high, though not all factors align.'
    elif score >= 40:
        return 'MODERATE', 'Moderate squeeze potential. Some indicators are elevated but conditions are not as extreme as prior squeeze events.'
    elif score >= 20:
        return 'LOW', 'Low squeeze probability. Most indicators are within normal ranges.'
    else:
        return 'VERY LOW', 'Short squeeze is very unlikely under current conditions.'


# ------------------------------------------------------------------
# Output: Console
# ------------------------------------------------------------------


def print_header():
    """Print report header."""
    print()
    print('=' * 80)
    print('  GME SQUEEZE PREDICTOR')
    print('  Analysis Date: {}'.format(TODAY.isoformat()))
    print('=' * 80)


def print_current_conditions(current):
    """Print snapshot of current conditions."""
    print('\n  CURRENT CONDITIONS (as of {}):'.format(current['date']))
    print('  ' + '-' * 60)
    print('  Price:           ${:.2f}'.format(current['close']))
    print('  Short Interest:  {:,} shares ({:.1f}% of float)'.format(
        current['short_interest'],
        current['short_interest'] / FLOAT_EST * 100,
    ))
    print('  Days-to-Cover:   {:.2f} days'.format(current['si_days_to_cover']))
    if 'borrow_rate_real' in current:
        print('  Borrow Rate:     {:.2f}% (IBKR real) [was {:.2f}% estimated]'.format(
            current['borrow_rate_real'], current['borrow_rate_est'],
        ))
        if current.get('borrow_shares_available', 0) > 0:
            print('  Shares to Short: {:,}'.format(current['borrow_shares_available']))
    else:
        print('  Borrow Rate Est: {:.2f}% (heuristic estimate)'.format(current['borrow_rate_est']))
    print('  Dark Pool ATS%:  {:.2f}%'.format(current['dp_ats_pct']))
    print('  Institutional:   {:,} shares ({} institutions)'.format(
        current['institutional_shares'], current['num_institutions'],
    ))
    print('  SMA-50:          ${:.2f}'.format(current['sma_50']))
    print('  SMA-200:         ${:.2f}'.format(current['sma_200']))
    print('  Volatility 20d:  {:.2f}%'.format(current['volatility_20d']))


def print_catalyst_timeline(zones, ats_trend, sma_data, seasonal_score, seasonal_desc):
    """Print Part 1 output."""
    print('\n')
    print('=' * 80)
    print('  PART 1: UPCOMING CATALYST TIMELINE')
    print('=' * 80)

    # Directional filter
    print('\n  DIRECTIONAL FILTERS:')
    print('  ' + '-' * 60)
    dir_map = {'bullish': '+', 'slightly bullish': '+', 'bearish': '-', 'slightly bearish': '-', 'neutral': '~'}
    dir_sym = dir_map.get(ats_trend['direction'], '~')
    print('  [{}] Dark Pool ATS% trend: {} (combined: {:+.1f}pp | short: {:+.1f}pp | medium: {:+.1f}pp)'.format(
        dir_sym,
        ats_trend['direction'].upper(),
        ats_trend['magnitude_pp'],
        ats_trend.get('short_term_change_pp', 0),
        ats_trend.get('medium_term_change_pp', 0),
    ))
    print('      Window: {} to {}'.format(
        ats_trend.get('first_date', '?'),
        ats_trend.get('last_date', '?'),
    ))
    sma_dir = 'BULLISH' if sma_data['above_sma_50'] and sma_data['above_sma_200'] else (
        'NEUTRAL-BULLISH' if sma_data['above_sma_50'] else 'BEARISH'
    )
    print('  [{}] Price vs SMAs: {} ({})'.format(
        '+' if 'BULLISH' in sma_dir else '~',
        sma_dir,
        '; '.join(sma_data['notes']),
    ))
    print('  [~] Seasonal: {} (score: {}/20)'.format(seasonal_desc, seasonal_score))

    if not zones:
        print('\n  No catalyst zones identified.')
        return

    print('\n  CATALYST CONVERGENCE ZONES (ranked by combined score):')
    print('  ' + '-' * 60)

    for i, zone in enumerate(zones):
        rating = 'HIGH' if zone['combined_score'] >= 50 else (
            'MODERATE' if zone['combined_score'] >= 25 else 'LOW'
        )
        print('\n  ZONE {} ({}) | {} to {} | Score: {:.0f}/100'.format(
            i + 1, rating,
            zone['window_start'], zone['window_end'],
            zone['combined_score'],
        ))
        print('  Catalysts: {} | Convergence: {:.1f}x'.format(
            zone['num_catalysts'], zone['convergence_multiplier'],
        ))

        for cat in zone['catalysts']:
            prefix = '    '
            if cat['type'] == 't35_settlement':
                print('{} T+35: {:,} FTD from {} -> settlement {} (score: {:.0f})'.format(
                    prefix, cat['ftd_quantity'], cat['ftd_date'],
                    cat['date'], cat.get('score', 0),
                ))
            elif cat['type'] == 'opex':
                qual = 'QUARTERLY' if cat.get('is_quarterly') else 'Monthly'
                print('{} {} OPEX: {} (score: {:.0f})'.format(
                    prefix, qual, cat['date'], cat.get('score', 0),
                ))
            else:
                print('{} {}: {} (score: {:.0f})'.format(
                    prefix, cat['type'], cat.get('description', ''),
                    cat.get('score', 0),
                ))


def print_squeeze_scorecard(factors, overall_score, interpretation):
    """Print Part 2 output."""
    level, explanation = interpretation

    print('\n')
    print('=' * 80)
    print('  PART 2: SHORT SQUEEZE PROBABILITY ASSESSMENT')
    print('=' * 80)

    print('\n  SCORECARD:')
    print('  ' + '-' * 76)
    print('  {:35s} {:>8s} {:>8s} {:>10s}'.format('Factor', 'Score', 'Weight', 'Weighted'))
    print('  ' + '-' * 76)

    name_to_key = {
        'SI % of Float': 'si_pct_float',
        'Days-to-Cover': 'days_to_cover',
        'Cost to Borrow': 'cost_to_borrow',
        'FTD Accumulation': 'ftd_accumulation',
        'Inst. Ownership (inverse)': 'inst_ownership',
        'Dark Pool Trend': 'dark_pool_trend',
        'Price vs SMA': 'sma_position',
        'Historical Similarity': 'historical_sim',
        'Options Gamma/Flow': 'options_gamma',
        'Daily Short Volume': 'short_volume',
        'Reg SHO Threshold': 'threshold_list',
    }
    for factor in factors:
        key = name_to_key.get(factor['name'], '')
        weight = SQUEEZE_WEIGHTS.get(key, 0)
        weighted = factor['score'] * weight

        val_str = '{} {}'.format(factor['value'], factor['unit'])
        print('  {:35s} {:>6.1f}   {:>5.0f}%   {:>8.1f}'.format(
            factor['name'] + ' (' + val_str + ')',
            factor['score'], weight * 100, weighted,
        ))

    print('  ' + '-' * 76)
    print('  {:35s} {:>6.1f} / 100'.format('OVERALL SQUEEZE SCORE', overall_score))
    print('  ' + '-' * 76)

    print('\n  INTERPRETATION: {}'.format(level))
    print('  {}'.format(explanation))


def print_historical_comparison(hist_factor):
    """Print historical comparison table."""
    comparisons = hist_factor.get('comparisons', {})
    if not comparisons:
        return

    print('\n  HISTORICAL COMPARISON:')
    print('  ' + '-' * 76)
    print('  {:20s} {:>12s} {:>12s} {:>12s} {:>12s}'.format(
        'Metric', 'Current', 'Pre-Jan 2021', 'Pre-Jun 2024', 'Pre-Mar 2025',
    ))
    print('  ' + '-' * 76)

    metrics = ['dtc', 'borrow_rate', 'si_pct_float', 'ats_pct']
    metric_labels = {
        'dtc': 'Days-to-Cover',
        'borrow_rate': 'Borrow Rate %',
        'si_pct_float': 'SI % of Float',
        'ats_pct': 'ATS %',
    }

    for metric in metrics:
        row_parts = ['  {:20s}'.format(metric_labels.get(metric, metric))]

        # Current value (from first available comparison)
        current_val = None
        for comp in comparisons.values():
            if comp.get('available') and metric in comp.get('metrics', {}):
                current_val = comp['metrics'][metric]['current']
                break

        if current_val is not None:
            row_parts.append('{:>12}'.format('{:.1f}'.format(current_val) if isinstance(current_val, float) else str(current_val)))
        else:
            row_parts.append('{:>12s}'.format('N/A'))

        # Each historical event
        for event_key in ['jan_2021_sneeze', 'jun_2024_dfv', 'mar_2025_spike']:
            comp = comparisons.get(event_key, {})
            if comp.get('available') and metric in comp.get('metrics', {}):
                val = comp['metrics'][metric]['pre_event']
                row_parts.append('{:>12}'.format('{:.1f}'.format(val) if isinstance(val, float) else str(val)))
            else:
                row_parts.append('{:>12s}'.format('N/A'))

        print(''.join(row_parts))

    # Similarity row
    print('  ' + '-' * 76)
    sim_parts = ['  {:20s}'.format('Similarity Score')]
    sim_parts.append('{:>12s}'.format('--'))
    for event_key in ['jan_2021_sneeze', 'jun_2024_dfv', 'mar_2025_spike']:
        comp = comparisons.get(event_key, {})
        if comp.get('available'):
            sim_parts.append('{:>12}'.format('{:.1f}%'.format(comp['avg_similarity'])))
        else:
            sim_parts.append('{:>12s}'.format('N/A'))
    print(''.join(sim_parts))


def print_key_insights(current, factors, zones, ats_trend):
    """Print actionable key insights."""
    print('\n')
    print('=' * 80)
    print('  KEY INSIGHTS')
    print('=' * 80)

    insights = []

    # DTC insight
    dtc = current['si_days_to_cover']
    if dtc > 10:
        insights.append(
            'CRITICAL: Days-to-cover is {:.1f} days -- this means it would take over {} '
            'trading weeks of average volume for ALL shorts to close. '
            'This is in the top percentile historically.'.format(dtc, int(dtc / 5))
        )

    # Borrow rate insight
    borrow = current['borrow_rate_est']
    if borrow > 30:
        annual_cost = current['short_interest'] * current['close'] * (borrow / 100)
        insights.append(
            'SIGNIFICANT: Estimated borrow rate is {:.1f}%, meaning shorts are paying '
            'an estimated ${:,.0f}/year to maintain {:,} short shares. '
            'This creates continuous pressure to close positions.'.format(
                borrow, annual_cost, current['short_interest'],
            )
        )

    # ATS trend insight
    if ats_trend['direction'] == 'bullish':
        insights.append(
            'BULLISH SIGNAL: Dark pool ATS% has decreased {:.1f}pp ({:.1f}% -> {:.1f}%) '
            'over recent weeks. Historically, ATS% decreases > 3pp have a 60% probability '
            'of positive 5-day returns (avg +7.67%).'.format(
                abs(ats_trend['magnitude_pp']),
                ats_trend.get('first_value', 0),
                ats_trend.get('last_value', 0),
            )
        )
    elif ats_trend['direction'] == 'bearish':
        insights.append(
            'BEARISH SIGNAL: Dark pool ATS% has increased {:.1f}pp. Historically, '
            'ATS% increases > 3pp have a 69.6% probability of negative 5-day returns.'.format(
                abs(ats_trend['magnitude_pp']),
            )
        )

    # Upcoming catalyst insight
    if zones:
        top = zones[0]
        insights.append(
            'UPCOMING: Strongest catalyst zone is {} to {} (score: {:.0f}/100) '
            'with {} converging catalysts ({}).'.format(
                top['window_start'], top['window_end'],
                top['combined_score'], top['num_catalysts'],
                ', '.join(top['catalyst_types']),
            )
        )

    # SMA proximity insight
    sma_200 = current['sma_200']
    close = current['close']
    pct_from_200 = ((close - sma_200) / sma_200) * 100 if sma_200 > 0 else 0
    if abs(pct_from_200) < 3:
        above_below = 'above' if close > sma_200 else 'below'
        insights.append(
            'WATCH: Price (${:.2f}) is only {:.1f}% {} the SMA-200 (${:.2f}). '
            'A decisive break {} this level could trigger momentum buying.'.format(
                close, abs(pct_from_200), above_below, sma_200,
                'above' if close < sma_200 else 'to hold above',
            )
        )

    # Options gamma insight
    opts = LIVE_DATA.get('options')
    if opts:
        if opts['gex_signal'] == 'amplifying':
            insights.append(
                'GAMMA SQUEEZE POTENTIAL: Net GEX is negative ({:,.0f}) — market makers '
                'must BUY into rallies to hedge, amplifying upward moves. '
                'Max pain is at ${:.2f} ({:+.1f}% from current price).'.format(
                    opts['net_gex'], opts['max_pain'],
                    opts['max_pain_distance_pct'],
                )
            )
        elif opts['overall_signal'] in ('BULLISH', 'SLIGHTLY_BULLISH'):
            insights.append(
                'OPTIONS FLOW: Bullish signal. PCR={:.2f}, Max Pain=${:.2f} '
                '({:+.1f}% from price). {} unusual activity events detected.'.format(
                    opts['pcr_volume'], opts['max_pain'],
                    opts['max_pain_distance_pct'],
                    opts.get('unusual_count', 0),
                )
            )

    # Threshold list insight
    thr = LIVE_DATA.get('threshold')
    if thr and thr.get('currently_on_list'):
        insights.append(
            'REG SHO ALERT: GME is CURRENTLY on the Reg SHO threshold list! '
            'This means FTDs exceeded 0.5% of outstanding shares for 5+ consecutive days. '
            'Forced buy-ins may be imminent.'
        )

    # What's missing for a squeeze
    missing = []
    si_pct = current['short_interest'] / FLOAT_EST * 100
    if si_pct < 30:
        missing.append('SI is {:.1f}% of float (Jan 2021 was ~140% -- current SI is much lower)'.format(si_pct))
    if current['institutional_shares'] > 200_000_000:
        missing.append('Institutional ownership is still {:,} shares -- many lendable shares remain'.format(
            current['institutional_shares'],
        ))

    if missing:
        insights.append(
            'LIMITING FACTORS: ' + '; '.join(missing) + '.'
        )

    print()
    for i, insight in enumerate(insights, 1):
        print('  {}. {}'.format(i, insight))
        print()


# ------------------------------------------------------------------
# Output: JSON
# ------------------------------------------------------------------


def export_results(current, zones, factors, overall_score, interpretation,
                   ats_trend, sma_data, hist_factor):
    """Write structured results to JSON."""
    level, explanation = interpretation

    # Identify which live sources were available
    live_sources = []
    if LIVE_DATA.get('ibkr_borrow'):
        live_sources.append('IBKR borrow rate')
    if LIVE_DATA.get('options'):
        live_sources.append('Options gamma/max pain')
    if LIVE_DATA.get('short_volume'):
        live_sources.append('FINRA daily short volume')
    if LIVE_DATA.get('threshold'):
        live_sources.append('Reg SHO threshold list')

    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'data_through': current['date'],
            'script': 'gme_squeeze_predictor.py',
            'shares_outstanding': SHARES_OUTSTANDING,
            'float_estimate': FLOAT_EST,
            'live_sources': live_sources,
            'total_factors': len(factors),
        },
        'current_conditions': current,
        'catalyst_timeline': [{
            'window_start': z['window_start'],
            'window_end': z['window_end'],
            'num_catalysts': z['num_catalysts'],
            'convergence_multiplier': z['convergence_multiplier'],
            'base_score': z['base_score'],
            'combined_score': z['combined_score'],
            'catalyst_types': z['catalyst_types'],
            'catalysts': z['catalysts'],
        } for z in zones],
        'directional_filters': {
            'ats_trend': ats_trend,
            'sma_position': {
                'close': sma_data['close'],
                'sma_50': sma_data['sma_50'],
                'sma_200': sma_data['sma_200'],
                'above_sma_50': sma_data['above_sma_50'],
                'above_sma_200': sma_data['above_sma_200'],
                'notes': sma_data['notes'],
            },
        },
        'squeeze_scorecard': {
            'overall_score': overall_score,
            'level': level,
            'interpretation': explanation,
            'factors': {
                f['name']: {
                    'value': f['value'],
                    'unit': f['unit'],
                    'score': f['score'],
                    'detail': f['detail'],
                }
                for f in factors
            },
        },
        'historical_comparison': hist_factor.get('comparisons', {}),
        'live_data': {k: v for k, v in LIVE_DATA.items()},
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print('\n  Results saved to: {}'.format(OUTPUT_PATH))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    """Orchestrate the full analysis."""
    # Load data
    rows, date_idx = load_csv_data()
    cache = load_cache_data()
    current = get_latest_conditions(rows)

    print_header()
    print('\n  Data: {} rows ({} to {})'.format(len(rows), rows[0]['date'], rows[-1]['date']))

    # Fetch live data from all available sources
    print('\n  FETCHING LIVE DATA:')
    print('  ' + '-' * 60)
    fetch_live_data('GME')

    # Override borrow rate with real IBKR data if available
    current = get_real_borrow_rate(current)

    print_current_conditions(current)

    # ---- Part 1: Catalyst Timeline ----

    # Find all catalysts
    t35_catalysts = find_pending_t35(rows, TODAY)
    opex_catalysts = compute_opex_catalysts(TODAY, CATALYST_HORIZON_MONTHS)
    all_catalysts = t35_catalysts + opex_catalysts

    # Directional filters
    ats_trend = compute_ats_trend(rows)
    sma_data = compute_sma_catalyst(rows)
    seasonal_score, seasonal_desc = compute_seasonal_score(TODAY, cache)

    # Find convergence zones
    zones = find_convergence_zones(all_catalysts)

    # Add directional lean to zones
    for zone in zones:
        zone['direction'] = ats_trend['direction']

    print_catalyst_timeline(zones, ats_trend, sma_data, seasonal_score, seasonal_desc)

    # ---- Part 2: Squeeze Probability ----

    factors = [
        score_si_pct_float(current),
        score_days_to_cover(current),
        score_cost_to_borrow(current),
        score_ftd_accumulation(rows, current),
        score_institutional_ownership(current),
        score_dark_pool_trend(ats_trend),
        score_sma_position(sma_data),
        score_historical_similarity(current, rows, date_idx),
    ]

    # Add live data factors (only if data was successfully fetched)
    for live_factor_fn in [score_options_gamma, score_short_volume, score_threshold_list]:
        result = live_factor_fn()
        if result is not None:
            factors.append(result)

    overall_score = compute_squeeze_score(factors)
    interpretation = interpret_score(overall_score)

    print_squeeze_scorecard(factors, overall_score, interpretation)

    # Historical comparison (from factor 8 — "Historical Similarity")
    hist_factor = next(
        (f for f in factors if f['name'] == 'Historical Similarity'),
        factors[-1],
    )
    print_historical_comparison(hist_factor)

    # Key insights
    print_key_insights(current, factors, zones, ats_trend)

    # Export
    export_results(current, zones, factors, overall_score, interpretation,
                   ats_trend, sma_data, hist_factor)


if __name__ == '__main__':
    main()
