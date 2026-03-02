# GME Holistic Data Analysis Report

**Generated:** February 18, 2026
**Data Range:** Feb 2002 - Feb 2026 (6,041 trading days)
**Sources:** SEC EDGAR (FTDs, 13F), FINRA (Short Interest, Dark Pool/ATS), Yahoo Finance (Price)

---

## Executive Summary

This report consolidates five distinct regulatory and market data streams for GameStop Corp (GME) onto a unified daily timeline. The analysis reveals persistent structural anomalies in GME's market microstructure that distinguish it from typical equities:

1. **Chronic settlement failures** (FTDs) far exceeding normal levels
2. **Sustained elevated short interest** with days-to-cover frequently above 10
3. **Significant dark pool routing** averaging ~20% ATS
4. **Declining then rebuilding institutional ownership** post-2021
5. **Cyclical price patterns** correlated with FTD settlement deadlines

---

## 1. Price Action Overview

| Metric | Value |
|--------|-------|
| Current Price | $23.26 (Feb 17, 2026) |
| All-Time High | $86.88 (Jan 27, 2021) / $120.75 intraday (Jan 28, 2021) |
| All-Time Low | $0.70 (Apr 3, 2020) |
| 2020 Average | $1.79 |
| 4:1 Split | July 22, 2022 |

### Key Price Periods

- **Pre-Sneeze (2020):** Stock languished under $2 for most of 2020, dismissed as a dying brick-and-mortar retailer
- **The Sneeze (Jan 2021):** $10.65 -> $120.75 intraday peak, 6.25 billion shares traded in 2 weeks
- **Feb 2021 Recovery:** $11.18 -> $87.12, a 492% rally in 13 trading days
- **DFV Return (May 2024):** $17.93 -> $64.83 peak when Roaring Kitty resurfaced on social media
- **Current:** Stabilized in $20-30 range with 65.75M shares short

---

## 2. Failure-to-Deliver (FTD) Analysis

**Source:** SEC Consolidated National Securities (CNS) data, half-monthly zip archives
**Coverage:** 2017-present (1,934 days with FTDs recorded)

### Key Statistics

| Metric | Value |
|--------|-------|
| Total FTD Days | 1,934 / 6,041 (32% of all trading days) |
| Total FTDs | 747,929,272 shares |
| Average FTD (when >0) | 386,727 shares |
| Peak FTD Day | 12,840,592 shares (Oct 13, 2020) |
| Peak FTD as % of Volume | 59.12% (Apr 30, 2020) |

### Top 10 FTD Days

| Date | FTDs | Close | FTD % of Vol |
|------|------|-------|-------------|
| Oct 13, 2020 | 12,840,592 | $2.97 | 31.5% |
| Sep 22, 2020 | 10,623,056 | $2.64 | 7.6% |
| Apr 17, 2020 | 8,422,860 | $1.22 | 37.3% |
| Jan 26, 2021 | 8,398,288 | $36.99 | 1.2% |
| Jan 27, 2021 | 7,891,448 | $86.88 | 2.1% |
| Apr 21, 2020 | 7,731,400 | $1.20 | 46.7% |
| Dec 3, 2020 | 7,148,764 | $4.03 | 28.4% |
| Feb 28, 2020 | 6,756,256 | $0.90 | 36.0% |
| Oct 16, 2019 | 6,465,760 | $1.47 | 41.8% |
| Jun 6, 2018 | 6,131,200 | $3.61 | 29.8% |

### FTD Spike Analysis

Identified 202 FTD spike days (>3x the 20-day average AND >100K shares):
- **54% (109/202)** were followed by a >5% price increase within 10 days
- **55% (111/202)** were followed by a >5% price decrease within 10 days
- This near-random split suggests FTD spikes alone aren't a directional signal, but they reliably precede high volatility

### T+35 Settlement Cycle

Under Reg SHO Rule 204, market makers have 35 calendar days to close FTD positions. Analysis of the 30 largest FTD days (>200K shares):

- **T+35 Win Rate: 40%** (12 of 30 showed positive price at T+35)
- Notable examples where T+35 forcing preceded major moves:
  - Jan 21, 2021 FTDs (5.76M shares) -> T+35 = Feb 25, 2021 -> **+152.7%** (the Feb recovery rally)
  - Sep 4, 2020 FTDs (4.19M shares) -> T+35 = Oct 9, 2020 -> **+57.1%**
  - Jul 21, 2020 FTDs (4.19M shares) -> T+35 = Aug 25, 2020 -> **+24.2%**

**Interpretation:** While the aggregate T+35 hit rate is only 40%, the largest FTD accumulations in late 2020 and early 2021 created the forced-buying pressure that contributed to the historic squeeze. The settlement cycle creates periodic buying pressure, but it's most impactful when FTDs cluster over consecutive days.

---

## 3. Short Interest Analysis

**Source:** FINRA Consolidated Short Interest reports (bi-monthly)
**Coverage:** May 2022 - Jan 2026 (63 reports)

### Key Statistics

| Metric | Value |
|--------|-------|
| Latest SI | 65,750,329 shares (Jan 15, 2026) |
| Latest DTC | 12.58 days |
| Peak SI | 77,185,174 shares (Jun 30, 2025) |
| Peak DTC | 24.62 days (Feb 29, 2024) |
| Average DTC | 10.02 days |

### Short Interest Timeline (Key Transitions)

| Period | SI (shares) | DTC | Context |
|--------|-------------|-----|---------|
| May 2022 | 15.5M | 3.12 | Pre-split baseline |
| Aug 2022 | 53.2M | 7.96 | Post-split: SI jumped 3.5x immediately |
| Feb 2024 | 60.5M | 24.62 | Peak DTC - extremely hard to borrow |
| May 2024 | 68.4M | 1.03 | DFV return: DTC collapsed as volume exploded |
| Jun 2024 | 46.5M | 1.00 | Post-DFV: SI dropped 32% (forced covering) |
| Jun 2025 | 77.2M | 4.96 | Peak SI: new all-time high in short shares |
| Jan 2026 | 65.8M | 12.58 | Current: elevated SI with moderately high DTC |

### Interpretation

The short interest data tells a compelling story:

1. **Post-split SI explosion (Aug 2022):** When GME split 4:1, the reported SI jumped from ~15M to ~53M shares. While split-adjusted, this represented significant new short positioning.

2. **The DTC peak of 24.62 days (Feb 2024)** is extraordinary. For context, DTC >5 is considered elevated for most stocks. A DTC of 24.62 means it would take nearly 5 weeks of average volume to close all short positions. This creates extreme vulnerability to any demand shock.

3. **DFV Effect (May-Jun 2024):** When Roaring Kitty returned, the DTC collapsed from 17.36 to 1.03 as volume exploded 18x. Despite the volume surge, SI only dropped from 64.4M to 46.5M (-28%), suggesting many shorts held through the volatility rather than covering.

4. **Persistent re-shorting:** After the DFV event drove SI to 46.5M, shorts steadily rebuilt to a new peak of 77.2M by June 2025. This represents a 66% increase in short positioning in just 12 months.

5. **Current state (Jan 2026):** SI at 65.8M with DTC at 12.58 represents a meaningful short squeeze setup. With average daily volume of ~5.2M shares, it would take 12.5 trading days of 100% short-covering volume to close all positions.

---

## 4. Dark Pool / ATS Analysis

**Source:** FINRA OTC/ATS Weekly Summary
**Coverage:** Dec 2021 - Jan 2026 (217 weekly reports)

### Key Statistics

| Metric | Value |
|--------|-------|
| Total OTC+ATS Volume | 4,921,696,307 shares |
| Total ATS (Dark Pool) Volume | 883,355,470 shares |
| Average ATS % | 19.6% |
| Peak ATS % | 39.4% (Sep 18, 2023) |
| Peak Weekly Volume | 358,989,223 (Jun 3, 2024 - DFV week) |

### ATS% Trend

- **2022 Average:** ~18-22% ATS
- **2023 Average:** ~19-24% ATS (with spikes to 39.4%)
- **2024 Average:** ~15-20% ATS (dropped during DFV volatility)
- **2025-26 Average:** ~18-22% ATS

### Interpretation

Dark pool activity on GME reveals several patterns:

1. **Persistent off-exchange routing:** Nearly 1 in 5 shares traded through dark pools/ATS venues. This is within the normal range for active stocks but on the higher end, suggesting meaningful institutional order flow management.

2. **Inverse volume relationship:** During high-volume events (DFV return, sneeze echoes), ATS% drops as more volume flows to lit exchanges. During quiet periods, ATS% rises, suggesting dark pools absorb a larger share of routine/retail order flow.

3. **ATS% spikes (>25%)** have occurred during periods of price decline or consolidation, consistent with the thesis that increased dark pool routing can suppress price discovery during accumulation phases.

4. **Recent weeks (Jan 2026):** ATS% at 19.2% on the most recent week with 46.3M total volume - a significant volume week suggesting institutional activity.

---

## 5. Institutional Ownership (13F)

**Source:** SEC EDGAR 13F-HR bulk data sets + EFTS full-text search API (direct from SEC, no third-party)
**Coverage:** Q4 2019 - Q4 2025 (24 quarterly snapshots)
**Note:** All share counts are post-split adjusted (GME 4:1 split July 2022)

### Ownership Timeline

| Quarter | Inst. Shares (split-adj) | # Institutions | Context |
|---------|--------------------------|----------------|---------|
| Q1 2020 | 403,126,004 | 341 | Pre-transformation (100.8M × 4) |
| Q3 2020 | 367,241,180 | 331 | RC Ventures entering (91.8M × 4) |
| Q4 2020 | 442,665,764 | 364 | Peak institutional ownership (110.7M × 4) |
| Q1 2021 | 211,501,320 | 396 | Post-squeeze sell-off (52.9M × 4) |
| Q2 2021 | 197,665,164 | 451 | Continued exit, but MORE holders |
| Q3 2021 | 148,790,064 | 452 | Trough (37.2M × 4) |
| Q1 2022 | 209,717,424 | 448 | Stabilization (52.4M × 4) |
| Q2 2022 | 141,494,068 | 449 | Last pre-split quarter (35.4M × 4) |
| Q3 2022 | 120,355,288 | 426 | First post-split quarter (as-filed) |
| Q4 2023 | 105,591,721 | 417 | Gradual rebuild |
| Q2 2024 | 123,415,716 | 489 | ATM offerings grow float |
| Q3 2024 | 155,176,699 | 492 | Accelerating accumulation |
| Q2 2025 | 248,056,152 | 559 | Strong institutional inflows |
| Q3 2025 | 346,318,462 | 575 | Peak — Tudor 88M position |
| Q4 2025 | 252,217,128 | 564 | Some Q4 filings still pending |

### Top 10 Institutional Holders (Q3 2025)

| Rank | Institution | Shares | % of Float |
|------|------------|--------|-----------|
| 1 | Tudor Investment Corp | 88,000,000 | ~19.7% |
| 2 | Vanguard Group | 38,500,000 | ~8.6% |
| 3 | BlackRock Inc. | 35,370,000 | ~7.9% |
| 4 | State Street Corp | 12,370,000 | ~2.8% |
| 5 | Susquehanna Intl Group | 8,280,000 | ~1.9% |
| 6 | Jane Street Group | 8,170,000 | ~1.8% |
| 7 | Geode Capital Management | 7,250,000 | ~1.6% |
| 8 | Marshall Wace LLP | 5,260,000 | ~1.2% |
| 9 | Invesco Ltd | 4,070,000 | ~0.9% |
| 10 | Renaissance Technologies | 3,580,000 | ~0.8% |

### Interpretation

1. **Pre-sneeze (2020):** Institutional ownership peaked at ~443M split-adjusted shares (Q4 2020) across 364 institutions. With GME's pre-split float at ~70M shares (280M post-split equivalent), this meant institutional ownership exceeded 150% of free float — a key catalyst for the January 2021 short squeeze.

2. **Post-sneeze reshuffling (2021):** Institutions sold aggressively during the squeeze: split-adjusted shares dropped from 443M to 212M in one quarter (Q4 2020 → Q1 2021). Remarkably, the number of institutional holders INCREASED from 364 to 396, meaning many new institutions entered smaller positions while large holders exited.

3. **Post-split trough (2022-2023):** Holdings fell to 120-141M shares with 415-449 institutions. The transition through the 4:1 split (July 2022) saw Q2 2022 at 141.5M split-adjusted, dropping to 120.4M in Q3 2022 as institutions repositioned around the corporate action.

4. **Accumulation phase (2024-2025):** Steady institutional buying from 108M (Q1 2024) to 346M (Q3 2025) — more than tripling in 18 months. Institution count grew from 456 to 575 (record high). Q3 2025 peak driven by Tudor Investment Corp's massive 88M-share position.

5. **Q4 2025 pullback:** Dropped to 252M shares with 564 institutions. Some decline may reflect late filings (deadline Feb 14, 2026) rather than actual selling.

5. **Ryan Cohen (insider, not 13F):** Holds ~75.2M shares (16.78%) through RC Ventures — not counted in institutional 13F totals.

---

## 6. Estimated Borrow Rates

**Source:** Derived from short interest DTC using piecewise linear heuristic
**Note:** These are estimates, not actual broker borrow rates

| Current Est. Rate | 40.92% (based on DTC 12.58) |
| Peak Est. Rate | 78.86% (Feb 29, 2024 - DTC 24.62) |
| Current Utilization | 21.56% (SI/shares outstanding) |

The borrow rate proxy suggests GME has been in "hard to borrow" territory for most of the 2023-2026 period, with brief exceptions during high-volume events when DTC temporarily compressed.

---

## 7. Cross-Data Correlations & Key Events

### Event 1: The Sneeze (Jan 22 - Feb 5, 2021)
- **Price:** $10.65 -> $120.75 peak (+1,034% intraday)
- **FTDs:** 24.5M shares failed in 11 trading days
- **Volume:** 6.25 billion shares traded (vs ~8M/day normal)
- **Note:** SI/DP/Ownership data not available for this exact period in our dataset

### Event 2: Post-Split Repositioning (Jul-Aug 2022)
- **Price:** Volatile, $36.88 -> $34.50 with $48 spike
- **SI jumped:** 14.9M -> 53.2M (257% increase)
- **FTDs:** 4.7M over the period
- **DP ATS%:** Rose from 16% to 22.4%
- **Interpretation:** The 4:1 split triggered massive new short positioning. The simultaneous rise in dark pool activity suggests market makers were routing significant flow off-exchange during the repositioning.

### Event 3: DTC Peak (Feb 2024)
- **DTC:** 24.62 days - the highest recorded
- **SI:** 60.5M shares
- **Price:** ~$12-13 range (low volatility)
- **DP ATS%:** ~18-22%
- **Interpretation:** Classic squeeze setup - extremely compressed volatility with extreme DTC. This was 3 months before the DFV return triggered the May 2024 eruption.

### Event 4: DFV Return (May-Jun 2024)
- **Price:** $17.93 -> $64.83 peak (+261% peak)
- **SI collapsed:** 68.4M -> 46.5M (-32%) as shorts forced to cover
- **DTC collapsed:** 17.36 -> 1.00 (volume explosion, not covering)
- **FTDs:** 2.66M relatively modest vs. 2021 (market structure improvements)
- **DP ATS%:** Dropped from 13.8% to 10.8% (more flow on lit exchanges)
- **Interpretation:** The DFV catalyst met a market loaded with short exposure. DTC's collapse was primarily volume-driven rather than covering-driven, meaning the squeeze pressure was partially dissipated by the GME ATM offering (75M new shares).

### Event 5: Short Rebuild to Peak (Jul 2024 - Jun 2025)
- **SI rebuilt:** 40.3M -> 77.2M (+91.5% over 12 months)
- **DTC:** Rose from 2.55 -> 4.96
- **Price:** Declined from ~$25 -> ~$22
- **Interpretation:** After the DFV squeeze subsided, shorts aggressively rebuilt positions, reaching an all-time high of 77.2M shares by June 2025. This steady accumulation during price decline suggests strong conviction among shorts that the stock is overvalued.

---

## 8. Current State Assessment (February 2026)

| Metric | Value | Assessment |
|--------|-------|------------|
| Price | $23.26 | Mid-range |
| SI | 65.75M shares | Elevated |
| DTC | 12.58 days | High - takes 2.5 weeks to cover |
| Est. Borrow Rate | 40.9% | Hard to borrow |
| ATS% | 19.2% | Normal-high |
| Inst. Ownership | ~252M (564 inst) | Q4 2025 — some filings pending |
| 20d Volatility | Low | Compression |
| SMA 50 vs 200 | Converging | Watch for cross |

### Risk Factors for Shorts:
- DTC at 12.58 means any demand shock takes weeks to resolve
- Estimated borrow rate of 40.9% makes carrying the position expensive
- Institutional ownership at 252M shares (~56% of 447M outstanding) is significant but down from Q3 2025 peak
- Any catalyst (earnings, RC activity, regulatory changes) could trigger rapid covering

### Risk Factors for Longs:
- GME's core business (gaming retail) faces structural headwinds
- No clear path to profitability in core operations
- Bitcoin treasury strategy adds crypto volatility risk
- ATM offering dilution risk (GME has used this before)

---

## 9. TradingView Indicator Guide

Two Pine Script indicators are provided:

### 1. GME Holistic Analysis (Overlay)
**File:** `tradingview/gme_holistic_indicator.pine`

- **Overlay on price chart** with SMA 50/200
- **Regime detection:** Green (bullish), Red (bearish), Yellow (squeeze setup)
- **Data dashboard** showing all current values (SI, DTC, ATS%, ownership, borrow rate, RVOL, volatility)
- **Event markers** for key historical dates
- **SI report markers** when new data arrives
- **RVOL spike markers** when volume exceeds 2x average
- **Golden/Death cross markers** for SMA 50/200
- **Alerts** for extreme DTC, high RVOL, squeeze setups, SMA crosses

### 2. GME Alt Data Panel (Lower Indicator)
**File:** `tradingview/gme_alt_data_panel.pine`

- **Five display modes:**
  - Short Interest (millions of shares)
  - Days to Cover (with threshold zones)
  - Dark Pool ATS %
  - Estimated Borrow Rate
  - All (Normalized 0-100 scale overlay)
- **Color-coded thresholds** (green/orange/red zones)
- **Report day highlighting** (orange for SI, purple for DP)
- **Configurable threshold levels**

### Installation:
1. Open TradingView, navigate to GME daily chart
2. Pine Script Editor -> New -> Paste indicator code
3. Add to chart (repeat for both indicators)
4. Configure input settings as desired

---

*Report generated from SEC EDGAR, FINRA, and Yahoo Finance data. Alternative data updated through January 2026. This report is for informational purposes only and does not constitute financial advice.*
