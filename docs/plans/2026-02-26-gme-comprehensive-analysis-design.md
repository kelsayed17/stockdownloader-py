# GME Comprehensive Options Analysis Platform

**Date:** 2026-02-26
**Approach:** Layered architecture (Data → Analytics → Strategy → Fusion)

## Goal

Build a full GME research platform that ingests 4 years of options chain data from Polygon, computes daily options analytics, backtests a comprehensive strategy suite, and fuses multi-signal anomaly detection with existing alt data (FTD, SI, dark pool, OCC OI) and the 80-feature ML pipeline.

## Decisions

- **Data scope:** Full chain (all strikes, all expirations)
- **Granularity:** Daily bars
- **Strategies:** All premium-selling + directional (wheel, iron condor, strangle, credit spread, long options, calendar spread)
- **Signal priority:** Full multi-signal fusion across 4 pillars
- **API strategy:** Batch overnight with resumable checkpoint

## Architecture

```
Polygon API ──► Fetcher ──► Parquet Storage
                                │
                                ▼
                       OptionsStateEngine (40 metrics)
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              Backtester   SignalFusion   Scorecard
              (6 strategies) (4 pillars)  (daily output)
```

---

## Section 1: Data Layer

### Resumable Polygon Options Pipeline

**Contract discovery:** Query `/v3/reference/options/contracts?underlying_ticker=GME` paginated with `expired=true/false`. Store contract metadata as JSON, grouped by expiration month.

**Daily bars:** For each contract, fetch `/v2/aggs/ticker/{optionsTicker}/range/1/day/{from}/{to}`. Rate limit at 0.2s between requests. Expected volume: 50-100K contracts over 4 years.

**Storage format:** Parquet files partitioned by expiration month:

```
data/GME/options/
├── contracts/                    # Raw contract metadata
├── 2022-01/daily_bars.parquet   # Columns: ticker, date, open, high, low, close, volume, vwap, oi
├── 2022-02/daily_bars.parquet
├── ...
├── 2026-02/daily_bars.parquet
└── checkpoint.json               # {"completed_months": ["2022-01", ...], "last_contract": "..."}
```

**Resumability:** JSON checkpoint file tracks completed expiration months. On restart, skip completed months. Within a month, track last processed contract for mid-month resume.

**Rate limiting:** 0.2s sleep between API calls. Polygon free tier allows 5 req/s; overnight batch runs ~18K requests/hour.

### Schema: daily_bars.parquet

| Column | Type | Description |
|--------|------|-------------|
| option_ticker | string | O:GME220121C00020000 format |
| underlying | string | GME |
| date | date | Trading day |
| expiration | date | Contract expiration |
| strike | float | Strike price |
| option_type | string | call / put |
| open | float | Daily open |
| high | float | Daily high |
| low | float | Daily low |
| close | float | Daily close (last trade or mid) |
| volume | int | Daily volume |
| open_interest | int | End-of-day OI |
| vwap | float | Volume-weighted average price |

---

## Section 2: Analytics Layer

### OptionsStateEngine

Reads raw Parquet bars and computes ~40 daily metrics across 5 categories.

```python
class OptionsStateEngine:
    def build(self, start: date, end: date) -> pd.DataFrame:
        """Process all daily bars → 40 daily metrics → options_state.parquet."""
```

**Output:** `data/GME/options_state.parquet` — one row per trading day, ~1000 rows for 4 years.

### Metric Categories

#### IV Surface (8 metrics)

Construct implied volatility surface using Black-Scholes inversion on the closest-to-ATM options.

| Metric | Description |
|--------|-------------|
| atm_iv_30d | ATM IV at 30-day tenor |
| atm_iv_60d | ATM IV at 60-day tenor |
| atm_iv_90d | ATM IV at 90-day tenor |
| iv_skew_25d_30 | 25-delta put IV minus 25-delta call IV (30-day) |
| iv_term_slope | Slope of ATM IV across tenors (contango vs backwardation) |
| iv_percentile | Current ATM IV rank over trailing 252 days |
| iv_change_1d | Daily change in ATM 30d IV |
| iv_change_5d | 5-day change in ATM 30d IV |

#### Gamma Exposure — GEX (5 metrics)

Estimate dealer gamma exposure from OI and delta.

| Metric | Description |
|--------|-------------|
| net_gex | Net dealer gamma (positive = long gamma, dampening) |
| gex_flip_price | Price where net GEX crosses zero |
| call_wall | Strike with highest call OI × gamma |
| put_wall | Strike with highest put OI × gamma |
| gex_concentration | Top-3 strikes as fraction of total GEX |

#### OI Flow (8 metrics)

Track open interest changes to detect positioning shifts.

| Metric | Description |
|--------|-------------|
| total_call_oi | Total call open interest |
| total_put_oi | Total put open interest |
| pc_oi_ratio | Put/call OI ratio |
| pc_oi_ratio_change | 1-day change in put/call ratio |
| oi_weighted_strike | OI-weighted average strike (center of mass) |
| oi_concentration_top5 | Top-5 strikes as % of total OI |
| near_term_oi_pct | OI expiring within 30 days as % of total |
| oi_skew_delta | Change in OI distribution skew (bullish/bearish shift) |

#### Volume & Premium Flow (6 metrics)

| Metric | Description |
|--------|-------------|
| call_volume | Total call volume |
| put_volume | Total put volume |
| pc_volume_ratio | Put/call volume ratio |
| call_premium | Total call premium (volume × price × 100) |
| put_premium | Total put premium |
| premium_imbalance | (call_premium - put_premium) / total_premium |

#### FTD/SI Cycle Metrics (6 metrics)

Sourced from existing alt data store, aligned to options state dates.

| Metric | Description |
|--------|-------------|
| ftd_t35_countdown | Days until T+35 from latest large FTD spike |
| si_pct_float | Short interest as % of float |
| si_change_2wk | 2-week change in SI |
| dark_pool_ratio | Dark pool volume / total volume |
| dark_pool_ratio_change | 5-day change in dark pool ratio |
| regsho_proximity | Distance from RegSHO threshold (% of float) |

### IV Calculation

Use Black-Scholes inversion via Newton-Raphson. For each trading day:
1. Find ATM calls/puts at 30/60/90-day tenors (closest expiration to target)
2. Invert mid-price to IV using risk-free rate from treasury yields
3. For 25-delta skew: find strikes where BS delta ≈ 0.25, interpolate IV
4. Fallback: if no exact match, interpolate between nearest strikes

---

## Section 3: Strategy Layer

### GMEOptionsBacktester

Walk-forward backtesting engine with pluggable strategy interface.

```python
class GMEOptionsStrategy(ABC):
    @abstractmethod
    def evaluate(self, date: date, state: OptionsState, chain: pd.DataFrame) -> list[Trade]:
        """Return trades to open today."""

    @abstractmethod
    def on_expiry(self, date: date, positions: list[Position]) -> list[Trade]:
        """Handle expiring positions (roll, close, let expire)."""

class GMEOptionsBacktester:
    def __init__(self, strategies: list[GMEOptionsStrategy], config: BacktestConfig):
        self.strategies = strategies
        self.config = config

    def run(self, start: date, end: date) -> BacktestResult:
        """Walk-forward: 2y train / 1y test, roll quarterly."""
```

### 6 Strategy Implementations

| Strategy | Logic | GME-Specific Twist |
|----------|-------|--------------------|
| **Wheel** | Sell CSPs → assignment → sell CCs → called away → repeat | Pause when T+35 < 5 days or IV > 80th pct |
| **Iron Condor** | Sell OTM call + put spread | Widen wings when GEX concentration > 60% |
| **Strangle** | Sell OTM call + put (naked or defined) | Only enter when IV percentile > 50th |
| **Credit Spread** | Bull put or bear call spreads | Direction from composite score sign |
| **Long Options** | Buy calls/puts on directional signals | Enter on cycle hot regime + composite > +1.5 |
| **Calendar Spread** | Sell near-term, buy far-term same strike | Enter when IV term structure in backwardation |

### Walk-Forward Validation

- **Training window:** 2 years (optimize strategy parameters)
- **Test window:** 1 year (out-of-sample evaluation)
- **Roll:** Quarterly (re-optimize every 3 months)
- **Metrics:** Sharpe, max drawdown, win rate, profit factor, avg trade P&L

### Output

Per-strategy results saved to `data/GME/backtest_results/{strategy}_results.parquet` with columns: date, strategy, direction, entry_price, exit_price, pnl, holding_days, regime_at_entry.

---

## Section 4: Signal Fusion & Multi-Signal Prediction Model

### GMESignalFusion

Fuses ~40 daily options metrics with existing alt data and ML features into a unified scoring system.

```python
class GMESignalFusion:
    def compute_daily_scorecard(self, date: date) -> GMEDailyScorecard:
        """Produce a single daily scorecard with composite score + regime."""
        options = self.options_state.get(date)
        alt = self.alt_data_store.snapshot(date)
        ml = self.ml_features.get(date)

        pillars = {
            "options_flow": self._score_options(options),
            "volume_premium": self._score_volume(options, alt),
            "cycle_timing": self._score_cycles(alt),
            "momentum": self._score_momentum(ml),
        }

        composite = sum(w * pillars[k] for k, w in self.weights.items())
        regime = self._detect_regime(options, alt, composite)

        return GMEDailyScorecard(
            date=date, pillar_scores=pillars,
            composite=composite, regime=regime,
            anomaly_flags=self._flag_anomalies(pillars),
        )
```

### Signal Groups (4 Pillars)

| Pillar | Key Signals | Weight |
|--------|-------------|--------|
| **Options Flow** | IV percentile, skew delta, GEX flip proximity, call wall distance, put/call OI ratio change | 30% |
| **Volume/Premium** | Unusual volume ratio, premium flow imbalance, dark pool premium vs lit | 20% |
| **Cycle Timing** | T+35 countdown proximity, SI change acceleration, RegSHO threshold distance | 30% |
| **Momentum** | Existing ML score, price vs VWAP, RSI regime, OBV trend | 20% |

### Composite Score

Each pillar produces a Z-score normalized signal (-3 to +3). Combined using learned weights (initially equal, optimized via walk-forward).

### Regime Detection (4 States)

| Regime | Trigger Conditions |
|--------|-------------------|
| **Squeeze** | Bollinger inside Keltner + IV < 30th percentile + GEX compressing |
| **Gamma Ramp** | GEX flip approaching price + call wall < 5% away + OI concentrating |
| **Cycle Hot** | T+35 within 5 days + SI change > 2 sigma + dark pool ratio diverging |
| **Neutral** | None of the above |

### Anomaly Flags

Any pillar exceeding +/-2 sigma triggers an anomaly flag with human-readable explanation.

### Output

`data/GME/daily_scorecards.parquet` — one row per trading day with pillar scores, composite, regime, and anomaly flags.

---

## Section 5: CLI & Integration

### Commands

```bash
# Data ingestion (resumable, overnight)
python -m stockdownloader.gme.options fetch --start 2022-01-01 --end 2026-02-26

# Build analytics state
python -m stockdownloader.gme.options build-state

# Run backtests
python -m stockdownloader.gme.options backtest --strategy wheel
python -m stockdownloader.gme.options backtest --all

# Generate daily scorecard
python -m stockdownloader.gme.options scorecard --date 2026-02-26
python -m stockdownloader.gme.options scorecard --range 2025-01-01 2026-02-26

# Full pipeline (fetch -> state -> scorecard -> backtest)
python -m stockdownloader.gme.options run-all
```

### Module Layout

```
src/stockdownloader/gme/
├── options/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry point
│   ├── fetcher.py               # Resumable Polygon pipeline
│   ├── state_engine.py          # OptionsStateEngine
│   ├── backtester.py            # GMEOptionsBacktester
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── wheel.py
│   │   ├── iron_condor.py
│   │   ├── strangle.py
│   │   ├── credit_spread.py
│   │   ├── long_options.py
│   │   └── calendar_spread.py
│   ├── signal_fusion.py         # GMESignalFusion
│   └── scorecard.py             # GMEDailyScorecard dataclass + output
├── pipeline.py                  # Existing (untouched)
├── prediction.py                # Existing (feeds into fusion)
└── ...existing files...
```

### Data Layout

```
data/GME/
├── options/
│   ├── contracts/               # Raw contract metadata
│   ├── 2022-01/daily_bars.parquet
│   ├── ...
│   ├── 2026-02/daily_bars.parquet
│   └── checkpoint.json
├── options_state.parquet        # 40 daily metrics
├── daily_scorecards.parquet     # Composite scores + regimes
├── backtest_results/
│   ├── wheel_results.parquet
│   └── ...
└── ...existing cached data...
```

### Integration Points

| Existing Module | Used By | How |
|----------------|---------|-----|
| `polygon_options_client.py` | `fetcher.py` | Contract discovery + bar fetching |
| `alt_data_store.py` | `signal_fusion.py` | FTD/SI/dark pool snapshots |
| `gme/prediction.py` | `signal_fusion.py` | ML scores for momentum pillar |
| `backtesting/engines/wheel.py` | `strategies/wheel.py` | Port logic for GME |
| FINRA/OCC clients | `fetcher.py` | Refresh SI + OI during fetch |

### Configuration

Single `GMEOptionsConfig` dataclass loaded from environment or CLI flags. Holds Polygon API key, date range, rate limits, strategy parameters.
