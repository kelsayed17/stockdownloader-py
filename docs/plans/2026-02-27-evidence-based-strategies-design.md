# Evidence-Based SPY Intraday Strategies Design

## Problem

A mega combinatorial backtest of 543 strategy configurations found 0/543 beat SPY buy-and-hold (+37.14% over 2 years). The existing strategy suite lacks three critical elements identified by academic research:

1. **No VIX regime filtering** — every high-performing published strategy uses VIX conditioning
2. **No time-of-day targeting** — exploitable alpha concentrates in the first and last 30 minutes
3. **No academically-validated entry logic** — existing strategies use conventional technical indicators (MACD, SMA crossovers) that have no peer-reviewed support for SPY intraday

## Solution

Implement 5 new strategies based on the highest-evidence academic and practitioner research, plus a MarketContext infrastructure layer enabling VIX/FOMC filtering across all strategies.

## Architecture

### MarketContext Provider

A separate context provider injected into the backtest engine at construction time. At each session boundary, the engine calls `provider.get_context(trading_date)` and stores the result. Strategies access it via `ctx.market_ctx` in BarContext.

```python
@dataclass(slots=True)
class MarketContext:
    """Daily market regime data, computed externally and injected per-session."""
    vix_close: Decimal          # Prior day VIX close
    vix_sma20: Decimal          # 20-day SMA of VIX
    vix_regime: str             # "low" (<15), "mid" (15-25), "high" (>25), "extreme" (>35)
    is_fomc_day: bool           # FOMC announcement day
    is_fomc_press_conf: bool    # FOMC press conference day (subset of FOMC days)
    is_opex: bool               # Monthly/quarterly options expiration
    daily_rsi2: Decimal         # 2-period RSI on daily closes
    daily_close_above_sma200: bool  # Trend filter
```

**Protocol:**

```python
class MarketContextProvider(Protocol):
    def get_context(self, trading_date: str) -> MarketContext | None: ...
```

**Concrete implementation:** `FileMarketContextProvider` loads a pre-computed CSV with columns `date,vix_close,vix_sma20,is_fomc,is_fomc_pc,is_opex,rsi2,above_sma200`.

**Integration points:**
- `BarContext` gains an optional `market_ctx: MarketContext | None` field (default None)
- `IntradayInfra` accepts an optional provider, calls it at session start in `DayTracker.on_new_day()`
- `IntradayBacktestEngine` accepts an optional provider, passes it through to the strategy
- When no provider is configured, everything works exactly as today

**Data pipeline:** `scripts/prepare_market_context.py` fetches VIX daily bars from Polygon, computes RSI(2)/SMA(200) on SPY daily, embeds FOMC dates (hardcoded list from Fed website), and writes the CSV.

### Strategy 1: Gao Intraday Momentum

**Source:** Gao, Han, Li & Zhou (2018), "Market Intraday Momentum," Journal of Financial Economics 129(2): 394-414. Sharpe 1.08, 20+ years of out-of-sample data.

**Mechanism:** The first half-hour return (9:30-10:00 AM) positively predicts the last half-hour return (3:30-4:00 PM). Driven by institutional close-of-day rebalancing and late-informed trader execution.

**Entry logic:** Compute first half-hour return `r1 = (close_bar6 - open_bar1) / open_bar1`. In the last 30 minutes (bars 73-78), enter long if `r1 > 0`, short if `r1 < 0`. Enhanced mode: also check penultimate half-hour return `r12` (bar 67-72) -- only enter when both agree (win rate jumps from 54% to 77%).

**Config:**

```python
@dataclass(frozen=True, slots=True)
class GaoMomentumConfig(InfraExitConfig):
    entry_start_bar: int = 73           # First eligible entry bar (last 30 min)
    first_hh_bars: int = 6              # Bars defining first half-hour
    require_dual_signal: bool = True    # Require both r1 and r12 agreement
    min_r1_magnitude: Decimal = Decimal("0.0005")  # Filter noise moves
    sl_atr_mult: Decimal = Decimal("1.5")
    allow_shorts: bool = True           # Paper trades both directions
    close_eod: bool = True              # Always close at EOD
    max_day: int = 1
```

**Exit:** Close at EOD (bar 78). SL = 1.5x ATR. No take-profit -- ride to close.

**VIX enhancement** (when MarketContext available): Only trade when `vix_regime != "low"` -- the paper shows R-squared jumps from 0.6% to 3.3% in high-vol regimes.

**Pattern:** BaseIntradayStrategy. Evaluates every bar but only fires signals in bars 73-78.

### Strategy 2: Noise Boundary Breakout

**Source:** Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)," Swiss Finance Institute Research Paper No. 24-97. Sharpe 1.33, 17 years.

**Mechanism:** Defines a "noise area" around the open based on historical intraday volatility at each time checkpoint. Price within the band is noise; breakouts beyond it signal genuine momentum.

**Noise band computation:** At session start, compute per-checkpoint volatility:
- `sigma(HH:MM) = avg(|close(t-j, HH:MM) / open(t-j) - 1|)` for j = 1..14
- Upper = `max(open_today, prev_close) * (1 + multiplier * sigma)`
- Lower = `min(open_today, prev_close) * (1 - multiplier * sigma)`
- Bands are gap-adjusted

**Entry:** At half-hourly checkpoints (bars 6, 12, 18, ..., 72), if price > upper boundary -> long; if price < lower boundary -> short. Fire-once per session.

**Exit:** Price reverses into noise area, OR crosses VWAP against position, OR EOD.

**Config:**

```python
@dataclass(frozen=True, slots=True)
class NoiseBoundaryConfig(InfraExitConfig):
    lookback_days: int = 14
    vol_multiplier: Decimal = Decimal("1.0")
    checkpoint_interval: int = 6        # Every 30 min (6 bars * 5 min)
    sl_atr_mult: Decimal = Decimal("2.0")
    allow_shorts: bool = True
    close_eod: bool = True
    max_day: int = 1
```

**Implementation note:** Requires storing per-checkpoint volatility computed from prior days' intraday data. The strategy maintains a rolling buffer of daily open prices + checkpoint closes computed from the raw `data[]` array passed to `evaluate()`.

### Strategy 3: RSI(2) Connors Mean Reversion

**Source:** Larry Connors, backtested on SPY 1993-present. 75% win rate, profit factor 2.3.

**Mechanism:** SPY is mean-reverting at short timeframes. RSI(2) < 5 identifies extreme oversold conditions that reliably bounce.

**Entry logic:** At session open (bar 1), check: if `daily_rsi2 < 5` AND `daily_close > SMA(200)` -> enter long. When MarketContext is available, uses its pre-computed values. Without MarketContext, computes RSI(2) internally from the DayTracker's aggregated daily bars.

**Exit:** When current close exceeds the 5-day SMA of daily closes (approximated from DayTracker aggregated bars). Also close at EOD since this is an intraday implementation.

**Config:**

```python
@dataclass(frozen=True, slots=True)
class ConnorsRSI2Config(InfraExitConfig):
    rsi_period: int = 2
    rsi_threshold: Decimal = Decimal("5")
    sma_trend_period: int = 200
    exit_sma_period: int = 5
    sl_atr_mult: Decimal = Decimal("2.0")  # Wide stop for mean reversion
    allow_shorts: bool = False              # Long-only per Connors
    close_eod: bool = True
    max_day: int = 1
```

### Strategy 4: VIX Regime Filter (Wrapper)

**Source:** Cross-cutting finding from all academic studies. VIX filtering improves risk-adjusted returns by 8-15 percentage points across all strategy categories.

**Design:** Not a standalone strategy -- a wrapper/decorator that adds VIX filtering to any existing `IntradayTradingStrategy`.

```python
class VixFilteredStrategy(IntradayTradingStrategy):
    """Wraps any intraday strategy with VIX regime filtering."""

    def __init__(self, inner: IntradayTradingStrategy,
                 allowed_regimes: list[str] = None) -> None:
        self._inner = inner
        self._allowed = allowed_regimes or ["mid", "high"]

    def evaluate(self, data, current_index) -> IntradaySignal:
        # Check MarketContext -- if VIX regime not in allowed list, HOLD
        # Otherwise delegate to inner strategy
```

**Usage:** `VixFilteredStrategy(inner=ORReversalStrategy(), allowed_regimes=["mid", "high"])`. This lets us test VIX filtering on ALL existing strategies without modifying them.

### Strategy 5: FOMC Drift

**Source:** Lucca & Moench (2015), "The Pre-FOMC Announcement Drift," Journal of Finance. Sharpe 1.14. Updated by Ignatieva & Ohashi (2024) -- now concentrated on press conference days only, Sharpe ~1.8.

**Mechanism:** Systematic upward drift before FOMC announcements, driven by portfolio rebalancing and risk appetite changes. Effect is strongest during elevated VIX.

**Entry logic:** On FOMC press conference days (checked via `market_ctx.is_fomc_press_conf`), enter long at market open (bar 1). Requires MarketContext -- strategy is inert without it.

**Exit:** Close at EOD. SL = 2x daily ATR (wide stop for event-driven trade).

**Config:**

```python
@dataclass(frozen=True, slots=True)
class FOMCDriftConfig(InfraExitConfig):
    entry_bar: int = 1
    sl_atr_mult: Decimal = Decimal("2.0")
    require_high_vix: bool = True       # Only trade when VIX > 20
    vix_threshold: Decimal = Decimal("20")
    allow_shorts: bool = False          # Long-only drift
    close_eod: bool = True
    max_day: int = 1
```

**Frequency:** ~8 FOMC press conference days per year. Very low frequency but high conviction per trade.

## File Layout

```
src/stockdownloader/
  strategies/intraday/
    market_context.py          # NEW: MarketContext dataclass + provider protocol
    gao_momentum.py            # NEW: Strategy 1
    noise_boundary.py          # NEW: Strategy 2
    connors_rsi2.py            # NEW: Strategy 3
    vix_regime_filter.py       # NEW: Strategy 4 (wrapper)
    fomc_drift.py              # NEW: Strategy 5
    session.py                 # MODIFY: Add market_ctx field to BarContext
    infra.py                   # MODIFY: Accept + propagate MarketContext
  backtesting/engines/
    intraday.py                # MODIFY: Accept MarketContextProvider param
scripts/
  prepare_market_context.py    # NEW: Data pipeline script
config/strategies/
  intraday_registrations.json  # MODIFY: Add 5 new registrations
tests/strategies/intraday/
  test_gao_momentum.py         # NEW
  test_noise_boundary.py       # NEW
  test_connors_rsi2.py         # NEW
  test_vix_regime_filter.py    # NEW
  test_fomc_drift.py           # NEW
```

## Testing

Each strategy gets 3 test categories:

1. **Unit tests** -- Synthetic price data with known outcomes. E.g., for Gao momentum: construct 78 bars where first 6 bars have positive return, verify long signal fires at bar 73+.
2. **Guard tests** -- Verify strategies correctly decline trades when conditions aren't met (wrong bar of day, no dual signal confirmation, VIX regime mismatch).
3. **Config serialization** -- Round-trip via StrategyConfigMixin.

VixFilteredStrategy gets a special test: wrap an existing strategy, inject MarketContext with different VIX regimes, verify it blocks/allows trades correctly.

Integration test: Run all 5 strategies against real SPY 5-minute data and verify non-zero trade counts.

## Backtest Plan

After implementation, run comparative backtest of:
- All 5 new strategies standalone
- VIX-filtered versions of the 3 best existing strategies (ORReversal, VWAP Pullback, Pattern Scalp)
- Combined portfolio of Gao + RSI(2) + FOMC drift (the research report's recommended portfolio)
