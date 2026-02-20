# Stock Downloader

A Python-based stock market analysis and backtesting platform for algorithmic trading strategy development, optimization, and validation. It supports equity, intraday, and options strategies across daily and 5-minute timeframes, with comprehensive technical analysis and PineScript export for TradingView.

## Features

- **Historical data download** from Yahoo Finance with automatic authentication
- **Intraday data** from Yahoo Finance and Polygon.io (5-minute bars with accumulation support)
- **16 trading strategies** across daily, intraday, and options categories (7 daily, 7 intraday, 2 options)
- **Strategy registry** with CLI name resolution and parameter spaces for optimization
- **Backtesting engines** for equity, intraday (with slippage and short selling), and options
- **Strategy optimization** with greedy-sequential parameter search and composite fitness scoring
- **Walk-forward validation** to prevent overfitting with in-sample/out-of-sample splits
- **Tournament framework** for comparing strategies head-to-head
- **Signal generation** with confluence scoring across trend, momentum, volume, and volatility indicators
- **PineScript generation** for exporting strategies to TradingView (Pine Script v6, indicator and strategy modes)
- **20+ technical indicators** with streaming O(1) incremental variants for backtesting performance
- **Options pricing** via Black-Scholes with historical volatility
- **Decimal precision** throughout for financial accuracy

## Requirements

- Python 3.11+
- pip

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package (enables CLI commands)
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

After installation with `pip install -e .`, all commands are available as CLI entry points:

```bash
# Run equity backtest on any symbol
spy-backtest AAPL
spy-backtest SPY --csv spy_data.csv

# Run intraday backtest
intraday-backtest --csv data/spy/5m_bars.csv

# Full symbol analysis with confluence alerts
symbol-analysis AAPL
symbol-analysis SPY 2y

# Optimize strategy parameters
strategy-optimize vwap-pullback --csv data/spy/5m_bars.csv

# Compare all strategies in a tournament
grand-tournament --csv data/spy/5m_bars.csv

# Generate PineScript for TradingView
generate-pinescript rsi
generate-pinescript vwap --output vwap_v11.pine

# List available strategies
spy-backtest --list-strategies
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `spy-backtest` | Run equity strategies against daily data |
| `options-backtest` | Run covered call and protective put strategies |
| `symbol-analysis` | Full analysis suite with confluence alerts |
| `trend-analysis` | Scan stock universes for price patterns |
| `intraday-backtest` | Backtest intraday strategies on 5-minute bars |
| `intraday-accumulate` | Download and accumulate intraday data |
| `strategy-optimize` | Optimize strategy parameters |
| `grand-tournament` | Compare all strategies head-to-head |
| `multi-timeframe-optimizer` | Optimize across multiple timeframes |
| `multi-timeframe-tournament` | Tournament across timeframes |
| `dmi-vwap-backtest` | Run the DMI VWAP composite strategy |
| `generate-pinescript` | Export strategies to TradingView Pine Script |
| `exit-tournament` | Compare exit mechanisms |
| `signal-stack-tournament` | Signal stacking strategy tournament |

## Trading Strategies

### Strategy Registry

All strategies are registered in a central registry with CLI names, enabling programmatic creation and optimization:

```python
from stockdownloader.strategy.registry import StrategyRegistry
from stockdownloader.strategy.registrations import ensure_registered

ensure_registered()
strategy = StrategyRegistry.create("rsi", period=21, oversold=25.0)
```

### Daily Strategies

All equity strategies start with $100,000 initial capital and zero commission.

| CLI Name | Strategy | Description |
|----------|----------|-------------|
| `sma` | SMA Crossover | Trend-following using SMA crossovers (default 9/21) |
| `rsi` | RSI | Mean-reversion on RSI oversold/overbought levels |
| `macd` | MACD | Momentum via MACD/signal line crossovers |
| `bollinger` | Bollinger Band + RSI | Combined volatility and momentum signals |
| `momentum` | Momentum Confluence | Multi-signal confirmation (MACD, RSI, Stochastic) |
| `breakout` | Breakout | Channel breakout based on high/low ranges |
| `multi` | Multi-Indicator | Confluence scoring across multiple indicators |

#### SMA Crossover

Long-term trend-following using moving average crossovers.

- **BUY**: Short SMA crosses above long SMA
- **SELL**: Short SMA crosses below long SMA
- **Default params**: short_period=9, long_period=21

#### RSI

Mean-reversion strategy using Relative Strength Index.

- **BUY**: RSI crosses above oversold threshold
- **SELL**: RSI crosses below overbought threshold
- **Default params**: period=14, oversold=30, overbought=70

#### MACD

Momentum strategy using Moving Average Convergence Divergence.

- **BUY**: MACD line crosses above signal line
- **SELL**: MACD line crosses below signal line
- **Default params**: fast=12, slow=26, signal=9

#### Bollinger Band + RSI

- **BUY**: Price touches lower Bollinger Band AND RSI < 30
- **SELL**: Price touches upper Bollinger Band AND RSI > 70

#### Momentum Confluence

Multi-signal momentum confirmation using MACD, RSI, and Stochastic.

#### Breakout

Channel breakout strategy based on high/low price channels.

#### Multi-Indicator

Scores multiple indicators and requires confluence for signals.

### Intraday Strategies

Intraday strategies operate on 5-minute bars and support both LONG and SHORT positions with configurable stop-loss, take-profit, slippage (default 5 bps), and risk-per-trade position sizing (default 1%).

| CLI Name | Strategy | Description |
|----------|----------|-------------|
| `vwap-pullback` | VWAP Pullback | Entries on pullbacks to VWAP in trending markets |
| `vwap-reversal` | VWAP Reversal | Bollinger Band reversal trades |
| `vwap-orb` | OR Breakout | Opening Range Breakout with RVOL confirmation |
| `vwap-orr` | OR Reversal | Opening Range mean reversion |
| `vwap-ps` | Pattern Scalp | Quick scalps on candlestick patterns |
| `avwap-pullback` | AVWAP Pullback | Anchored VWAP pullback with zone detection |
| `smc-structure` | SMC Structure | Smart Money Concepts order block entries |
| `ml-oversold` | ML Oversold | ML-driven mean-reversion (gradient boosting model decides entries) |

Additional intraday strategies (not registry-registered):

- **DMI VWAP** — Directional Movement Index with VWAP, run via `dmi-vwap-backtest`

### Options Strategies

Options strategies start with $100,000 initial capital and $0.65/contract commission. Premiums estimated via Black-Scholes with 20-day historical volatility.

| CLI Name | Strategy | Description |
|----------|----------|-------------|
| `covered-call` | Covered Call | Sell OTM calls against long stock for income |
| `protective-put` | Protective Put | Buy OTM puts to hedge downside risk |

#### Covered Call Variants

| Variant | MA Period | OTM % | DTE | Exit Threshold |
|---------|-----------|-------|-----|----------------|
| Aggressive income | 20 | 3% | 30 | 3% |
| Standard | 20 | 5% | 30 | 3% |
| Conservative | 50 | 5% | 45 | 4% |

#### Protective Put Variants

| Variant | MA Period | OTM % | DTE | Momentum Lookback |
|---------|-----------|-------|-----|--------------------|
| Standard hedge | 20 | 5% | 30 | 5 bars |
| Aggressive protection | 20 | 3% | 45 | 10 bars |
| Conservative long-term | 50 | 5% | 60 | 10 bars |

## Optimization & Validation

### Strategy Optimizer

Greedy-sequential parameter optimization with composite fitness scoring:

```
score = (sharpe * 40) + (win_rate * 20) + (profit_factor * 20) - (max_drawdown * 20)
        - trade_penalty + trade_bonus + trades_per_day_bonus
```

Minimum 80 trades required, targeting ~0.5 trades/day.

```bash
strategy-optimize vwap-pullback --csv data/spy/5m_bars.csv
```

### Walk-Forward Validation

Anti-overfitting framework with in-sample/out-of-sample splits and degradation ratio tracking.

### Tournament Framework

- **Grand Tournament**: All strategies compete on the same dataset
- **Exit Tournament**: Tests multiple exit mechanisms against each entry logic
- **Signal Stack Tournament**: Evaluates signal stacking combinations
- **Multi-Timeframe Tournament**: Cross-timeframe strategy comparison

## Technical Indicators

20+ indicators implemented with both batch and streaming (O(1) incremental) variants:

| Category | Indicators |
|----------|------------|
| Trend | SMA, EMA, Ichimoku Cloud, Parabolic SAR, ADX (+DI/-DI) |
| Momentum | RSI, MACD, Stochastic (%K/%D), Williams %R, CCI, ROC |
| Volume | OBV, MFI, VWAP (session-anchored with std dev bands) |
| Volatility | Bollinger Bands, ATR |
| Other | Fibonacci Retracement, Support/Resistance detection |

Incremental indicators in `incremental_indicators.py` provide O(1) per-bar updates for RSI, EMA, ATR, ADX, MACD, Parabolic SAR, and session-anchored VWAP, preventing O(n^2) recalculation during backtesting.

## PineScript Generation

20 strategies can be exported to TradingView Pine Script v6 with configurable inputs, signal labels, alert conditions, and background coloring. Two output modes:

- **Indicator mode** — visual overlays with manual position tracking, alertcondition() triggers
- **Strategy mode** — full backtesting with strategy.entry()/exit(), ATR position sizing, breakeven management, circuit breaker, EOD close

```bash
generate-pinescript rsi
generate-pinescript vwap --output vwap_v11.pine
generate-pinescript --list                        # Show all 20 available strategies
```

## Data Sources

| Source | Data Type | Module |
|--------|-----------|--------|
| Yahoo Finance | Daily OHLCV, intraday, options chains, quotes | `yahoo_data_client.py` |
| Polygon.io | Intraday bars | `polygon_data_client.py` |
| Morningstar | Financial data | `morningstar_client.py` |
| CSV files | Daily and intraday price data | `csv_price_data_loader.py`, `intraday_csv_loader.py` |
| TradingView | Trade import | `tradingview_trade_loader.py` |

## Project Structure

```
src/stockdownloader/
  app/                                  # CLI entry points (17 apps)
    spy_backtest_app.py                   # Equity backtest runner
    options_backtest_app.py               # Options backtest runner
    intraday_backtest_app.py              # Intraday backtest runner
    # intraday_accumulate_app.py merged into backtest_app.py (main_accumulate)
    symbol_analysis_app.py                # Full analysis with confluence alerts
    trend_analysis_app.py                 # Stock universe pattern scanner
    optimize_app.py                       # Strategy parameter optimizer
    grand_tournament.py                   # All-strategy tournament
    exit_tournament_app.py                # Exit mechanism comparison
    signal_stack_tournament.py            # Signal stacking tournament
    multi_timeframe_optimizer.py          # Multi-timeframe optimization
    multi_timeframe_tournament.py         # Multi-timeframe tournament
    dmi_vwap_backtest.py                  # DMI VWAP strategy runner
    generate_pinescript.py                # PineScript export
  strategy/                             # Trading strategies
    trading_strategy.py                   # Daily strategy ABC
    intraday_trading_strategy.py          # Intraday strategy ABC
    options_strategy.py                   # Options strategy ABC
    registry.py                           # Strategy registry
    registrations.py                      # Strategy registration definitions
    daily_to_intraday_adapter.py          # Adapts daily strategies for intraday use
    dmi_vwap_strategy.py                  # DMI VWAP composite strategy
    ensemble_strategy.py                  # Ensemble strategy combiner
    exit_mechanism.py                     # Configurable exit mechanisms
    regime_detector.py                    # Market regime detection
    daily/                                # Daily strategies (7)
      sma_crossover_strategy.py
      rsi_strategy.py
      macd_strategy.py
      bollinger_band_rsi_strategy.py
      momentum_confluence_strategy.py
      breakout_strategy.py
      multi_indicator_strategy.py
    intraday/                             # Intraday strategies (7 registered + infra)
      or_breakout_strategy.py               # Opening Range Breakout
      or_reversal_strategy.py               # Opening Range Reversal
      pullback_strategy.py                  # VWAP Pullback
      pattern_scalp_strategy.py             # Pattern Scalp
      reversal_strategy.py                  # Bollinger Band Reversal
      avwap_pullback_strategy.py            # Anchored VWAP Pullback
      smc_structure_strategy.py             # Smart Money Concepts
      trail_strategy.py                     # Trailing stop strategies (ATR, VWAP, BE)
      infra.py                              # Shared intraday infrastructure (BarContext)
      exit_manager.py                       # Intraday exit management
      entry_helpers.py                      # Shared entry signal helpers
      base_config.py                        # InfraExitConfig base for all configs
      *_config.py                           # Per-strategy frozen config dataclasses
    signals/                              # Signal stacking framework
      signal_generator.py                   # Signal generator ABC
      signal_registry.py                    # Signal registry
      stacked_signal_engine.py              # Multi-signal confluence engine
      stacked_daily_strategy.py             # Signal-stacked daily strategy adapter
      stacked_intraday_strategy.py          # Signal-stacked intraday strategy adapter
      multi_timeframe_aligner.py            # Multi-timeframe signal alignment
      generators/                           # Atomic signal generators (trend, momentum, volume, volatility)
    exit_mechanisms/                       # Pluggable exit strategies (6)
      atr_trail_exit.py                     # ATR-based trailing stop
      hybrid_exit.py                        # Hybrid multi-exit
      vwap_band_exit.py                     # VWAP band exit
      vwap_cross_exit.py                    # VWAP cross exit
      trailing_stop_exit.py                 # Simple trailing stop
      time_decay_exit.py                    # Time-based exit
    options/                              # Options strategies (2)
      covered_call_strategy.py
      protective_put_strategy.py
  backtest/                             # Backtesting engines
    backtest_engine.py                    # Daily equity engine
    intraday_backtest_engine.py           # Intraday engine (long/short, slippage)
    options_backtest_engine.py            # Options engine (Black-Scholes)
    backtest_result.py                    # Result metrics
    report_formatter.py                   # Report output formatting
    strategy_optimizer.py                 # Intraday strategy optimizer
    daily_strategy_optimizer.py           # Daily strategy optimizer
    optimizer_scoring.py                  # Composite fitness scoring
    combinatorial_tester.py               # Grid search parameter testing
    exit_tournament_engine.py             # Exit mechanism tournament
    walk_forward.py                       # Walk-forward validation
  model/                                # Data models (frozen dataclasses)
    price_data.py                         # Daily OHLCV
    intraday_price_data.py                # Intraday OHLCV with datetime
    intraday_signal.py                    # Signal with stops and confluence
    trade.py                              # Equity trade tracking
    options.py                            # Option contract and chain
    quote_data.py                         # Quote data
    unified_market_data.py                # Consolidated symbol data
    indicator_values.py                   # Technical indicator snapshot
    alert_result.py                       # Trading alert with recommendations
    pattern_result.py                     # Pattern analysis result
    tournament_trade.py                   # Tournament trade tracking
  data/                                 # Data fetching and I/O
    yahoo_data_client.py                  # Yahoo Finance API client
    yahoo_auth_helper.py                  # Yahoo authentication
    polygon_data_client.py                # Polygon.io client
    morningstar_client.py                 # Morningstar financial data
    csv_price_data_loader.py              # Daily CSV parser
    intraday_csv_loader.py                # Intraday CSV parser
    intraday_csv_writer.py                # Intraday CSV writer
    intraday_data_accumulator.py          # Intraday data accumulation
    tradingview_trade_loader.py           # TradingView trade import
    stock_list_downloader.py              # Stock list downloads
  util/                                 # Utilities and indicators
    technical_indicators.py               # 20+ batch indicators
    incremental_indicators.py             # O(1) streaming indicators
    indicator_hub.py                      # Indicator caching layer
    intraday_indicators.py                # Intraday-specific indicators
    moving_average_calculator.py          # SMA and EMA
    black_scholes_calculator.py           # Option pricing and greeks
    big_decimal_math.py                   # Decimal arithmetic helpers
    crossover.py                          # Crossover detection
    date_helper.py                        # Market date calculations
    csv_parser.py                         # CSV parsing utilities
    file_helper.py                        # File I/O utilities
    retry_executor.py                     # Retry logic
    timeframe_aggregator.py               # Multi-timeframe aggregation
    pinescript_generator.py               # PineScript v6 generator (indicator + strategy modes)
    pinescript_models.py                  # PineScript data models
    pinescript_strategies.py              # Pre-built PineScript strategies + STRATEGY_CATALOG
    pinescript_spy_strategies.py          # SPY tournament winner strategies (v1 indicator, v2 strategy)
    pinescript_composites.py              # Multi-mode composite PineScript strategies
    pinescript_gme_prediction.py          # GME regime prediction indicator
  analysis/                             # Analysis tools
    signal_generator.py                   # Trading alerts with confluence
    formula_calculator.py                 # Stock valuation formulas
    pattern_analyzer.py                   # Price pattern analysis
tests/                                  # Test suite (2,900+ tests)
  model/                                  # Model unit tests
  util/                                   # Utility unit tests
  strategy/                               # Strategy tests (daily, intraday, options)
  backtest/                               # Backtest engine tests
  data/                                   # Data layer tests
  analysis/                               # Analysis unit tests
  integration/                            # Integration tests
  e2e/                                    # End-to-end tests
  live/                                   # Live API tests (skipped by default)
scripts/                                # Utility scripts
  profile_strategy.py                     # Strategy profiling tool
  archive/                                # Archived research scripts (8)
data/                                   # Market data organized by symbol
  spy/
    5m_bars.csv                           # SPY 5-minute intraday bars
  cache/                                  # Auto-managed data cache (.gitignored)
docs/                                   # Documentation
  reports/                                # Backtest and tournament reports (6)
  gme/                                    # GME analysis and PineScript indicators
  pinescript/                             # PineScript documentation
output/                                 # Generated artifacts
  pinescript/                             # PineScript v6 files organized by category
    general/                                # Generic strategies (rsi, macd, sma, etc.)
    spy/                                    # SPY-specific strategies
    gme/                                    # GME-specific indicators
    composite/                              # Multi-mode composite strategies
  models/spy/                             # Trained ML models
  patterns/spy/                           # Pattern catalogs by timeframe
```

## Tests

```bash
# Run all tests (excludes live API tests)
pytest

# Run with coverage
pytest --cov=stockdownloader

# Run specific modules
pytest tests/strategy/
pytest tests/backtest/
pytest tests/e2e/

# Run live API tests (requires network)
pytest -m live
```

## Programmatic Usage

```python
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.strategy.daily.simple_strategies import RSIStrategy
from stockdownloader.backtest.backtest_engine import BacktestEngine
from decimal import Decimal

# Fetch data
client = YahooDataClient()
data = client.fetch_price_data("SPY", "5y", "1d")

# Create strategy
strategy = RSIStrategy(period=14, oversold=30, overbought=70)

# Run backtest
engine = BacktestEngine(initial_capital=Decimal("100000"))
result = engine.run(strategy, data)

print(f"Return: {result.total_return}%")
print(f"Sharpe: {result.sharpe_ratio()}")
print(f"Win Rate: {result.win_rate}%")
print(f"Max Drawdown: {result.max_drawdown}%")
```

## Documentation

Detailed analysis reports are available in `docs/reports/`:

| Report | Description |
|--------|-------------|
| [Grand Tournament Results](docs/reports/grand_tournament_results.md) | Walk-forward validated strategy rankings (12 candidates, 501 sessions) |
| [Optimizer Tournament Results](docs/reports/optimizer_tournament_results.md) | Multi-timeframe parameter optimization (1,080 configurations) |
| [Backtest Findings](docs/reports/backtest_findings.md) | Parameter sensitivity analysis and optimization results |
| [Exit Tournament Report](docs/reports/exit_tournament_report.md) | Comparative analysis of 8 exit mechanisms |
| [DMI VWAP Strategy Report](docs/reports/dmi_vwap_strategy_report.md) | DMI + VWAP composite strategy specification |
| [Code Audit Report](docs/reports/code_audit_report.md) | Code quality audit with 28 identified issues |

## Disclaimer

This software is for **educational purposes only**. It is not financial advice. Past backtest performance does not guarantee future results. Options trading involves significant risk of loss. Always do your own research before making investment decisions.
