"""Unified entry point for dynamic symbol analysis, backtesting, and alert generation.

Fetches live data from Yahoo Finance for any symbol, then:
1. Displays current indicator snapshot
2. Generates trading alerts with buy/sell signals
3. Provides options recommendations (calls, puts, strike prices)
4. Runs all equity strategies as backtests
5. Runs all options strategies as backtests
6. Compares results

Usage:
    python -m stockdownloader.app.symbol_analysis_app SYMBOL [RANGE]

Supported ranges: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max (default: 5y)
"""
from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.analysis.signal_generator import generate_alert
from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_report_formatter import BacktestReportFormatter
from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
from stockdownloader.backtest.options_backtest_report_formatter import OptionsBacktestReportFormatter
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.strategy.sma_crossover_strategy import SMACrossoverStrategy
from stockdownloader.strategy.rsi_strategy import RSIStrategy
from stockdownloader.strategy.macd_strategy import MACDStrategy
from stockdownloader.strategy.bollinger_band_rsi_strategy import BollingerBandRSIStrategy
from stockdownloader.strategy.momentum_confluence_strategy import MomentumConfluenceStrategy
from stockdownloader.strategy.breakout_strategy import BreakoutStrategy
from stockdownloader.strategy.multi_indicator_strategy import MultiIndicatorStrategy
from stockdownloader.strategy.covered_call_strategy import CoveredCallStrategy
from stockdownloader.strategy.protective_put_strategy import ProtectivePutStrategy

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = Decimal("100000.00")
EQUITY_COMMISSION = Decimal("0")
OPTIONS_COMMISSION = Decimal("0.65")


def _print_summary(
    symbol: str,
    data: list,
    alert,
    equity_results: list,
    options_results: list,
) -> None:
    """Print the unified analysis summary."""
    print()
    print("\u2554" + "\u2550" * 66 + "\u2557")
    print("\u2551" + "                      ANALYSIS SUMMARY                           " + "\u2551")
    print("\u255a" + "\u2550" * 66 + "\u255d")
    print()
    print(f"  Symbol:            {symbol}")
    print(f"  Current Price:     ${data[-1].close.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print(f"  Signal:            {alert.direction.value}")
    print(f"  Confluence:        {alert.signal_strength}")
    print()

    print("  Call Recommendation:")
    print(f"    {alert.call_recommendation}")
    print("  Put Recommendation:")
    print(f"    {alert.put_recommendation}")
    print()

    # Best equity strategy
    best_equity = None
    for r in equity_results:
        if best_equity is None or r.total_return > best_equity.total_return:
            best_equity = r
    if best_equity is not None:
        print(
            f"  Best Equity Strategy:    {best_equity.strategy_name} "
            f"({best_equity.total_return.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}% return)"
        )

    # Best options strategy
    best_options = None
    for r in options_results:
        if best_options is None or r.total_return > best_options.total_return:
            best_options = r
    if best_options is not None:
        print(
            f"  Best Options Strategy:   {best_options.strategy_name} "
            f"({best_options.total_return.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}% return)"
        )

    print()
    print("  DISCLAIMER: This is for educational purposes only.")
    print("  Not financial advice. Past performance does not guarantee future results.")
    print("  Always do your own research before trading.")
    print()


def _print_usage() -> None:
    """Print usage information."""
    print("Stock Analysis & Backtesting Platform")
    print()
    print("Usage: symbol_analysis_app SYMBOL [RANGE]")
    print()
    print("Arguments:")
    print("  SYMBOL   Ticker symbol to analyze (e.g., AAPL, SPY, TSLA, MSFT)")
    print("  RANGE    Historical data range (default: 5y)")
    print("           Options: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max")
    print()
    print("Examples:")
    print("  symbol_analysis_app AAPL          # Analyze Apple with 5 years data")
    print("  symbol_analysis_app SPY 2y        # Analyze SPY with 2 years data")
    print("  symbol_analysis_app TSLA 1y       # Analyze Tesla with 1 year data")
    print("  symbol_analysis_app MSFT max      # Analyze Microsoft with all data")
    print()
    print("The tool will:")
    print("  1. Fetch live data from Yahoo Finance")
    print("  2. Compute 20+ technical indicators")
    print("  3. Generate buy/sell alerts with confluence scoring")
    print("  4. Recommend options trades (calls/puts with strike prices)")
    print("  5. Backtest 9 equity strategies and 6 options strategies")
    print("  6. Compare all strategy performance")


def main() -> None:
    """Entry point for the unified symbol analysis application."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis & Backtesting Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported ranges: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max (default: 5y)\n"
            "\n"
            "Examples:\n"
            "  symbol_analysis_app AAPL          # Analyze Apple with 5 years data\n"
            "  symbol_analysis_app SPY 2y        # Analyze SPY with 2 years data\n"
            "  symbol_analysis_app TSLA 1y       # Analyze Tesla with 1 year data\n"
        ),
    )
    parser.add_argument("symbol", help="Ticker symbol to analyze (e.g., AAPL, SPY, TSLA)")
    parser.add_argument(
        "range",
        nargs="?",
        default="5y",
        help="Historical data range (default: 5y). Options: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    data_range = args.range

    print("\u2554" + "\u2550" * 66 + "\u2557")
    print("\u2551" + "           STOCK ANALYSIS & BACKTESTING PLATFORM                 " + "\u2551")
    print("\u255a" + "\u2550" * 66 + "\u255d")
    print()

    # === PHASE 1: Fetch Data ===
    print(f"Fetching {symbol} data from Yahoo Finance (range: {data_range})...")
    client = YahooDataClient()
    data = client.fetch_price_data(symbol, data_range, "1d")

    if not data:
        print(f"ERROR: Could not fetch data for symbol '{symbol}'.")
        print("Verify the symbol is valid and try again.")
        return

    print(f"Loaded {len(data)} trading days for {symbol}")
    print(f"Date range: {data[0].date} to {data[-1].date}")
    print(f"Current price: ${data[-1].close.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print(f"Starting capital: ${INITIAL_CAPITAL.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print()

    # === PHASE 2: Generate Alerts ===
    print("Analyzing indicators and generating signals...")
    print()
    alert = generate_alert(symbol, data)
    print(alert)

    # === PHASE 3: Run Equity Backtests ===
    print()
    print("\u2554" + "\u2550" * 66 + "\u2557")
    print("\u2551" + "                    EQUITY STRATEGY BACKTESTS                    " + "\u2551")
    print("\u255a" + "\u2550" * 66 + "\u255d")
    print()

    equity_strategies = [
        # Classic strategies
        SMACrossoverStrategy(50, 200),
        SMACrossoverStrategy(20, 50),
        RSIStrategy(14, 30, 70),
        RSIStrategy(14, 25, 75),
        MACDStrategy(12, 26, 9),
        # Multi-indicator strategies
        BollingerBandRSIStrategy(),
        MomentumConfluenceStrategy(),
        BreakoutStrategy(),
        MultiIndicatorStrategy(),
    ]

    equity_engine = BacktestEngine(INITIAL_CAPITAL, EQUITY_COMMISSION)
    equity_results = []

    for strategy in equity_strategies:
        try:
            print(f"Running: {strategy.get_name()}...")
            result = equity_engine.run(strategy, data)
            equity_results.append(result)
            BacktestReportFormatter.print_report(result, data)
        except Exception as exc:
            logger.warning("Failed to run strategy %s: %s", strategy.get_name(), exc)

    if equity_results:
        BacktestReportFormatter.print_comparison(equity_results, data)

    # === PHASE 4: Run Options Backtests ===
    print()
    print("\u2554" + "\u2550" * 66 + "\u2557")
    print("\u2551" + "                   OPTIONS STRATEGY BACKTESTS                    " + "\u2551")
    print("\u255a" + "\u2550" * 66 + "\u255d")
    print()

    options_strategies = [
        CoveredCallStrategy(20, Decimal("0.03"), 30, Decimal("0.03")),
        CoveredCallStrategy(20, Decimal("0.05"), 30, Decimal("0.03")),
        CoveredCallStrategy(50, Decimal("0.05"), 45, Decimal("0.04")),
        ProtectivePutStrategy(20, Decimal("0.05"), 30, 5),
        ProtectivePutStrategy(20, Decimal("0.03"), 45, 10),
        ProtectivePutStrategy(50, Decimal("0.05"), 60, 10),
    ]

    options_engine = OptionsBacktestEngine(INITIAL_CAPITAL, OPTIONS_COMMISSION)
    options_results = []

    for strategy in options_strategies:
        try:
            print(f"Running: {strategy.get_name()}...")
            result = options_engine.run(strategy, data)
            options_results.append(result)
            OptionsBacktestReportFormatter.print_report(result)
        except Exception as exc:
            logger.warning("Failed to run options strategy %s: %s", strategy.get_name(), exc)

    if options_results:
        OptionsBacktestReportFormatter.print_comparison(options_results)

    # === PHASE 5: Summary ===
    _print_summary(symbol, data, alert, equity_results, options_results)


if __name__ == "__main__":
    main()
