"""Quick profiling script to measure per-strategy backtest times."""
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.base_registry import StrategyRegistry

from stockdownloader.strategy.registration_loader import ensure_registered
ensure_registered()


def main():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "spy" / "5m_bars.csv"
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        return

    data = IntradayCsvLoader.load_from_file(csv_path)
    print(f"Loaded {len(data)} bars")

    engine = IntradayBacktestEngine(
        initial_capital=Decimal("100000"),
        risk_per_trade=Decimal("0.01"),
    )

    # Test just a few strategies
    strategies_to_test = ["rsi", "sma", "macd", "momentum", "multi"]

    for name in strategies_to_test:
        try:
            entry = StrategyRegistry.get(name)
        except ValueError:
            print(f"Strategy {name!r} not found")
            continue

        strategy = DailyToIntradayAdapter(entry.factory(**entry.default_kwargs))

        start = time.time()
        result = engine.run(strategy, data)
        elapsed = time.time() - start

        print(
            f"{name:12s}: {elapsed:7.1f}s | "
            f"P/L=${float(result.total_pnl):>10,.2f} | "
            f"WR={float(result.win_rate):5.1f}% | "
            f"Trades={result.total_trades}"
        , flush=True)


if __name__ == "__main__":
    main()
