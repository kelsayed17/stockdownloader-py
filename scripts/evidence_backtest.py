"""Backtest evidence-based strategies against SPY 5m data.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/evidence_backtest.py

Runs GaoMomentum, NoiseBoundary, ConnorsRSI2, FOMCDrift and
VIX-filtered variants. Compares against SPY buy-and-hold.
"""
from __future__ import annotations

import csv
import sys
import time
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.intraday.gao_momentum import GaoMomentumStrategy
from stockdownloader.strategies.intraday.noise_boundary import NoiseBoundaryStrategy
from stockdownloader.strategies.intraday.connors_rsi2 import ConnorsRSI2Strategy
from stockdownloader.strategies.intraday.fomc_drift import FOMCDriftStrategy
from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy
from stockdownloader.strategies.intraday.market_context import FileMarketContextProvider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
CONTEXT_CSV = DATA_DIR / "SPY" / "market_context.csv"

INITIAL_CAPITAL = Decimal("100000")
RISK_PER_TRADE = Decimal("0.01")
SLIPPAGE_PCT = Decimal("0.0002")


def load_5m_bars(csv_path: Path) -> list[IntradayPriceData]:
    """Load SPY 5-minute OHLCV bars from CSV."""
    bars: list[IntradayPriceData] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(IntradayPriceData(
                date=row["Datetime"],
                open=Decimal(row["Open"]),
                high=Decimal(row["High"]),
                low=Decimal(row["Low"]),
                close=Decimal(row["Close"]),
                adj_close=Decimal(row["Close"]),
                volume=int(row["Volume"]),
            ))
    return bars


def compute_buy_and_hold(bars: list[IntradayPriceData]) -> Decimal:
    """Compute buy-and-hold return for SPY over the data period."""
    if not bars:
        return Decimal("0")
    first_close = bars[0].close
    last_close = bars[-1].close
    return ((last_close - first_close) / first_close) * 100


def make_engine(**extra: object) -> IntradayBacktestEngine:
    """Create a standard backtest engine."""
    return IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        **extra,
    )


def run_strategy(name: str, strategy, bars: list[IntradayPriceData]) -> dict:
    """Run a single strategy and return metrics dict."""
    engine = make_engine()
    t0 = time.time()
    result = engine.run(strategy, bars)
    elapsed = time.time() - t0

    try:
        sharpe = float(result.sharpe_ratio())
    except Exception:
        sharpe = 0.0
    try:
        sortino = float(result.sortino_ratio())
    except Exception:
        sortino = 0.0
    try:
        calmar = float(result.calmar_ratio())
    except Exception:
        calmar = 0.0

    return {
        "name": name,
        "total_return_pct": float(result.total_return),
        "total_trades": result.total_trades,
        "win_rate_pct": float(result.win_rate),
        "profit_factor": float(result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "avg_win": float(result.average_win),
        "avg_loss": float(result.average_loss),
        "total_pnl": float(result.total_pnl),
        "elapsed_sec": round(elapsed, 1),
    }


def build_strategies(provider: FileMarketContextProvider) -> list[tuple[str, object]]:
    """Build all strategy variants to test."""
    strategies: list[tuple[str, object]] = []

    # ── Gao Momentum variants ──────────────────────────────────────
    strategies.append((
        "Gao Momentum (default)",
        GaoMomentumStrategy(market_ctx_provider=provider),
    ))
    strategies.append((
        "Gao Momentum (dual-signal)",
        GaoMomentumStrategy(
            market_ctx_provider=provider,
            require_dual_signal=True,
            min_r1_magnitude=Decimal("0.0005"),
        ),
    ))
    strategies.append((
        "Gao Momentum (no-VIX)",
        GaoMomentumStrategy(
            market_ctx_provider=provider,
            vix_filter=False,
        ),
    ))
    strategies.append((
        "Gao Momentum (tight SL)",
        GaoMomentumStrategy(
            market_ctx_provider=provider,
            sl_atr_mult=Decimal("1.0"),
        ),
    ))

    # ── Noise Boundary variants ────────────────────────────────────
    strategies.append((
        "Noise Boundary (default)",
        NoiseBoundaryStrategy(market_ctx_provider=provider),
    ))
    strategies.append((
        "Noise Boundary (7d lookback)",
        NoiseBoundaryStrategy(
            market_ctx_provider=provider,
            lookback_days=7,
        ),
    ))
    strategies.append((
        "Noise Boundary (21d lookback)",
        NoiseBoundaryStrategy(
            market_ctx_provider=provider,
            lookback_days=21,
        ),
    ))
    strategies.append((
        "Noise Boundary (wide vol 1.2x)",
        NoiseBoundaryStrategy(
            market_ctx_provider=provider,
            vol_multiplier=Decimal("1.2"),
        ),
    ))

    # ── Connors RSI(2) variants ────────────────────────────────────
    strategies.append((
        "Connors RSI(2) (default, thresh=5)",
        ConnorsRSI2Strategy(market_ctx_provider=provider),
    ))
    strategies.append((
        "Connors RSI(2) (thresh=3)",
        ConnorsRSI2Strategy(
            market_ctx_provider=provider,
            rsi_threshold=Decimal("3"),
        ),
    ))
    strategies.append((
        "Connors RSI(2) (thresh=10)",
        ConnorsRSI2Strategy(
            market_ctx_provider=provider,
            rsi_threshold=Decimal("10"),
        ),
    ))
    strategies.append((
        "Connors RSI(2) (wide SL 3x ATR)",
        ConnorsRSI2Strategy(
            market_ctx_provider=provider,
            sl_atr_mult=Decimal("3.0"),
        ),
    ))

    # ── FOMC Drift variants ────────────────────────────────────────
    strategies.append((
        "FOMC Drift (default, VIX>20)",
        FOMCDriftStrategy(market_ctx_provider=provider),
    ))
    strategies.append((
        "FOMC Drift (no VIX filter)",
        FOMCDriftStrategy(
            market_ctx_provider=provider,
            require_high_vix=False,
        ),
    ))
    strategies.append((
        "FOMC Drift (VIX>25)",
        FOMCDriftStrategy(
            market_ctx_provider=provider,
            vix_threshold=Decimal("25"),
        ),
    ))
    strategies.append((
        "FOMC Drift (VIX>15)",
        FOMCDriftStrategy(
            market_ctx_provider=provider,
            vix_threshold=Decimal("15"),
        ),
    ))

    # ── VIX-Filtered wrappers ──────────────────────────────────────
    # Wrap Gao Momentum with VIX filter
    strategies.append((
        "VIX-Filtered Gao (mid+high)",
        VixFilteredStrategy(
            inner=GaoMomentumStrategy(market_ctx_provider=provider),
            allowed_regimes=["mid", "high"],
            market_ctx_provider=provider,
        ),
    ))
    strategies.append((
        "VIX-Filtered Gao (high only)",
        VixFilteredStrategy(
            inner=GaoMomentumStrategy(market_ctx_provider=provider),
            allowed_regimes=["high", "extreme"],
            market_ctx_provider=provider,
        ),
    ))

    # Wrap Noise Boundary with VIX filter
    strategies.append((
        "VIX-Filtered Noise (mid+high)",
        VixFilteredStrategy(
            inner=NoiseBoundaryStrategy(market_ctx_provider=provider),
            allowed_regimes=["mid", "high"],
            market_ctx_provider=provider,
        ),
    ))

    # Wrap Connors with VIX filter
    strategies.append((
        "VIX-Filtered Connors (mid+high)",
        VixFilteredStrategy(
            inner=ConnorsRSI2Strategy(market_ctx_provider=provider),
            allowed_regimes=["mid", "high"],
            market_ctx_provider=provider,
        ),
    ))

    return strategies


def main() -> None:
    """Run all evidence-based strategy backtests."""
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)
    if not CONTEXT_CSV.exists():
        print(f"ERROR: Market context not found at {CONTEXT_CSV}")
        print("Run: PYTHONPATH=src python3 scripts/prepare_market_context.py")
        sys.exit(1)

    print("=" * 80)
    print("EVIDENCE-BASED STRATEGIES BACKTEST")
    print("=" * 80)

    # Load data
    print("\nLoading SPY 5m bars...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    # Buy and hold benchmark
    bnh_return = compute_buy_and_hold(bars)
    print(f"  SPY Buy-and-Hold return: {bnh_return:.2f}%")

    # Load market context
    provider = FileMarketContextProvider(CONTEXT_CSV)
    print(f"  Market context loaded from {CONTEXT_CSV}")

    # Build strategies
    strats = build_strategies(provider)
    print(f"\nRunning {len(strats)} strategy variants...\n")

    # Run backtests
    results: list[dict] = []
    for i, (name, strat) in enumerate(strats, 1):
        print(f"  [{i}/{len(strats)}] {name}...", end=" ", flush=True)
        try:
            r = run_strategy(name, strat, bars)
            results.append(r)
            print(f"Return={r['total_return_pct']:+.2f}% | "
                  f"Trades={r['total_trades']} | "
                  f"Win={r['win_rate_pct']:.1f}% | "
                  f"Sharpe={r['sharpe']:.2f} | "
                  f"MaxDD={r['max_drawdown_pct']:.2f}% | "
                  f"{r['elapsed_sec']}s")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "name": name,
                "total_return_pct": 0.0,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_pnl": 0.0,
                "elapsed_sec": 0.0,
                "error": str(e),
            })

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBenchmark: SPY Buy-and-Hold = {bnh_return:+.2f}%\n")

    # Sort by total return
    results.sort(key=lambda x: x["total_return_pct"], reverse=True)

    # Header
    print(f"{'Strategy':<40} {'Return%':>8} {'Trades':>7} {'Win%':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PnL$':>10}")
    print("-" * 110)

    for r in results:
        beat = "**" if r["total_return_pct"] > float(bnh_return) else "  "
        print(f"{r['name']:<40} {r['total_return_pct']:>+7.2f}% "
              f"{r['total_trades']:>7} {r['win_rate_pct']:>5.1f}% "
              f"{r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['sortino']:>8.2f} {r['max_drawdown_pct']:>6.2f}% "
              f"{r['total_pnl']:>10.2f} {beat}")

    # Strategies that beat buy-and-hold
    winners = [r for r in results if r["total_return_pct"] > float(bnh_return)]
    print(f"\n{'=' * 80}")
    if winners:
        print(f"WINNERS (beat B&H {bnh_return:+.2f}%): {len(winners)}/{len(results)}")
        for w in winners:
            print(f"  * {w['name']}: {w['total_return_pct']:+.2f}% "
                  f"(Sharpe={w['sharpe']:.2f}, MaxDD={w['max_drawdown_pct']:.2f}%)")
    else:
        print(f"NO strategies beat buy-and-hold ({bnh_return:+.2f}%)")

    # Best risk-adjusted
    if results:
        best_sharpe = max(results, key=lambda x: x["sharpe"])
        best_sortino = max(results, key=lambda x: x["sortino"])
        best_pf = max(results, key=lambda x: x["profit_factor"])
        lowest_dd = min((r for r in results if r["total_trades"] > 0),
                        key=lambda x: x["max_drawdown_pct"], default=None)

        print(f"\nBest Sharpe:        {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
        print(f"Best Sortino:       {best_sortino['name']} ({best_sortino['sortino']:.2f})")
        print(f"Best Profit Factor: {best_pf['name']} ({best_pf['profit_factor']:.2f})")
        if lowest_dd:
            print(f"Lowest Max DD:      {lowest_dd['name']} ({lowest_dd['max_drawdown_pct']:.2f}%)")

    # Save results CSV
    out_csv = DATA_DIR / "SPY" / "evidence_backtest_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "total_return_pct", "total_trades", "win_rate_pct",
            "profit_factor", "max_drawdown_pct", "sharpe", "sortino",
            "calmar", "avg_win", "avg_loss", "total_pnl", "elapsed_sec",
        ])
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\nResults saved to {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
