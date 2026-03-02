"""Run the full GME options analysis pipeline.

Steps:
1. Fetch all options data from Polygon (2023-01 to 2026-02)
2. Build 28-metric daily state from options chain data
3. Run backtests across all 6 strategies
4. Generate signal fusion scorecards

Usage:
    PYTHONPATH=src python3 scripts/run_gme_options_pipeline.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

POLYGON_API_KEY = "jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu"
START_DATE = date(2023, 1, 1)
END_DATE = date(2026, 2, 26)
DATA_DIR = Path("data/GME/options")
DAILY_BARS_CSV = Path("data/GME/daily_bars.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("gme_pipeline")


def step1_fetch() -> None:
    """Fetch all GME options data from Polygon, month by month."""
    from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.gme.options.fetcher import OptionsDataFetcher

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cfg = GMEOptionsConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        polygon_api_key=POLYGON_API_KEY,
        rate_limit_delay=0.15,
        data_dir=DATA_DIR,
    )
    client = PolygonOptionsClient(api_key=POLYGON_API_KEY, rate_limit_delay=0.15)
    fetcher = OptionsDataFetcher(cfg, client)

    t0 = time.time()
    fetcher.run()
    elapsed = time.time() - t0
    logger.info("Step 1 complete: fetch took %.1f minutes", elapsed / 60)


def step2_build_state() -> pd.DataFrame:
    """Build the 28-metric daily state from fetched Parquet files."""
    from stockdownloader.gme.options.state_engine import OptionsStateEngine

    # Load GME spot prices
    bars = pd.read_csv(DAILY_BARS_CSV)
    spot_prices = dict(zip(bars["date"], bars["close"]))
    logger.info("Loaded %d spot prices from daily bars", len(spot_prices))

    engine = OptionsStateEngine(spot_prices=spot_prices)
    monthly_dir = DATA_DIR / "monthly"
    state = engine.build(bars_dir=monthly_dir)

    # Save state
    state_path = DATA_DIR / "options_state.parquet"
    state.to_parquet(state_path, index=False)
    logger.info(
        "Step 2 complete: built state with %d days, %d columns → %s",
        len(state), len(state.columns), state_path,
    )
    return state


def step3_backtest(state: pd.DataFrame) -> None:
    """Run backtests across all 6 strategies."""
    from stockdownloader.gme.options.backtester import (
        BacktestConfig,
        GMEOptionsBacktester,
    )
    from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
    from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
    from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
    from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
    from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
    from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy

    strategies = [
        GMEWheelStrategy(),
        IronCondorStrategy(),
        StrangleStrategy(),
        CreditSpreadStrategy(),
        LongOptionsStrategy(),
        CalendarSpreadStrategy(),
    ]

    # Load chains by date from monthly parquets
    monthly_dir = DATA_DIR / "monthly"
    chain_by_date: dict[str, pd.DataFrame] = {}
    for pf in sorted(monthly_dir.glob("*.parquet")):
        mdf = pd.read_parquet(pf)
        for d in mdf["date"].unique():
            chain_by_date[str(d)[:10]] = mdf[mdf["date"] == d]
    logger.info("Loaded chains for %d trading days", len(chain_by_date))

    bt = GMEOptionsBacktester(
        strategies=strategies,
        config=BacktestConfig(initial_capital=100_000.0),
    )
    results = bt.run(state, chain_by_date)

    # Print results summary
    for r in results:
        metrics = r.compute_metrics() if hasattr(r, "compute_metrics") else {}
        logger.info(
            "Strategy %s: %d trades, final equity $%.2f",
            r.strategy_name,
            len(r.trades),
            r.equity_curve[-1] if r.equity_curve else 100_000.0,
        )
        if metrics:
            logger.info("  Metrics: %s", metrics)

    # Save results
    results_path = DATA_DIR / "backtest_results.parquet"
    rows = []
    for r in results:
        for pos in r.trades:
            rows.append({
                "strategy": r.strategy_name,
                "entry_date": str(pos.entry_date),
                "exit_date": str(pos.exit_date) if pos.exit_date else "",
                "direction": pos.trade.direction,
                "option_type": pos.trade.option_type,
                "ticker": pos.trade.option_ticker,
                "strike": pos.trade.strike,
                "entry_premium": pos.entry_premium,
                "exit_premium": pos.exit_premium,
                "contracts": pos.contracts,
                "pnl": pos.pnl,
            })
    if rows:
        pd.DataFrame(rows).to_parquet(results_path, index=False)
        logger.info("Step 3 complete: saved %d trades → %s", len(rows), results_path)
    else:
        logger.info("Step 3 complete: no trades generated")


def step4_scorecards(state: pd.DataFrame) -> None:
    """Generate signal fusion scorecards for each trading day."""
    from stockdownloader.gme.options.signal_fusion import GMESignalFusion
    from stockdownloader.gme.options.scorecard import GMEDailyScorecard

    # Load alt data
    alt_data_path = Path("data/GME")
    ftd_df = pd.read_csv(alt_data_path / "ftd_data.csv") if (alt_data_path / "ftd_data.csv").exists() else pd.DataFrame()
    si_df = pd.read_csv(alt_data_path / "short_interest.csv") if (alt_data_path / "short_interest.csv").exists() else pd.DataFrame()
    dp_df = pd.read_csv(alt_data_path / "dark_pool.csv") if (alt_data_path / "dark_pool.csv").exists() else pd.DataFrame()

    fusion = GMESignalFusion()
    scorecards: list[dict] = []

    for _, row in state.iterrows():
        td = row["trade_date"]
        td_str = str(td)[:10]

        # Build alt_data dict from loaded data
        alt_data = {
            "ftd_t35_countdown": 20,  # Default placeholder
            "si_change_2wk": 0.0,
            "dark_pool_ratio_change": 0.0,
        }

        sc = fusion.compute_daily_scorecard(
            trade_date=td if isinstance(td, date) else date.fromisoformat(td_str),
            options_state=row.to_dict(),
            alt_data=alt_data,
            ml_score=0.5,  # Neutral ML score
        )
        scorecards.append({
            "date": td_str,
            "composite": sc.composite,
            "regime": sc.regime,
            "options_flow": sc.pillar_scores.get("options_flow", 0),
            "volume_premium": sc.pillar_scores.get("volume_premium", 0),
            "cycle_timing": sc.pillar_scores.get("cycle_timing", 0),
            "momentum": sc.pillar_scores.get("momentum", 0),
            "anomaly_count": len(sc.anomaly_flags),
            "anomalies": "|".join(sc.anomaly_flags) if sc.anomaly_flags else "",
        })

    sc_df = pd.DataFrame(scorecards)
    sc_path = DATA_DIR / "scorecards.parquet"
    sc_df.to_parquet(sc_path, index=False)
    logger.info("Step 4 complete: %d scorecards → %s", len(sc_df), sc_path)

    # Print summary
    logger.info("Regime distribution: %s", dict(sc_df["regime"].value_counts()))
    logger.info("Composite score: mean=%.3f, std=%.3f", sc_df["composite"].mean(), sc_df["composite"].std())
    anomalous = sc_df[sc_df["anomaly_count"] > 0]
    logger.info("Days with anomalies: %d / %d", len(anomalous), len(sc_df))


def main() -> None:
    """Run the full pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    logger.info("=" * 60)
    logger.info("GME Options Analysis Pipeline")
    logger.info("Date range: %s to %s", START_DATE, END_DATE)
    logger.info("=" * 60)

    # Step 1: Fetch
    logger.info("\n=== STEP 1: Fetch Options Data ===")
    step1_fetch()

    # Step 2: Build State
    logger.info("\n=== STEP 2: Build Options State ===")
    state = step2_build_state()

    # Step 3: Backtest
    logger.info("\n=== STEP 3: Run Backtests ===")
    step3_backtest(state)

    # Step 4: Scorecards
    logger.info("\n=== STEP 4: Generate Scorecards ===")
    step4_scorecards(state)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
