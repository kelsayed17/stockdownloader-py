"""CLI entry point for the Intelligent Signal Advisory System.

Supports one-shot analysis and continuous monitoring with JSON output.

Usage::

    # One-shot (default)
    signal-monitor SPY
    signal-monitor SPY AAPL MSFT --json-output output/signals.json

    # Continuous monitoring every 5 minutes
    signal-monitor SPY --monitor --interval 300

    # Filter by confidence
    signal-monitor SPY --min-confidence 0.6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from stockdownloader.analysis.alert_store import AlertStore
from stockdownloader.analysis.signal_advisor import AdvisorConfig, SignalAdvisor
from stockdownloader.app.helpers import fetch_daily_data
from stockdownloader.core.models.signal import AdvisoryAction, SignalAdvisory
from stockdownloader.core.config import DEFAULT_ALERT_HISTORY

logger = logging.getLogger(__name__)

_DEFAULT_SYMBOLS = ["SPY"]
_DEFAULT_INTERVAL = 300  # seconds
_DEFAULT_MIN_CONFIDENCE = 0.0


# ------------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------------


def _analyze_symbol(
    symbol: str,
    advisor: SignalAdvisor,
    store: AlertStore,
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    json_output: Path | None = None,
    print_fn: object = print,
) -> SignalAdvisory | None:
    """Fetch data, evaluate, persist, and optionally print.

    Returns the :class:`SignalAdvisory` if new and above threshold,
    otherwise ``None``.
    """
    _print = print_fn  # type: ignore[assignment]

    data = fetch_daily_data(symbol, period="5y")
    if not data:
        _print(f"  [{symbol}] No data available — skipping.")  # type: ignore[operator]
        return None

    advisory = advisor.evaluate(symbol, data)
    is_new = store.save(advisory)

    if not is_new:
        logger.debug("%s: duplicate advisory — skipping", symbol)
        return None

    if advisory.confidence < min_confidence:
        logger.debug(
            "%s: confidence %.2f below threshold %.2f",
            symbol, advisory.confidence, min_confidence,
        )
        return None

    _print_advisory(advisory, print_fn=_print)  # type: ignore[arg-type]

    if json_output:
        _append_json(advisory, json_output)

    return advisory


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------


def _print_advisory(
    advisory: SignalAdvisory,
    *,
    print_fn: object = print,
) -> None:
    """Human-readable console output for a single advisory."""
    p = print_fn  # type: ignore[assignment]

    p("")  # type: ignore[operator]
    p("=" * 70)  # type: ignore[operator]
    p(f"  SIGNAL ADVISORY: {advisory.symbol}")  # type: ignore[operator]
    p("=" * 70)  # type: ignore[operator]
    p(f"  Timestamp:       {advisory.timestamp}")  # type: ignore[operator]
    p(f"  Action:          {advisory.action.value}")  # type: ignore[operator]
    p(f"  Confidence:      {advisory.confidence:.0%}")  # type: ignore[operator]
    p(f"  Regime:          {advisory.regime} ({advisory.regime_confidence:.0%})")  # type: ignore[operator]
    p("")  # type: ignore[operator]
    p(f"  Entry:           ${advisory.entry_price:,.2f}")  # type: ignore[operator]
    p(f"  Stop Loss:       ${advisory.stop_loss:,.2f}")  # type: ignore[operator]
    p(f"  Take Profit:     ${advisory.take_profit:,.2f}")  # type: ignore[operator]
    p(f"  Risk/Reward:     {advisory.risk_reward:.2f}")  # type: ignore[operator]
    p(f"  Position Size:   {advisory.position_size_pct:.1f}%")  # type: ignore[operator]
    p("")  # type: ignore[operator]

    r = advisory.reasoning
    p("-" * 70)  # type: ignore[operator]
    p(f"  Confluence:      {r.signal_confluence}")  # type: ignore[operator]
    p(f"  Regime:          {r.regime_alignment}")  # type: ignore[operator]
    p(f"  Walk-Forward:    {r.walk_forward_validated}")  # type: ignore[operator]

    if r.key_bullish:
        p("")  # type: ignore[operator]
        p("  Bullish:")  # type: ignore[operator]
        for b in r.key_bullish:
            p(f"    + {b}")  # type: ignore[operator]
    if r.key_bearish:
        p("")  # type: ignore[operator]
        p("  Bearish:")  # type: ignore[operator]
        for b in r.key_bearish:
            p(f"    - {b}")  # type: ignore[operator]

    if advisory.call_advisory:
        ca = advisory.call_advisory
        p("")  # type: ignore[operator]
        p(f"  Call: {ca.action} ${ca.strike:.0f} strike, {ca.dte}DTE, delta {ca.delta:.2f}, ~${ca.est_premium:.2f}")  # type: ignore[operator]
    if advisory.put_advisory:
        pa = advisory.put_advisory
        p(f"  Put:  {pa.action} ${pa.strike:.0f} strike, {pa.dte}DTE, delta {pa.delta:.2f}, ~${pa.est_premium:.2f}")  # type: ignore[operator]

    p("")  # type: ignore[operator]
    p("=" * 70)  # type: ignore[operator]
    p("  DISCLAIMER: For educational purposes only. Not financial advice.")  # type: ignore[operator]
    p("=" * 70)  # type: ignore[operator]
    p("")  # type: ignore[operator]


# ------------------------------------------------------------------
# JSON output
# ------------------------------------------------------------------


def _append_json(advisory: SignalAdvisory, filepath: Path) -> None:
    """Append advisory dict to a JSON array file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if filepath.exists():
        try:
            existing = json.loads(filepath.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(advisory.to_dict())
    filepath.write_text(
        json.dumps(existing, indent=2), encoding="utf-8",
    )


# ------------------------------------------------------------------
# Run modes
# ------------------------------------------------------------------


def _run_once(
    symbols: list[str],
    advisor: SignalAdvisor,
    store: AlertStore,
    *,
    min_confidence: float,
    json_output: Path | None,
) -> list[SignalAdvisory]:
    """Single analysis pass over all symbols."""
    results: list[SignalAdvisory] = []
    for sym in symbols:
        adv = _analyze_symbol(
            sym, advisor, store,
            min_confidence=min_confidence,
            json_output=json_output,
        )
        if adv is not None:
            results.append(adv)
    return results


def _run_monitor(
    symbols: list[str],
    advisor: SignalAdvisor,
    store: AlertStore,
    *,
    interval: int,
    min_confidence: float,
    json_output: Path | None,
) -> None:  # pragma: no cover
    """Continuous monitoring loop.  Runs until ``KeyboardInterrupt``."""
    print(f"Monitoring {', '.join(symbols)} every {interval}s (Ctrl+C to stop)")
    print()

    while True:
        try:
            _run_once(
                symbols, advisor, store,
                min_confidence=min_confidence,
                json_output=json_output,
            )
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-monitor",
        description="Intelligent Signal Advisory System — regime-aware buy/sell signals.",
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        default=_DEFAULT_SYMBOLS,
        help="Ticker symbols to analyze (default: SPY)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run in continuous monitoring mode",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=_DEFAULT_INTERVAL,
        help=f"Seconds between monitoring cycles (default: {_DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=_DEFAULT_MIN_CONFIDENCE,
        help="Minimum confidence to display (default: 0.0)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON output file",
    )
    parser.add_argument(
        "--alert-store",
        type=str,
        default=str(DEFAULT_ALERT_HISTORY),
        help="Path to alert history JSON file",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        default=None,
        help="Path to trained ML model (.joblib) for confidence boosting",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = AdvisorConfig(ml_model_path=args.ml_model)
    advisor = SignalAdvisor(config=cfg)
    store = AlertStore(filepath=args.alert_store)
    json_out = Path(args.json_output) if args.json_output else None

    if args.monitor:
        _run_monitor(
            args.symbols, advisor, store,
            interval=args.interval,
            min_confidence=args.min_confidence,
            json_output=json_out,
        )
    else:
        results = _run_once(
            args.symbols, advisor, store,
            min_confidence=args.min_confidence,
            json_output=json_out,
        )
        if not results:
            print("No new signals above threshold.")


if __name__ == "__main__":
    main()
