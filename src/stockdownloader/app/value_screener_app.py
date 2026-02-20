"""Deep value stock screener across all US-listed tickers.

Two-phase screening:
  Phase 1 — Batch-fetch coarse quote data (v7 API, ~80 requests for ~8K tickers)
             → coarse filter (market cap, EPS, P/E, price) → ~200-500 survivors.
  Phase 2 — Fetch detailed financials per survivor (v10 API)
             → multi-factor scoring (valuation, growth, quality, income) → ranked list.

Scoring methodology:
  Valuation (35%): P/E, Earnings Yield, PEG, EV/EBITDA, P/B, P/S, Graham MoS
  Quality   (25%): Piotroski F-Score, ROE, ROA, Op Margin, Profit Margin, D/E, CR
  Growth    (20%): EPS growth, forward P/E improvement, revenue/earnings growth
  Income    (20%): Dividend yield, FCF yield, payout ratio, earnings yield

Usage::

    python -m stockdownloader.app.value_screener_app [--top N] [--min-cap NUM]
    python -m stockdownloader.app.value_screener_app --skip-detailed --top 20
    python -m stockdownloader.app.value_screener_app --sector Technology --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.analysis.value_screener import ValueScreener
from stockdownloader.data.data_parsers import format_market_cap
from stockdownloader.data.morningstar_client import MorningstarClient
from stockdownloader.data.stock_list_downloader import StockListDownloader
from stockdownloader.data.yahoo_base_client import YahooAuthHelper
from stockdownloader.data.yahoo_finance_client import YahooFinanceClient
from stockdownloader.model.financial_models import ValueScreenerResult

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")

# ---------------------------------------------------------------------------
# Thread-local client for parallel Phase 2 downloads
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _get_thread_client() -> MorningstarClient:
    """Return a per-thread :class:`MorningstarClient` with its own session.

    Each thread gets its own :class:`YahooAuthHelper` (and thus its own
    ``requests.Session`` + cookie/crumb pair) to avoid the thread-safety
    issues inherent in sharing a single ``requests.Session``.
    """
    if not hasattr(_thread_local, "client"):
        auth = YahooAuthHelper()
        auth.authenticate()
        _thread_local.client = MorningstarClient(auth=auth)
    return _thread_local.client


def _download_one_detailed(symbol: str) -> tuple[str, object | None]:
    """Download detailed financials for *symbol* in a worker thread.

    Returns ``(symbol, DetailedFinancialData)`` on success, or
    ``(symbol, None)`` on failure / incomplete data.
    """
    try:
        client = _get_thread_client()
        data = client.download_detailed(symbol)
        return (symbol, data if not data.incomplete else None)
    except (OSError, ConnectionError, TimeoutError, ValueError) as exc:
        import logging
        logging.getLogger(__name__).debug(
            "Failed to download detailed data for %s: %s", symbol, exc,
        )
        return (symbol, None)


def _q(d: Decimal) -> str:
    """Quantize a Decimal to 2 decimal places for display."""
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _pct(d: Decimal) -> str:
    """Format a Decimal as a percentage string."""
    return f"{float(d) * 100:.1f}%"


def _na(d: Decimal) -> str:
    """Format a Decimal or show N/A if zero."""
    if d == _ZERO or d is None:
        return "N/A"
    return _q(d)


# ------------------------------------------------------------------
# Executive Summary
# ------------------------------------------------------------------


def _print_executive_summary(
    results: list[ValueScreenerResult],
    total_screened: int,
    survivors: int,
) -> None:
    """Print executive summary section."""
    print()
    print("\u2554" + "\u2550" * 78 + "\u2557")
    print(
        "\u2551" + "  EXECUTIVE SUMMARY".ljust(78) + "\u2551"
    )
    print("\u255a" + "\u2550" * 78 + "\u255d")
    print()
    print(f"  Total tickers screened:      {total_screened:>8,}")
    print(f"  Passed coarse filter:        {survivors:>8,}")
    print(f"  Scored with detailed data:   {len(results):>8,}")

    if results:
        print()
        top = results[0]
        avg_score = sum(r.composite_score for r in results) / len(results)
        median_idx = len(results) // 2
        print(f"  Top scorer:     {top.symbol:<8} ({top.composite_score:.1f}/100)")
        print(f"  Average score:  {avg_score:.1f}")
        print(f"  Median score:   {results[median_idx].composite_score:.1f}")

        # Score distribution
        high = sum(1 for r in results if r.composite_score >= 70)
        mid = sum(1 for r in results if 40 <= r.composite_score < 70)
        low = sum(1 for r in results if r.composite_score < 40)
        print()
        print(f"  Score distribution:")
        print(f"    High (\u226570):  {high:>5,}  {'#' * min(high, 50)}")
        print(f"    Mid (40-69): {mid:>5,}  {'#' * min(mid // 10, 50)}")
        print(f"    Low (<40):   {low:>5,}  {'#' * min(low // 10, 50)}")

    print()


# ------------------------------------------------------------------
# Ranked Results Table
# ------------------------------------------------------------------


def _print_results(results: list[ValueScreenerResult], top_n: int) -> None:
    """Print the top-N results as a formatted table."""
    shown = results[:top_n]

    print()
    print("\u2554" + "\u2550" * 120 + "\u2557")
    print(
        "\u2551"
        + "  DEEP VALUE STOCK SCREENER \u2014 TOP RANKED RESULTS".ljust(120)
        + "\u2551"
    )
    print("\u255a" + "\u2550" * 120 + "\u255d")
    print()

    # Header
    print(
        f"{'Rank':<5} {'Symbol':<8} {'Price':>8} {'P/E':>7} {'PEG':>6} "
        f"{'EV/EB':>6} {'P/B':>6} {'Div%':>6} {'E-Yld':>6} "
        f"{'F-Scr':>5} {'MoS':>7} {'Score':>7}"
    )
    print("-" * 120)

    for i, r in enumerate(shown, 1):
        peg_str = _q(r.peg_ratio) if r.peg_ratio > _ZERO else "N/A"
        eveb_str = _q(r.ev_ebitda) if r.ev_ebitda > _ZERO else "N/A"
        print(
            f"{i:<5} {r.symbol:<8} ${_q(r.price):>7} "
            f"{_q(r.trailing_pe):>7} {peg_str:>6} {eveb_str:>6} "
            f"{_q(r.price_to_book):>6} {_pct(r.dividend_yield):>6} "
            f"{_pct(r.earnings_yield):>6} "
            f"{r.piotroski_score:>3}/9 {_pct(r.margin_of_safety):>7} "
            f"{r.composite_score:>7.1f}"
        )

    print()
    print(f"  Showing top {len(shown)} of {len(results)} scored stocks")

    # Sub-scores for top 10
    if len(shown) > 0:
        print()
        print("  Score breakdown (top 10):")
        print(
            f"  {'Symbol':<8} {'Valuation':>10} {'Growth':>8} "
            f"{'Quality':>9} {'Income':>8} {'Composite':>10}"
        )
        print("  " + "-" * 55)
        for r in shown[:10]:
            print(
                f"  {r.symbol:<8} {r.valuation_score:>10.1f} "
                f"{r.growth_score:>8.1f} {r.quality_score:>9.1f} "
                f"{r.income_score:>8.1f} {r.composite_score:>10.1f}"
            )

    print()


# ------------------------------------------------------------------
# Per-Stock Detail Cards
# ------------------------------------------------------------------


def _compute_risk_flags(r: ValueScreenerResult) -> list[str]:
    """Identify risk flags for a stock result."""
    flags: list[str] = []

    if float(r.beta) > 1.5:
        flags.append(f"HIGH BETA ({_q(r.beta)})")

    de = float(r.debt_to_equity)
    if de > 1.5:
        flags.append(f"HIGH DEBT (D/E={_q(r.debt_to_equity)})")

    cr = float(r.current_ratio)
    if 0 < cr < 1.0:
        flags.append(f"LOW LIQUIDITY (CR={_q(r.current_ratio)})")

    if r.free_cash_flow < 0:
        flags.append("NEGATIVE FCF")

    if float(r.payout_ratio) > 0.80:
        flags.append(f"HIGH PAYOUT ({_pct(r.payout_ratio)})")

    if r.piotroski_score <= 3 and r.piotroski_score > 0:
        flags.append(f"LOW F-SCORE ({r.piotroski_score}/9)")

    if float(r.margin_of_safety) < -0.30:
        flags.append("ABOVE GRAHAM NUMBER")

    # Data quality warnings
    if float(r.dividend_yield) >= 0.30:
        flags.append("DIVIDEND YIELD CLAMPED (≥30%)")
    if float(r.fcf_yield) >= 1.0 or float(r.fcf_yield) <= -1.0:
        flags.append("FCF YIELD CLAMPED")

    return flags


def _print_score_bars(r: ValueScreenerResult) -> None:
    """Print visual bar chart for each dimension score."""
    for label, sc in [
        ("Valuation", r.valuation_score),
        ("Growth   ", r.growth_score),
        ("Quality  ", r.quality_score),
        ("Income   ", r.income_score),
        ("COMPOSITE", r.composite_score),
    ]:
        bar_len = int(sc / 100.0 * 40)
        bar = "\u2588" * bar_len + "\u2591" * (40 - bar_len)
        print(f"\u2502    {label} [{bar}] {sc:5.1f}")


def _print_detail_cards(
    results: list[ValueScreenerResult], top_n: int = 20
) -> None:
    """Print per-stock detail cards for top N results."""
    shown = min(top_n, len(results))
    if shown == 0:
        return

    print()
    print("\u2554" + "\u2550" * 78 + "\u2557")
    print(
        "\u2551"
        + f"  DETAILED ANALYSIS \u2014 TOP {shown} STOCKS".ljust(78)
        + "\u2551"
    )
    print("\u255a" + "\u2550" * 78 + "\u255d")

    for i, r in enumerate(results[:top_n], 1):
        print()
        print("\u250c" + "\u2500" * 78 + "\u2510")
        mcap_str = format_market_cap(r.market_cap)
        beta_str = _na(r.beta)
        header = f" #{i:<3} {r.symbol:<10} Price: ${_q(r.price):<10} MktCap: {mcap_str:<12} Beta: {beta_str}"
        print(f"\u2502{header:<78}\u2502")
        print("\u251c" + "\u2500" * 78 + "\u2524")

        # Valuation
        line = f"  VALUATION ({r.valuation_score:.1f}/100)"
        print(f"\u2502{line:<78}\u2502")
        line = (
            f"    P/E: {_q(r.trailing_pe):<8} Fwd P/E: {_q(r.forward_pe):<8} "
            f"PEG: {_na(r.peg_ratio):<8} EV/EBITDA: {_na(r.ev_ebitda)}"
        )
        print(f"\u2502{line:<78}\u2502")
        line = (
            f"    P/B: {_q(r.price_to_book):<8} P/S: {_q(r.price_to_sales):<8} "
            f"Graham#: ${_q(r.graham_number):<7} MoS: {_pct(r.margin_of_safety)}"
        )
        print(f"\u2502{line:<78}\u2502")
        line = f"    Earnings Yield: {_pct(r.earnings_yield)}"
        print(f"\u2502{line:<78}\u2502")

        # Quality
        print("\u251c" + "\u2500" * 78 + "\u2524")
        line = f"  QUALITY ({r.quality_score:.1f}/100)"
        print(f"\u2502{line:<78}\u2502")
        line = (
            f"    Piotroski F-Score: {r.piotroski_score}/9    "
            f"ROE: {_pct(r.return_on_equity):<8} ROA: {_pct(r.return_on_assets)}"
        )
        print(f"\u2502{line:<78}\u2502")
        line = (
            f"    Op Margin: {_pct(r.operating_margin):<8} "
            f"D/E: {_q(r.debt_to_equity):<8} "
            f"Current Ratio: {_q(r.current_ratio)}"
        )
        print(f"\u2502{line:<78}\u2502")

        # Growth
        print("\u251c" + "\u2500" * 78 + "\u2524")
        line = f"  GROWTH ({r.growth_score:.1f}/100)"
        print(f"\u2502{line:<78}\u2502")
        line = (
            f"    Revenue Growth: {_pct(r.revenue_growth):<10} "
            f"Earnings Growth: {_pct(r.earnings_growth)}"
        )
        print(f"\u2502{line:<78}\u2502")

        # Income
        print("\u251c" + "\u2500" * 78 + "\u2524")
        line = f"  INCOME ({r.income_score:.1f}/100)"
        print(f"\u2502{line:<78}\u2502")
        line = (
            f"    Div Yield: {_pct(r.dividend_yield):<8} "
            f"FCF Yield: {_pct(r.fcf_yield):<8} "
            f"Payout Ratio: {_pct(r.payout_ratio)}"
        )
        print(f"\u2502{line:<78}\u2502")

        # Risk flags
        flags = _compute_risk_flags(r)
        if flags:
            print("\u251c" + "\u2500" * 78 + "\u2524")
            line = "  \u26a0 RISK FLAGS:"
            print(f"\u2502{line:<78}\u2502")
            for flag in flags:
                line = f"    - {flag}"
                print(f"\u2502{line:<78}\u2502")

        # Score bars
        print("\u251c" + "\u2500" * 78 + "\u2524")
        _print_score_bars(r)
        print("\u2514" + "\u2500" * 78 + "\u2518")

    print()


# ------------------------------------------------------------------
# CSV Export
# ------------------------------------------------------------------


def _write_csv(
    results: list[ValueScreenerResult], output_path: str
) -> None:
    """Write results to a CSV file with all metrics."""
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Rank", "Symbol", "Price", "Market Cap",
            "Trailing P/E", "Forward P/E", "PEG Ratio", "EV/EBITDA",
            "P/B", "P/S", "Earnings Yield", "FCF Yield",
            "Dividend Yield", "Graham Number", "Margin of Safety",
            "Piotroski F-Score", "ROE", "ROA",
            "Operating Margin", "D/E", "Current Ratio",
            "Revenue Growth", "Earnings Growth",
            "Beta", "Payout Ratio", "Enterprise Value", "FCF",
            "Valuation Score", "Growth Score", "Quality Score",
            "Income Score", "Composite Score",
        ])
        for i, r in enumerate(results, 1):
            writer.writerow([
                i, r.symbol, _q(r.price), r.market_cap,
                _q(r.trailing_pe), _q(r.forward_pe),
                _q(r.peg_ratio), _q(r.ev_ebitda),
                _q(r.price_to_book), _q(r.price_to_sales),
                _pct(r.earnings_yield), _pct(r.fcf_yield),
                _pct(r.dividend_yield), _q(r.graham_number),
                _pct(r.margin_of_safety),
                r.piotroski_score, _pct(r.return_on_equity),
                _pct(r.return_on_assets),
                _pct(r.operating_margin), _q(r.debt_to_equity),
                _q(r.current_ratio),
                _pct(r.revenue_growth), _pct(r.earnings_growth),
                _q(r.beta), _pct(r.payout_ratio),
                r.enterprise_value, r.free_cash_flow,
                r.valuation_score, r.growth_score,
                r.quality_score, r.income_score,
                r.composite_score,
            ])

    print(f"  Results written to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Entry point for the deep value screener."""
    parser = argparse.ArgumentParser(
        description="Deep Value Stock Screener — find undervalued stocks across all US exchanges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  value_screener_app --top 20                   # Top 20 deep value stocks\n"
            "  value_screener_app --skip-detailed --top 50   # Fast coarse screen\n"
            "  value_screener_app --sector Technology         # Tech sector only\n"
            "  value_screener_app --output results.csv        # Export to CSV\n"
        ),
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Number of top results to display (default: 50)",
    )
    parser.add_argument(
        "--min-cap", type=int, default=100_000_000,
        help="Minimum market capitalization in dollars (default: 100000000)",
    )
    parser.add_argument(
        "--sector", type=str, default=None,
        help="Filter to specific sector (e.g. Technology, Healthcare)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output file path",
    )
    parser.add_argument(
        "--skip-detailed", action="store_true",
        help="Phase 1 only (coarse scores, much faster)",
    )
    parser.add_argument(
        "--detail-cards", type=int, default=20,
        help="Number of per-stock detail cards to print (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel download threads for Phase 2 (default: 8)",
    )
    args = parser.parse_args()

    print("\u2554" + "\u2550" * 66 + "\u2557")
    print("\u2551" + "              DEEP VALUE STOCK SCREENER                           " + "\u2551")
    print("\u2551" + "  Industry-Standard Valuation | Piotroski F-Score | Graham MoS   " + "\u2551")
    print("\u255a" + "\u2550" * 66 + "\u255d")
    print()

    # === Step 1: Download ticker lists ===
    print("Downloading ticker lists...")
    sd = StockListDownloader()
    sd.download_nasdaq()
    sd.download_others()

    all_tickers: set[str] = set()
    all_tickers.update(sd.nasdaq_list)
    all_tickers.update(sd.others_list)
    total_screened = len(all_tickers)

    # Sector filter using Nasdaq API metadata
    if args.sector:
        metadata = sd.ticker_metadata
        filtered: set[str] = set()
        for sym in all_tickers:
            meta = metadata.get(sym, {})
            if meta.get("sector", "").lower() == args.sector.lower():
                filtered.add(sym)
        print(f"  Sector filter '{args.sector}': {len(all_tickers)} \u2192 {len(filtered)} tickers")
        all_tickers = filtered

    sorted_tickers = sorted(all_tickers)
    print(f"  Total tickers to screen: {len(sorted_tickers)}")
    print()

    # === Step 2: Authenticate and create clients ===
    auth = YahooAuthHelper()
    quote_client = YahooFinanceClient(auth=auth)
    financial_client = MorningstarClient(auth=auth)
    screener = ValueScreener()

    # === Step 3: Phase 1 — Batch download quotes ===
    print("Phase 1: Batch downloading quote data...")
    t0 = time.time()
    quotes = quote_client.download_batch(sorted_tickers)
    t1 = time.time()
    print(f"  Downloaded {len(quotes)} quotes in {t1 - t0:.1f}s")

    # === Step 4: Coarse filter ===
    survivors = screener.coarse_filter(quotes, min_market_cap=args.min_cap)
    num_survivors = len(survivors)
    print(f"  Coarse filter: {len(quotes)} \u2192 {num_survivors} candidates")
    print()

    if not survivors:
        print("No stocks passed the coarse filter. Try lowering --min-cap.")
        return

    # === Step 5: Phase 2 — Detailed financials (optional) ===
    detailed_map: dict[str, object] = {}

    if not args.skip_detailed:
        max_workers = min(args.workers, num_survivors)
        print(
            f"Phase 2: Downloading detailed financials for {num_survivors} "
            f"stocks ({max_workers} threads)..."
        )
        t0 = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_one_detailed, sym): sym
                for sym in sorted(survivors)
            }
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0 or completed == num_survivors:
                    print(f"  Progress: {completed}/{num_survivors}")
                sym, data = future.result()
                if data is not None:
                    detailed_map[sym] = data

        t1 = time.time()
        print(
            f"  Downloaded {len(detailed_map)} detailed financials in {t1 - t0:.1f}s"
        )
        print()

    # === Step 6: Score all survivors ===
    print("Scoring stocks...")
    results: list[ValueScreenerResult] = []

    for sym in survivors:
        quote = quotes[sym]
        det = detailed_map.get(sym)
        try:
            result = screener.score(sym, quote, det)
            results.append(result)
        except Exception as exc:
            logger.debug("Failed scoring for %s: %s", sym, exc)

    # Sort by composite score descending
    results.sort(key=lambda r: r.composite_score, reverse=True)
    print(f"  Scored {len(results)} stocks")

    # === Step 7: Output ===
    _print_executive_summary(results, total_screened, num_survivors)
    _print_results(results, args.top)
    _print_detail_cards(results, args.detail_cards)

    if args.output:
        _write_csv(results, args.output)

    print("  DISCLAIMER: This is for educational purposes only.")
    print("  Not financial advice. Past performance does not guarantee future results.")
    print("  Always do your own research before investing.")
    print()


if __name__ == "__main__":
    main()
