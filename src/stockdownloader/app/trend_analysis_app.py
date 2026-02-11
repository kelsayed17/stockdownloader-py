"""Main entry point for stock trend pattern analysis.

Downloads ticker lists, fetches historical data, and analyzes price movement patterns.

Usage:
    python -m stockdownloader.app.trend_analysis_app
"""
from __future__ import annotations

import logging
from collections import defaultdict

from stockdownloader.analysis.pattern_analyzer import analyze, print_results
from stockdownloader.data.stock_list_downloader import StockListDownloader
from stockdownloader.data.yahoo_historical_client import YahooHistoricalClient
from stockdownloader.util.date_helper import DateHelper

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the stock universe pattern scanner."""
    stock_list: set[str] = set()

    dates = DateHelper()
    sd = StockListDownloader()
    historical_client = YahooHistoricalClient()

    print("Downloading lists, please wait...")

    sd.download_nasdaq()
    sd.download_others()
    sd.download_zacks()
    sd.download_yahoo_earnings(dates.today_market)
    sd.download_yahoo_earnings(dates.tomorrow_market)
    sd.download_yahoo_earnings(dates.yesterday_market)
    sd.read_incomplete()

    stock_list.update(sd.nasdaq_list)
    stock_list.update(sd.others_list)
    stock_list -= sd.incomplete_list

    # Sort for deterministic iteration order
    sorted_stock_list = sorted(stock_list)

    print(f"Nasdaq stocks downloaded: {len(sd.nasdaq_list)}")
    print(f"Other stocks downloaded: {len(sd.others_list)}")
    print(f"Mutual funds downloaded: {len(sd.mfunds_list)}")
    print(f"Zacks stocks downloaded: {len(sd.zacks_list)}")
    print(f"Earnings stocks downloaded: {len(sd.earnings_list)}")
    print(f"Stocks with incomplete data: {len(sd.incomplete_list)}")
    print(f"Total stocks to be processed: {len(sorted_stock_list)}")
    print()

    patterns: dict[str, set[str]] = defaultdict(set)
    count = 1

    for ticker in sorted_stock_list:
        data = historical_client.download(ticker)

        if data.incomplete:
            sd.append_incomplete(ticker)

        # Merge patterns
        for pattern_key, symbols in data.patterns.items():
            patterns[pattern_key].update(symbols)

        if count == 100:
            for pattern_key in patterns:
                print(f"{pattern_key}\t{patterns[pattern_key]}")

        count += 1

    results = analyze(patterns)
    print_results(results)


if __name__ == "__main__":
    main()
