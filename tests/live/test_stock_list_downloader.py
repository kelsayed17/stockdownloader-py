"""Live tests for StockListDownloader (NASDAQ API, NYSE/AMEX, Yahoo Earnings).

Skips Zacks (aggressive bot protection) and mutual funds (unreliable FTP).
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from stockdownloader.data.stock_list import StockListDownloader

pytestmark = pytest.mark.live


class TestStockListDownloaderNasdaq:

    def test_download_nasdaq_does_not_raise(self, tmp_path, monkeypatch):
        """NASDAQ API and FTP fallback may both be blocked by bot detection.

        We verify the method runs without exception and returns a frozenset.
        When the API is reachable, the list should contain major tickers.
        """
        monkeypatch.chdir(tmp_path)
        dl = StockListDownloader()
        dl.download_nasdaq()
        nasdaq = dl.nasdaq_list
        assert isinstance(nasdaq, frozenset)
        if len(nasdaq) > 0:
            assert "AAPL" in nasdaq or "MSFT" in nasdaq


class TestStockListDownloaderOthers:

    def test_download_others_does_not_raise(self, tmp_path, monkeypatch):
        """NYSE/AMEX API and FTP fallback may both be blocked by bot detection.

        We verify the method runs without exception and returns a frozenset.
        """
        monkeypatch.chdir(tmp_path)
        dl = StockListDownloader()
        dl.download_others()
        others = dl.others_list
        assert isinstance(others, frozenset)


class TestStockListDownloaderYahooEarnings:

    def test_download_yahoo_earnings_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dl = StockListDownloader()
        yesterday = date.today() - timedelta(days=1)
        dl.download_yahoo_earnings(yesterday)
        assert isinstance(dl.earnings_list, frozenset)
