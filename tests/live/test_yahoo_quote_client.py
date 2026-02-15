"""Live tests for YahooQuoteClient (page scraping + crumb + CSV download)."""
from __future__ import annotations

import time

import pytest

from stockdownloader.data.yahoo_quote_client import YahooQuoteClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def quote_client(yahoo_auth) -> YahooQuoteClient:
    return YahooQuoteClient(auth=yahoo_auth)


class TestYahooQuoteClient:

    def test_get_page_returns_html(self, quote_client):
        html = quote_client.get_page("SPY")
        assert html is not None
        assert len(html) > 1000

    def test_get_page_contains_spy_reference(self, quote_client):
        html = quote_client.get_page("SPY")
        assert html is not None
        assert "SPY" in html

    def test_get_crumb_returns_non_empty_string(self, quote_client):
        crumb = quote_client.get_crumb("SPY")
        assert isinstance(crumb, str)
        assert len(crumb) > 0

    def test_download_data_does_not_raise(self, quote_client, tmp_path, monkeypatch):
        """The v7 download endpoint may return 401/403 depending on auth state.

        We verify the method runs without exception. If it succeeds, the
        CSV file should contain valid header and data rows.
        """
        monkeypatch.chdir(tmp_path)
        crumb = quote_client.get_crumb("SPY")
        now = int(time.time())
        one_month_ago = now - (30 * 86400)
        quote_client.download_data("SPY", one_month_ago, now, crumb)
        csv_file = tmp_path / "SPY.csv"
        if csv_file.exists():
            content = csv_file.read_text()
            assert "Date" in content or "Open" in content
            lines = content.strip().split("\n")
            assert len(lines) > 1
