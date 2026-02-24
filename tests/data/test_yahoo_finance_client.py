"""Unit tests for YahooFinanceClient — batch download and parse logic."""
from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.yahoo_finance_client import YahooFinanceClient
from stockdownloader.core.models.financial import QuoteData


def _mock_auth() -> MagicMock:
    auth = MagicMock()
    auth.crumb = "test_crumb"
    auth.session = MagicMock()
    return auth


def _make_quote_response(*quotes: dict) -> str:
    """Build a Yahoo v7 quote response JSON string."""
    return json.dumps({"quoteResponse": {"result": list(quotes)}})


_AAPL_QUOTE = {
    "symbol": "AAPL",
    "regularMarketPrice": 195.5,
    "epsTrailingTwelveMonths": 6.42,
    "epsForward": 7.10,
    "trailingPE": 30.45,
    "forwardPE": 27.54,
    "priceToBook": 48.2,
    "bookValue": 4.05,
    "priceToSalesTrailing12Months": 7.8,
    "trailingAnnualDividendYield": 0.005,
    "fiftyTwoWeekHigh": 199.62,
    "fiftyTwoWeekLow": 164.08,
    "fiftyDayAverage": 188.0,
    "twoHundredDayAverage": 182.0,
    "regularMarketPreviousClose": 194.0,
    "regularMarketOpen": 195.0,
    "regularMarketDayHigh": 196.0,
    "regularMarketDayLow": 194.5,
    "regularMarketVolume": 45000000,
    "fiftyTwoWeekRange": "164.08 - 199.62",
    "marketCap": 3000000000000,
}

_GOOGL_QUOTE = {
    "symbol": "GOOGL",
    "regularMarketPrice": 140.5,
    "epsTrailingTwelveMonths": 5.80,
    "trailingPE": 24.22,
    "forwardPE": 21.0,
    "priceToBook": 6.5,
    "bookValue": 21.6,
    "marketCap": 1700000000000,
}

_MSFT_QUOTE = {
    "symbol": "MSFT",
    "regularMarketPrice": 420.0,
    "epsTrailingTwelveMonths": 11.5,
    "trailingPE": 36.52,
    "forwardPE": 32.0,
    "priceToBook": 13.0,
    "bookValue": 32.3,
    "marketCap": 3100000000000,
}


class TestParseSingleQuote:
    """Tests for _parse_single_quote static method."""

    def test_parses_all_fields(self) -> None:
        data = YahooFinanceClient._parse_single_quote(_AAPL_QUOTE)

        assert data.last_trade_price_only == Decimal("195.5")
        assert data.diluted_eps == Decimal("6.42")
        assert data.trailing_pe == Decimal("30.45")
        assert data.forward_pe == Decimal("27.54")
        assert data.price_to_book == Decimal("48.2")
        assert data.book_value == Decimal("4.05")
        assert data.price_sales == Decimal("7.8")
        assert data.market_capitalization == 3000000000000
        assert data.year_high == Decimal("199.62")
        assert data.year_low == Decimal("164.08")

    def test_handles_missing_fields(self) -> None:
        data = YahooFinanceClient._parse_single_quote({"symbol": "TEST"})
        assert data.trailing_pe == Decimal("0")
        assert data.forward_pe == Decimal("0")
        assert data.price_to_book == Decimal("0")
        assert data.book_value == Decimal("0")


class TestDownloadBatch:
    """Tests for download_batch method."""

    def test_batch_three_tickers(self) -> None:
        auth = _mock_auth()
        client = YahooFinanceClient(auth=auth)

        resp = MagicMock()
        resp.text = _make_quote_response(_AAPL_QUOTE, _GOOGL_QUOTE, _MSFT_QUOTE)
        auth.session.get.return_value = resp

        results = client.download_batch(["AAPL", "GOOGL", "MSFT"])

        assert len(results) == 3
        assert "AAPL" in results
        assert "GOOGL" in results
        assert "MSFT" in results
        assert results["AAPL"].trailing_pe == Decimal("30.45")
        assert results["GOOGL"].trailing_pe == Decimal("24.22")
        assert results["MSFT"].trailing_pe == Decimal("36.52")

    def test_partial_response(self) -> None:
        """Some tickers may be missing from the response."""
        auth = _mock_auth()
        client = YahooFinanceClient(auth=auth)

        # Response only has AAPL, not GOOGL
        resp = MagicMock()
        resp.text = _make_quote_response(_AAPL_QUOTE)
        auth.session.get.return_value = resp

        results = client.download_batch(["AAPL", "GOOGL"])

        assert len(results) == 1
        assert "AAPL" in results
        assert "GOOGL" not in results

    def test_empty_batch(self) -> None:
        auth = _mock_auth()
        client = YahooFinanceClient(auth=auth)

        results = client.download_batch([])
        assert results == {}
        # Should not make any API calls
        auth.session.get.assert_not_called()

    def test_empty_response(self) -> None:
        auth = _mock_auth()
        client = YahooFinanceClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps({"quoteResponse": {"result": []}})
        auth.session.get.return_value = resp

        results = client.download_batch(["AAPL"])
        assert results == {}
