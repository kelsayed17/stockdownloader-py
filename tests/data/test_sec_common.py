"""Unit tests for sec_common — shared SEC HTTP helpers."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from stockdownloader.data.sec_common import (
    ARCHIVE_BASE,
    EFTS_BASE_URL,
    SEC_RATE_LIMIT_DELAY,
    SplitAdjustment,
    download_with_retry,
    fetch_all_efts_hits,
    fetch_efts_page,
    fetch_url_text,
    ipo_quarter_floor,
    prev_quarter,
    quarter_iterator,
    splits_for_symbol,
)


# ------------------------------------------------------------------
# prev_quarter
# ------------------------------------------------------------------


class TestPrevQuarter:
    """prev_quarter returns the previous calendar quarter."""

    def test_q2_to_q1(self) -> None:
        assert prev_quarter(2024, 2) == (2024, 1)

    def test_q3_to_q2(self) -> None:
        assert prev_quarter(2024, 3) == (2024, 2)

    def test_q4_to_q3(self) -> None:
        assert prev_quarter(2024, 4) == (2024, 3)

    def test_q1_wraps_to_prev_year_q4(self) -> None:
        assert prev_quarter(2024, 1) == (2023, 4)

    def test_q1_wrap_year_2000(self) -> None:
        assert prev_quarter(2000, 1) == (1999, 4)


# ------------------------------------------------------------------
# download_with_retry
# ------------------------------------------------------------------


def _mock_session() -> MagicMock:
    """Create a mock requests.Session."""
    return MagicMock(spec=requests.Session)


class TestDownloadWithRetry:
    """download_with_retry handles HTTP status codes and retries."""

    def test_success_200(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"binary data"
        session.get.return_value = resp

        result = download_with_retry(session, "https://example.com/file")
        assert result == b"binary data"
        session.get.assert_called_once_with("https://example.com/file", timeout=120)

    def test_404_returns_none(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 404
        session.get.return_value = resp

        result = download_with_retry(session, "https://example.com/missing")
        assert result is None
        # Should not retry on 404
        assert session.get.call_count == 1

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_retry_on_500(self, mock_sleep: MagicMock) -> None:
        session = _mock_session()

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = b"ok"

        session.get.side_effect = [resp_500, resp_200]

        result = download_with_retry(session, "https://example.com/retry")
        assert result == b"ok"
        assert session.get.call_count == 2
        mock_sleep.assert_called_once_with(2.0)

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep: MagicMock) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 500
        session.get.return_value = resp

        result = download_with_retry(session, "https://example.com/fail")
        assert result is None
        assert session.get.call_count == 3
        assert mock_sleep.call_count == 2  # sleeps between retries

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_request_exception_triggers_retry(self, mock_sleep: MagicMock) -> None:
        session = _mock_session()
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.content = b"recovered"
        session.get.side_effect = [requests.ConnectionError("fail"), resp_ok]

        result = download_with_retry(session, "https://example.com/flaky")
        assert result == b"recovered"
        assert session.get.call_count == 2

    def test_rate_limit_fn_called(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"data"
        session.get.return_value = resp
        rate_fn = MagicMock()

        download_with_retry(session, "https://example.com/rl", rate_limit_fn=rate_fn)
        rate_fn.assert_called_once()

    def test_custom_max_retries(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 500
        session.get.return_value = resp

        with patch("stockdownloader.data.sec_common.time.sleep"):
            result = download_with_retry(
                session, "https://example.com/x", max_retries=5
            )
        assert result is None
        assert session.get.call_count == 5


# ------------------------------------------------------------------
# fetch_url_text
# ------------------------------------------------------------------


class TestFetchUrlText:
    """fetch_url_text returns decoded text or None."""

    def test_success(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html>hello</html>"
        session.get.return_value = resp

        result = fetch_url_text(session, "https://example.com/page")
        assert result == "<html>hello</html>"
        session.get.assert_called_once_with("https://example.com/page", timeout=30)

    def test_404_returns_none(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 404
        session.get.return_value = resp

        result = fetch_url_text(session, "https://example.com/missing")
        assert result is None

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_retries_on_error(self, mock_sleep: MagicMock) -> None:
        session = _mock_session()
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.text = "ok"
        session.get.side_effect = [requests.ConnectionError("err"), resp_ok]

        result = fetch_url_text(session, "https://example.com/flaky")
        assert result == "ok"
        assert session.get.call_count == 2

    def test_rate_limit_fn_called(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "text"
        session.get.return_value = resp
        rate_fn = MagicMock()

        fetch_url_text(session, "https://example.com/page", rate_limit_fn=rate_fn)
        rate_fn.assert_called_once()


# ------------------------------------------------------------------
# fetch_efts_page
# ------------------------------------------------------------------


class TestFetchEftsPage:
    """fetch_efts_page parses EFTS JSON responses."""

    def test_success_with_hits(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        hits = [{"_id": "1", "_source": {"file_num": "005-12345"}}]
        resp.json.return_value = {"hits": {"hits": hits}}
        session.get.return_value = resp

        result = fetch_efts_page(session, "https://efts.sec.gov/LATEST/search-index?q=test")
        assert result == hits

    def test_empty_response(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": {"hits": []}}
        session.get.return_value = resp

        result = fetch_efts_page(session, "https://efts.sec.gov/LATEST/search-index?q=nothing")
        assert result == []

    def test_missing_hits_key(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {}
        session.get.return_value = resp

        result = fetch_efts_page(session, "https://efts.sec.gov/LATEST/search-index?q=x")
        assert result == []

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_retry_on_500(self, mock_sleep: MagicMock) -> None:
        session = _mock_session()
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {"hits": {"hits": [{"_id": "2"}]}}
        session.get.side_effect = [resp_500, resp_ok]

        result = fetch_efts_page(session, "https://efts.sec.gov/LATEST/search-index?q=retry")
        assert result == [{"_id": "2"}]
        assert session.get.call_count == 2

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_json_decode_error_retries(self, mock_sleep: MagicMock) -> None:
        session = _mock_session()
        resp_bad = MagicMock()
        resp_bad.status_code = 200
        resp_bad.json.side_effect = json.JSONDecodeError("bad", "", 0)
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {"hits": {"hits": [{"_id": "3"}]}}
        session.get.side_effect = [resp_bad, resp_ok]

        result = fetch_efts_page(session, "https://efts.sec.gov/LATEST/search-index?q=json_err")
        assert result == [{"_id": "3"}]

    def test_rate_limit_fn_called(self) -> None:
        session = _mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": {"hits": []}}
        session.get.return_value = resp
        rate_fn = MagicMock()

        fetch_efts_page(session, "https://efts.sec.gov/LATEST/search-index?q=x", rate_limit_fn=rate_fn)
        rate_fn.assert_called_once()


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


class TestConstants:
    """Module-level constants have expected values."""

    def test_sec_rate_limit_delay(self) -> None:
        assert SEC_RATE_LIMIT_DELAY == 0.11

    def test_efts_base_url(self) -> None:
        assert EFTS_BASE_URL == "https://efts.sec.gov/LATEST/search-index"

    def test_archive_base(self) -> None:
        assert ARCHIVE_BASE == "https://www.sec.gov/Archives/edgar/data"


# ------------------------------------------------------------------
# quarter_iterator
# ------------------------------------------------------------------


class TestQuarterIterator:
    """quarter_iterator yields (year, quarter) tuples backwards."""

    @patch("stockdownloader.data.sec_common.date")
    def test_quarter_iterator_basic(self, mock_date: MagicMock) -> None:
        """Yields correct quarters in reverse from the current quarter."""
        mock_date.today.return_value = date(2025, 5, 15)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        result = list(quarter_iterator(4))
        assert result == [
            (2025, 2),
            (2025, 1),
            (2024, 4),
            (2024, 3),
        ]

    @patch("stockdownloader.data.sec_common.date")
    def test_quarter_iterator_ipo_floor(self, mock_date: MagicMock) -> None:
        """Stops at min_year/min_quarter before exhausting num_quarters."""
        mock_date.today.return_value = date(2025, 5, 15)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        result = list(quarter_iterator(
            100, min_year=2025, min_quarter=1,
        ))
        assert result == [
            (2025, 2),
            (2025, 1),
        ]

    @patch("stockdownloader.data.sec_common.date")
    def test_quarter_iterator_zero(self, mock_date: MagicMock) -> None:
        """Yields nothing for num_quarters=0."""
        mock_date.today.return_value = date(2025, 5, 15)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        result = list(quarter_iterator(0))
        assert result == []


# ------------------------------------------------------------------
# ipo_quarter_floor
# ------------------------------------------------------------------


class TestIpoQuarterFloor:
    """ipo_quarter_floor returns the earliest useful quarter."""

    @patch("stockdownloader.core.models.symbol.get_symbol_info")
    def test_ipo_quarter_floor_known_symbol(
        self, mock_get_info: MagicMock,
    ) -> None:
        """Returns IPO quarter for a known symbol (e.g. TSLA)."""
        from stockdownloader.core.models.symbol import SymbolInfo

        mock_get_info.return_value = SymbolInfo(
            symbol="TSLA",
            cusip="88160R101",
            ipo_date=date(2010, 6, 29),
            name="Tesla Inc",
        )
        result = ipo_quarter_floor("TSLA")
        assert result == (2010, 2)

    @patch("stockdownloader.core.models.symbol.get_symbol_info")
    def test_ipo_quarter_floor_unknown(
        self, mock_get_info: MagicMock,
    ) -> None:
        """Returns (2003, 1) default for unknown symbol."""
        mock_get_info.return_value = None
        result = ipo_quarter_floor("ZZZZ")
        assert result == (2003, 1)

    @patch("stockdownloader.core.models.symbol.get_symbol_info")
    def test_ipo_quarter_floor_pre_2003(
        self, mock_get_info: MagicMock,
    ) -> None:
        """Returns (2003, 1) when IPO is before 2003."""
        from stockdownloader.core.models.symbol import SymbolInfo

        mock_get_info.return_value = SymbolInfo(
            symbol="AAPL",
            cusip="037833100",
            ipo_date=date(1980, 12, 12),
            name="Apple Inc",
        )
        result = ipo_quarter_floor("AAPL")
        assert result == (2003, 1)


# ------------------------------------------------------------------
# splits_for_symbol
# ------------------------------------------------------------------


class TestSplitsForSymbol:
    """splits_for_symbol filters splits correctly."""

    def test_splits_for_symbol(self) -> None:
        """Returns only splits matching the requested symbol."""
        gme_splits = splits_for_symbol("GME")
        assert len(gme_splits) == 1
        assert gme_splits[0].symbol == "GME"
        assert gme_splits[0].split_ratio == Decimal("4")

    def test_splits_for_symbol_case_insensitive(self) -> None:
        """Handles case-insensitive symbol lookup."""
        gme_splits = splits_for_symbol("gme")
        assert len(gme_splits) == 1
        assert gme_splits[0].symbol == "GME"

    def test_splits_for_symbol_multiple(self) -> None:
        """Returns multiple splits for symbols with multiple events."""
        nvda_splits = splits_for_symbol("NVDA")
        assert len(nvda_splits) == 2

    def test_splits_for_symbol_none(self) -> None:
        """Returns empty list for symbol with no known splits."""
        result = splits_for_symbol("ZZZZ")
        assert result == []

    def test_splits_for_symbol_extra(self) -> None:
        """Includes extra splits passed by the caller."""
        extra = [
            SplitAdjustment(
                symbol="ZZZZ",
                split_date=date(2024, 1, 1),
                split_ratio=Decimal("2"),
            ),
        ]
        result = splits_for_symbol("ZZZZ", extra_splits=extra)
        assert len(result) == 1
        assert result[0].split_ratio == Decimal("2")


# ------------------------------------------------------------------
# fetch_all_efts_hits
# ------------------------------------------------------------------


class TestFetchAllEftsHits:
    """fetch_all_efts_hits paginates through EFTS results."""

    @patch("stockdownloader.data.sec_common.fetch_efts_page")
    def test_fetch_all_efts_hits_pagination(
        self, mock_fetch: MagicMock,
    ) -> None:
        """Paginates correctly across multiple pages."""
        page1 = [{"_id": str(i)} for i in range(100)]
        page2 = [{"_id": str(i)} for i in range(100, 150)]

        mock_fetch.side_effect = [page1, page2]

        session = _mock_session()
        result = fetch_all_efts_hits(
            session, "https://efts.sec.gov/search?q=test",
        )
        assert len(result) == 150
        assert mock_fetch.call_count == 2

        # Verify URL construction
        first_url = mock_fetch.call_args_list[0][0][1]
        assert "&from=0&size=100" in first_url
        second_url = mock_fetch.call_args_list[1][0][1]
        assert "&from=100&size=100" in second_url

    @patch("stockdownloader.data.sec_common.fetch_efts_page")
    def test_fetch_all_efts_hits_empty(
        self, mock_fetch: MagicMock,
    ) -> None:
        """Returns empty list when first page has no results."""
        mock_fetch.return_value = []

        session = _mock_session()
        result = fetch_all_efts_hits(
            session, "https://efts.sec.gov/search?q=nothing",
        )
        assert result == []
        assert mock_fetch.call_count == 1

    @patch("stockdownloader.data.sec_common.fetch_efts_page")
    def test_fetch_all_efts_hits_none_response(
        self, mock_fetch: MagicMock,
    ) -> None:
        """Returns empty list when fetch_efts_page returns None."""
        mock_fetch.return_value = None

        session = _mock_session()
        result = fetch_all_efts_hits(
            session, "https://efts.sec.gov/search?q=fail",
        )
        assert result == []

    @patch("stockdownloader.data.sec_common.fetch_efts_page")
    def test_fetch_all_efts_hits_rate_limit(
        self, mock_fetch: MagicMock,
    ) -> None:
        """Passes rate_limit_fn through to fetch_efts_page."""
        mock_fetch.return_value = [{"_id": "1"}]  # partial page stops loop
        rate_fn = MagicMock()

        session = _mock_session()
        fetch_all_efts_hits(
            session, "https://efts.sec.gov/search?q=test",
            rate_limit_fn=rate_fn,
        )
        # rate_limit_fn is passed through (not called directly)
        _, kwargs = mock_fetch.call_args
        assert kwargs["rate_limit_fn"] is rate_fn
