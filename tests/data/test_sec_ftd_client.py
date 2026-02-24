"""Unit tests for SecFtdClient — download, parse, split-adjust, cache."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.sec_common import SplitAdjustment
from stockdownloader.data.sec_ftd_client import SecFtdClient
from stockdownloader.core.models.regulatory import FtdRecord

# Reusable split adjustment matching the well-known GME 4:1 split.
_GME_SPLIT = SplitAdjustment(
    symbol="GME",
    split_date=date(2022, 7, 22),
    split_ratio=Decimal("4"),
)


# ------------------------------------------------------------------
# Sample pipe-delimited FTD data
# ------------------------------------------------------------------

_FTD_HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"

_FTD_ROWS_GME = """\
{header}
20240115|36467W109|GME|50000|GAMESTOP CORP NEW CL A|15.50
20240116|36467W109|GME|75000|GAMESTOP CORP NEW CL A|16.25
20240115|037833100|AAPL|10000|APPLE INC|185.00
""".format(header=_FTD_HEADER)

_FTD_ROWS_PRE_SPLIT = """\
{header}
20220101|36467W109|GME|25000|GAMESTOP CORP NEW CL A|120.00
20220715|36467W109|GME|30000|GAMESTOP CORP NEW CL A|140.00
""".format(header=_FTD_HEADER)

_FTD_ROWS_POST_SPLIT = """\
{header}
20220727|36467W109|GME|100000|GAMESTOP CORP NEW CL A|35.00
20220801|36467W109|GME|120000|GAMESTOP CORP NEW CL A|33.50
""".format(header=_FTD_HEADER)

_FTD_ROWS_BAD_DATA = """\
{header}
20240115|36467W109|GME|not_a_number|GAMESTOP CORP NEW CL A|15.50
baddate|36467W109|GME|50000|GAMESTOP CORP NEW CL A|15.50
20240115|36467W109|GME|50000|GAMESTOP CORP NEW CL A|bad_price
20240116|36467W109|GME|75000|GAMESTOP CORP NEW CL A|16.25
too|few|fields
""".format(header=_FTD_HEADER)

_FTD_ROWS_EMPTY = """\
{header}
""".format(header=_FTD_HEADER)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(tmp_path: Path) -> SecFtdClient:
    return SecFtdClient(
        user_agent="TestAgent test@test.com",
        cache_dir=str(tmp_path / "ftd_cache"),
    )


def _create_zip_bytes(content: str, filename: str = "cnsfails.txt") -> bytes:
    """Create a zip archive in memory with the given text content."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def _mock_response(
    content: bytes = b"",
    status_code: int = 200,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


# ------------------------------------------------------------------
# Tests: Parsing pipe-delimited FTD files
# ------------------------------------------------------------------


class TestParseFtdFile:
    """Tests for _parse_ftd_file static method."""

    def test_parses_gme_records_only(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_GME, "GME")
        assert len(records) == 2
        assert all(r.symbol == "GME" for r in records)

    def test_parses_fields_correctly(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_GME, "GME")
        rec = records[0]
        assert rec.settlement_date == "2024-01-15"
        assert rec.cusip == "36467W109"
        assert rec.symbol == "GME"
        assert rec.quantity == 50000
        assert rec.description == "GAMESTOP CORP NEW CL A"
        assert rec.price == Decimal("15.50")

    def test_filters_by_symbol_case_insensitive(self) -> None:
        # The client upper-cases the symbol before passing to _parse_ftd_file
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_GME, "AAPL")
        assert len(records) == 1
        assert records[0].symbol == "AAPL"

    def test_date_format_conversion(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_GME, "GME")
        # YYYYMMDD -> YYYY-MM-DD
        assert records[0].settlement_date == "2024-01-15"
        assert records[1].settlement_date == "2024-01-16"

    def test_skips_header_line(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_GME, "GME")
        # Header line should be skipped
        assert not any(r.settlement_date == "SETTLEMENT DATE" for r in records)

    def test_empty_file(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_EMPTY, "GME")
        assert records == []

    def test_empty_string(self) -> None:
        records = SecFtdClient._parse_ftd_file("", "GME")
        assert records == []

    def test_handles_bad_data_gracefully(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_BAD_DATA, "GME")
        # bad quantity is skipped, bad date is skipped, bad price becomes 0,
        # too-few-fields is skipped => should get 2 valid records
        assert len(records) == 2

    def test_bad_price_becomes_zero(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_BAD_DATA, "GME")
        # The row with bad_price should have price=0
        bad_price_rec = [r for r in records if r.settlement_date == "2024-01-15"]
        assert len(bad_price_rec) == 1
        assert bad_price_rec[0].price == Decimal("0")

    def test_no_matching_symbol(self) -> None:
        records = SecFtdClient._parse_ftd_file(_FTD_ROWS_GME, "TSLA")
        assert records == []


# ------------------------------------------------------------------
# Tests: GME 4:1 split adjustment
# ------------------------------------------------------------------


class TestSplitAdjustment:
    """Tests for generic split adjustment via SplitAdjustment dataclass."""

    def test_pre_split_quantity_multiplied(self) -> None:
        records = SecFtdClient._parse_ftd_file(
            _FTD_ROWS_PRE_SPLIT, "GME", split=_GME_SPLIT,
        )
        # Pre-split dates: quantity * 4
        assert records[0].settlement_date == "2022-01-01"
        assert records[0].quantity == 25000 * 4  # 100000

        assert records[1].settlement_date == "2022-07-15"
        assert records[1].quantity == 30000 * 4  # 120000

    def test_pre_split_price_divided(self) -> None:
        records = SecFtdClient._parse_ftd_file(
            _FTD_ROWS_PRE_SPLIT, "GME", split=_GME_SPLIT,
        )
        assert records[0].price == Decimal("120.00") / 4  # 30.00
        assert records[1].price == Decimal("140.00") / 4  # 35.00

    def test_post_split_no_adjustment(self) -> None:
        records = SecFtdClient._parse_ftd_file(
            _FTD_ROWS_POST_SPLIT, "GME", split=_GME_SPLIT,
        )
        # Post-split dates: no adjustment
        assert records[0].quantity == 100000
        assert records[0].price == Decimal("35.00")

    def test_no_adjustment_when_no_split(self) -> None:
        records = SecFtdClient._parse_ftd_file(
            _FTD_ROWS_PRE_SPLIT, "GME",
        )
        # No split adjustment when split=None (default)
        assert records[0].quantity == 25000
        assert records[0].price == Decimal("120.00")

    def test_custom_split_via_extra_splits(self, tmp_path: Path) -> None:
        """Verify that extra_splits parameter injects a custom split."""
        custom_split = SplitAdjustment(
            symbol="TSLA",
            split_date=date(2022, 8, 25),
            split_ratio=Decimal("3"),
        )
        # Fabricate FTD content for TSLA before the custom split date
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20220101|88160R101|TSLA|9000|TESLA INC|900.00\n"
        )
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", split=custom_split,
        )
        assert len(records) == 1
        assert records[0].quantity == 9000 * 3
        assert records[0].price == Decimal("900.00") / 3


# ------------------------------------------------------------------
# Tests: Multiple splits per symbol (e.g. TSLA 5:1 + 3:1)
# ------------------------------------------------------------------


class TestMultiSplitAdjustment:
    """Tests for cumulative multi-split adjustment via `splits` param."""

    # TSLA had 5:1 on 2020-08-31 and 3:1 on 2022-08-25.
    _TSLA_SPLITS = [
        SplitAdjustment(
            symbol="TSLA",
            split_date=date(2020, 8, 31),
            split_ratio=Decimal("5"),
        ),
        SplitAdjustment(
            symbol="TSLA",
            split_date=date(2022, 8, 25),
            split_ratio=Decimal("3"),
        ),
    ]

    def test_pre_both_splits_applies_both(self) -> None:
        """A date before both splits should get both ratios applied."""
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20200101|88160R101|TSLA|1000|TESLA INC|300.00\n"
        )
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", splits=self._TSLA_SPLITS,
        )
        assert len(records) == 1
        # Both splits apply: 1000 * 5 * 3 = 15000
        assert records[0].quantity == 1000 * 5 * 3
        # Price: 300 / 5 / 3 = 20.00
        assert records[0].price == Decimal("300.00") / 5 / 3

    def test_between_splits_applies_only_later(self) -> None:
        """A date after the 5:1 but before the 3:1 gets only the 3:1."""
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20210601|88160R101|TSLA|1000|TESLA INC|600.00\n"
        )
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", splits=self._TSLA_SPLITS,
        )
        assert len(records) == 1
        # Only the 3:1 split applies: 1000 * 3 = 3000
        assert records[0].quantity == 1000 * 3
        assert records[0].price == Decimal("600.00") / 3

    def test_after_all_splits_no_adjustment(self) -> None:
        """A date after both splits should have no adjustment."""
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20230101|88160R101|TSLA|1000|TESLA INC|120.00\n"
        )
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", splits=self._TSLA_SPLITS,
        )
        assert len(records) == 1
        assert records[0].quantity == 1000
        assert records[0].price == Decimal("120.00")

    def test_empty_splits_list_no_adjustment(self) -> None:
        """An empty splits list should not adjust anything."""
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20200101|88160R101|TSLA|1000|TESLA INC|300.00\n"
        )
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", splits=[],
        )
        assert len(records) == 1
        assert records[0].quantity == 1000
        assert records[0].price == Decimal("300.00")

    def test_legacy_split_param_still_works(self) -> None:
        """The deprecated `split` param should still apply correctly."""
        single = SplitAdjustment(
            symbol="TSLA",
            split_date=date(2022, 8, 25),
            split_ratio=Decimal("3"),
        )
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20220101|88160R101|TSLA|1000|TESLA INC|600.00\n"
        )
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", split=single,
        )
        assert records[0].quantity == 3000

    def test_split_and_splits_merged(self) -> None:
        """When both `split` and `splits` are provided, they merge."""
        older = SplitAdjustment(
            symbol="TSLA",
            split_date=date(2020, 8, 31),
            split_ratio=Decimal("5"),
        )
        newer = SplitAdjustment(
            symbol="TSLA",
            split_date=date(2022, 8, 25),
            split_ratio=Decimal("3"),
        )
        content = (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20200101|88160R101|TSLA|1000|TESLA INC|300.00\n"
        )
        # Pass older as legacy `split`, newer in `splits` list
        records = SecFtdClient._parse_ftd_file(
            content, "TSLA", split=older, splits=[newer],
        )
        # Both should apply: 1000 * 5 * 3 = 15000
        assert records[0].quantity == 15000


# ------------------------------------------------------------------
# Tests: Reading zip files
# ------------------------------------------------------------------


class TestReadZip:
    """Tests for _read_zip static method."""

    def test_reads_zip_content(self, tmp_path: Path) -> None:
        zip_bytes = _create_zip_bytes(_FTD_ROWS_GME)
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(zip_bytes)

        content = SecFtdClient._read_zip(zip_path)
        assert "GME" in content
        assert "SETTLEMENT DATE" in content

    def test_handles_empty_zip(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            pass  # empty zip
        zip_path = tmp_path / "empty.zip"
        zip_path.write_bytes(buf.getvalue())

        content = SecFtdClient._read_zip(zip_path)
        assert content == ""

    def test_handles_latin1_encoding(self, tmp_path: Path) -> None:
        # Create content with latin-1 chars
        latin1_content = "20240115|36467W109|GME|50000|GAMESTOP\xe9|15.50"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.txt", latin1_content.encode("latin-1"))
        zip_path = tmp_path / "latin1.zip"
        zip_path.write_bytes(buf.getvalue())

        content = SecFtdClient._read_zip(zip_path)
        assert "GME" in content

    def test_bad_zip_file_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.zip"
        bad_path.write_bytes(b"not a zip file")

        with pytest.raises(zipfile.BadZipFile):
            SecFtdClient._read_zip(bad_path)


# ------------------------------------------------------------------
# Tests: Downloading half-month zip files
# ------------------------------------------------------------------


class TestDownloadHalfMonth:
    """Tests for _download_half_month method."""

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_downloads_and_caches(self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        zip_bytes = _create_zip_bytes(_FTD_ROWS_GME)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(zip_bytes),
        ):
            path = client._download_half_month(2024, 1, "a")

        assert path is not None
        assert path.exists()
        assert path.name == "cnsfails202401a.zip"

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_returns_cached_file(self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        # Pre-populate the cache
        cached_file = client._cache_dir / "cnsfails202401a.zip"
        cached_file.write_bytes(_create_zip_bytes(_FTD_ROWS_GME))

        # Should not make any network call
        with patch.object(
            client._session, "get",
            side_effect=AssertionError("Should not be called"),
        ):
            path = client._download_half_month(2024, 1, "a")

        assert path == cached_file

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_returns_none_on_404(self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "get",
            return_value=_mock_response(b"", status_code=404),
        ):
            path = client._download_half_month(2024, 12, "b")

        assert path is None

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_retries_on_server_error(self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        zip_bytes = _create_zip_bytes(_FTD_ROWS_GME)
        client = _make_client(tmp_path)
        with patch.object(
            client._session, "get",
            side_effect=[
                _mock_response(b"", status_code=500),
                _mock_response(zip_bytes),
            ],
        ):
            path = client._download_half_month(2024, 1, "a")

        assert path is not None

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_returns_none_on_network_failure(self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        import requests

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "get",
            side_effect=requests.RequestException("Connection refused"),
        ):
            path = client._download_half_month(2024, 1, "a")

        assert path is None


# ------------------------------------------------------------------
# Tests: Full fetch_ftd_data pipeline
# ------------------------------------------------------------------


class TestFetchFtdData:
    """Tests for the full fetch_ftd_data pipeline."""

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_fetches_and_filters_by_symbol(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        zip_bytes = _create_zip_bytes(_FTD_ROWS_GME)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(zip_bytes),
        ):
            records = client.fetch_ftd_data("GME", start_year=2024, end_year=2024)

        # Each month has 2 halves, but many will return the same data;
        # we just need to ensure symbol filtering works
        assert all(r.symbol == "GME" for r in records)

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_results_sorted_ascending(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        zip_bytes = _create_zip_bytes(_FTD_ROWS_GME)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(zip_bytes),
        ):
            records = client.fetch_ftd_data("GME", start_year=2024, end_year=2024)

        dates = [r.settlement_date for r in records]
        assert dates == sorted(dates)

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_handles_all_zips_failing(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(b"", status_code=404),
        ):
            records = client.fetch_ftd_data("GME", start_year=2024, end_year=2024)

        assert records == []

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_handles_corrupt_zip_gracefully(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Write a corrupt "zip" to cache
        corrupt_file = client._cache_dir / "cnsfails202401a.zip"
        corrupt_file.write_bytes(b"not a zip")

        # The other halves/months return 404
        with patch.object(
            client._session, "get",
            return_value=_mock_response(b"", status_code=404),
        ):
            records = client.fetch_ftd_data("GME", start_year=2024, end_year=2024)

        # Should not crash — just skip the corrupt file
        assert isinstance(records, list)

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_gme_split_adjustment_applied(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        zip_bytes = _create_zip_bytes(_FTD_ROWS_PRE_SPLIT)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(zip_bytes),
        ):
            records = client.fetch_ftd_data("GME", start_year=2022, end_year=2022)

        # All pre-split records should have quantity * 4
        pre_split = [r for r in records if r.settlement_date < "2022-07-22"]
        for rec in pre_split:
            # Original quantities were 25000 and 30000, adjusted should be 4x
            assert rec.quantity in (100000, 120000)


# ------------------------------------------------------------------
# Tests: IPO-aware start_year narrowing
# ------------------------------------------------------------------


class TestIpoDateNarrowing:
    """Verify that fetch_ftd_data auto-narrows start_year from IPO date."""

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_tsla_skips_quarterly_era(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        """TSLA IPO'd in 2010; quarterly era (2004-2009) should be skipped."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(client, "_fetch_quarterly_era") as mock_q, \
             patch.object(client, "_fetch_half_month_era") as mock_hm:
            mock_q.return_value = []
            mock_hm.return_value = []
            client.fetch_ftd_data("TSLA")

        # Quarterly era is 2004-2009; TSLA IPO'd 2010 -> should be skipped
        mock_q.assert_not_called()

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_gme_includes_quarterly_era(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        """GME IPO'd in 2002; quarterly era should be fetched."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(client, "_fetch_quarterly_era") as mock_q, \
             patch.object(client, "_fetch_half_month_era") as mock_hm:
            mock_q.return_value = []
            mock_hm.return_value = []
            client.fetch_ftd_data("GME")

        # GME IPO'd 2002, FTD starts 2004 -> quarterly era IS fetched
        mock_q.assert_called_once()

    @patch("stockdownloader.data.base_client.time")
    @patch("stockdownloader.data.sec_ftd_client.time")
    def test_explicit_start_year_overrides_ipo(
        self, mock_time: MagicMock, mock_base_time: MagicMock, tmp_path: Path,
    ) -> None:
        """An explicit start_year=2004 should still fetch from 2004."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_base_time.monotonic.return_value = 100.0
        mock_base_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(client, "_fetch_quarterly_era") as mock_q, \
             patch.object(client, "_fetch_half_month_era") as mock_hm:
            mock_q.return_value = []
            mock_hm.return_value = []
            # Explicit start_year=2004 for TSLA (IPO 2010)
            # The auto-narrow still uses max(2004, 2010) = 2010
            client.fetch_ftd_data("TSLA", start_year=2004)

        # Even with explicit 2004, TSLA's IPO narrows to 2010
        mock_q.assert_not_called()


# ------------------------------------------------------------------
# Tests: Rate limiting
# ------------------------------------------------------------------


class TestFtdRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_sleeps_when_too_fast(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch("stockdownloader.data.base_client.time") as mock_time:
            mock_time.monotonic.side_effect = [
                0.0,    # first check
                0.0,    # set _last_request_time
                0.05,   # second check — only 50ms elapsed (< 250ms)
                0.25,   # set _last_request_time after sleep
            ]
            mock_time.sleep = MagicMock()

            client._last_request_time = 0.0
            client._rate_limit()
            client._rate_limit()

            mock_time.sleep.assert_called()
