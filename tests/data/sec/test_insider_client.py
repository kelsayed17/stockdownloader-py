"""Unit tests for SecInsiderClient — caching, CUSIP resolution, model validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from stockdownloader.data.sec.insider_client import SecInsiderClient
from stockdownloader.core.models.regulatory import (
    BeneficialOwner,
    InsiderTransaction,
)


def _make_client(tmp_path: Path) -> SecInsiderClient:
    return SecInsiderClient(data_dir=str(tmp_path))


# ------------------------------------------------------------------
# Tests: Caching
# ------------------------------------------------------------------


class TestInsiderCaching:
    def test_save_and_load_transactions_csv(self, tmp_path: Path) -> None:
        """Transactions round-trip through CSV."""
        client = _make_client(tmp_path)
        txns = [
            InsiderTransaction(
                filing_date="2024-03-15",
                transaction_date="2024-03-15",
                owner_name="Cohen Ryan",
                owner_cik="9999999",
                owner_title="",
                is_director=True,
                is_officer=False,
                is_ten_pct_owner=True,
                transaction_code="P",
                shares=100000,
                price_per_share=15.50,
                shares_owned_after=9100000,
                direct_or_indirect="D",
            ),
        ]
        client._save_transaction_cache("GME", txns)

        # CSV file must exist (not JSON)
        assert (tmp_path / "GME" / "insider_transactions.csv").exists()
        assert not (tmp_path / "GME" / "insider_transactions.json").exists()

        loaded = client._load_transaction_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "Cohen Ryan"
        assert loaded[0].shares == 100000

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._load_transaction_cache("NOSYMBOL") is None

    def test_load_corrupt_csv_returns_none(self, tmp_path: Path) -> None:
        """Corrupt CSV falls through (no JSON fallback either)."""
        client = _make_client(tmp_path)
        sym_dir = tmp_path / "BAD"
        sym_dir.mkdir()
        # Write a CSV whose header doesn't match the dataclass fields
        # so _load_csv returns None (KeyError on missing column).
        (sym_dir / "insider_transactions.csv").write_text(
            "bad_col\nvalue\n", encoding="utf-8",
        )
        assert client._load_transaction_cache("BAD") is None

    def test_json_fallback_migrates_to_csv(self, tmp_path: Path) -> None:
        """Legacy JSON cache is loaded and migrated to CSV."""
        client = _make_client(tmp_path)
        sym_dir = tmp_path / "GME"
        sym_dir.mkdir()
        data = [
            {
                "filing_date": "2024-03-15",
                "transaction_date": "2024-03-15",
                "owner_name": "Legacy Owner",
                "owner_cik": "1111111",
                "owner_title": "",
                "is_director": False,
                "is_officer": False,
                "is_ten_pct_owner": False,
                "transaction_code": "P",
                "shares": 500,
                "price_per_share": 10.0,
                "shares_owned_after": 500,
                "direct_or_indirect": "D",
            },
        ]
        (sym_dir / "insider_transactions.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        loaded = client._load_transaction_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "Legacy Owner"
        # CSV should now exist after migration
        assert (sym_dir / "insider_transactions.csv").exists()

    def test_save_and_load_beneficial_owners_csv(self, tmp_path: Path) -> None:
        """Beneficial owners round-trip through CSV."""
        client = _make_client(tmp_path)
        owners = [
            BeneficialOwner(
                filing_date="2024-01-15",
                owner_name="RC Ventures LLC",
                owner_cik="1822844",
                form_type="SC 13D/A",
                shares_beneficially_owned=36300000,
                percent_of_class=11.9,
                sole_voting_power=36300000,
                shared_voting_power=0,
                sole_dispositive_power=36300000,
                shared_dispositive_power=0,
                filing_url="https://sec.gov/example",
            ),
        ]
        client._save_beneficial_owners_cache("GME", owners)

        # CSV file must exist (not JSON)
        assert (tmp_path / "GME" / "beneficial_owners.csv").exists()
        assert not (tmp_path / "GME" / "beneficial_owners.json").exists()

        loaded = client._load_beneficial_owners_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "RC Ventures LLC"
        assert loaded[0].shares_beneficially_owned == 36300000

    def test_beneficial_owners_json_fallback(self, tmp_path: Path) -> None:
        """Legacy JSON beneficial owners cache migrates to CSV."""
        client = _make_client(tmp_path)
        sym_dir = tmp_path / "GME"
        sym_dir.mkdir()
        data = [
            {
                "filing_date": "2024-01-15",
                "owner_name": "Legacy Fund",
                "owner_cik": "9999",
                "form_type": "SC 13D",
                "shares_beneficially_owned": 100000,
                "percent_of_class": 5.0,
                "sole_voting_power": 100000,
                "shared_voting_power": 0,
                "sole_dispositive_power": 100000,
                "shared_dispositive_power": 0,
                "filing_url": "https://sec.gov/legacy",
            },
        ]
        (sym_dir / "beneficial_owners.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        loaded = client._load_beneficial_owners_cache("GME")
        assert loaded is not None
        assert loaded[0].owner_name == "Legacy Fund"
        assert (sym_dir / "beneficial_owners.csv").exists()

    def test_quarter_cache_roundtrip(self, tmp_path: Path) -> None:
        """Quarter cache writes to .progress/insider/."""
        client = _make_client(tmp_path)
        txns = [
            InsiderTransaction(
                filing_date="2024-03-15",
                transaction_date="2024-03-15",
                owner_name="Test Owner",
                owner_cik="1111111",
                owner_title="CEO",
                is_director=False,
                is_officer=True,
                is_ten_pct_owner=False,
                transaction_code="P",
                shares=5000,
                price_per_share=20.0,
                shares_owned_after=10000,
                direct_or_indirect="D",
            ),
        ]
        client._save_quarter_transactions("GME", 2024, 1, txns)

        # File must be under .progress/insider/ (not insider/)
        progress_file = tmp_path / "GME" / ".progress" / "insider" / "2024Q1.json"
        legacy_file = tmp_path / "GME" / "insider" / "2024Q1.json"
        assert progress_file.exists()
        assert not legacy_file.exists()

        loaded = client._load_quarter_transactions("GME", 2024, 1)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "Test Owner"

    def test_quarter_cache_legacy_migration(self, tmp_path: Path) -> None:
        """Legacy insider/{YYYY}Q{Q}.json migrates to .progress/insider/."""
        client = _make_client(tmp_path)
        legacy_dir = tmp_path / "GME" / "insider"
        legacy_dir.mkdir(parents=True)
        data = [
            {
                "filing_date": "2023-06-15",
                "transaction_date": "2023-06-15",
                "owner_name": "Legacy Quarter",
                "owner_cik": "2222222",
                "owner_title": "",
                "is_director": False,
                "is_officer": False,
                "is_ten_pct_owner": False,
                "transaction_code": "P",
                "shares": 1000,
                "price_per_share": 25.0,
                "shares_owned_after": 1000,
                "direct_or_indirect": "D",
            },
        ]
        (legacy_dir / "2023Q2.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        loaded = client._load_quarter_transactions("GME", 2023, 2)
        assert loaded is not None
        assert loaded[0].owner_name == "Legacy Quarter"
        # Legacy file should have been moved
        assert not (legacy_dir / "2023Q2.json").exists()
        assert (tmp_path / "GME" / ".progress" / "insider" / "2023Q2.json").exists()

    def test_quarter_cache_missing_returns_none(
        self, tmp_path: Path,
    ) -> None:
        client = _make_client(tmp_path)
        assert client._load_quarter_transactions("GME", 2020, 1) is None


# ------------------------------------------------------------------
# Tests: CUSIP auto-resolution
# ------------------------------------------------------------------


class TestCusipAutoResolution:
    def test_gme_uses_default_cusip(self, tmp_path: Path) -> None:
        """GME auto-resolves to known CUSIP from registry."""
        client = _make_client(tmp_path)
        with patch.object(
            client, "_fetch_13d_13g_from_efts", return_value=[],
        ) as mock_fetch:
            client.fetch_beneficial_owners("GME")

        # Should have been called with GME's CUSIP
        mock_fetch.assert_called_once()
        args = mock_fetch.call_args
        assert args[0][1] == "36467W109"

    def test_explicit_cusip_not_overridden(self, tmp_path: Path) -> None:
        """Explicit CUSIP takes precedence over registry."""
        client = _make_client(tmp_path)
        with patch.object(
            client, "_fetch_13d_13g_from_efts", return_value=[],
        ) as mock_fetch:
            client.fetch_beneficial_owners("GME", cusip="CUSTOM123")

        args = mock_fetch.call_args
        assert args[0][1] == "CUSTOM123"


# ------------------------------------------------------------------
# Tests: IPO-aware quarter floor
# ------------------------------------------------------------------


class TestIpoDateNarrowing:
    def test_tsla_skips_pre_ipo_quarters(self, tmp_path: Path) -> None:
        """TSLA (IPO 2010-06-29) should not fetch quarters before 2010."""
        client = _make_client(tmp_path)

        fetched_quarters: list[tuple[int, int]] = []

        def mock_fetch_from_bulk(symbol, year, quarter):
            fetched_quarters.append((year, quarter))
            return []

        with patch.object(
            client, "_fetch_from_bulk", side_effect=mock_fetch_from_bulk,
        ):
            client.fetch_insider_transactions(
                "TSLA", num_quarters=80, force_refresh=True,
            )

        # TSLA IPO is 2010-06-29 (Q2) — no quarter before 2010-Q2
        for y, q in fetched_quarters:
            assert (y, q) >= (2010, 2), (
                f"Fetched pre-IPO quarter {y}Q{q} for TSLA"
            )


# ------------------------------------------------------------------
# Tests: Model validation
# ------------------------------------------------------------------


class TestInsiderTransactionModel:
    def test_create_valid(self) -> None:
        txn = InsiderTransaction(
            filing_date="2024-01-15",
            transaction_date="2024-01-15",
            owner_name="Test",
            owner_cik="123",
            owner_title="CEO",
            is_director=False,
            is_officer=True,
            is_ten_pct_owner=False,
            transaction_code="P",
            shares=1000,
            price_per_share=10.0,
            shares_owned_after=5000,
            direct_or_indirect="D",
        )
        assert txn.shares == 1000
        assert txn.is_officer is True

    def test_empty_filing_date_raises(self) -> None:
        with pytest.raises(ValueError, match="filing_date"):
            InsiderTransaction(
                filing_date="",
                transaction_date="2024-01-15",
                owner_name="Test",
                owner_cik="123",
                owner_title="",
                is_director=False,
                is_officer=False,
                is_ten_pct_owner=False,
                transaction_code="P",
                shares=100,
                price_per_share=10.0,
                shares_owned_after=100,
                direct_or_indirect="D",
            )

    def test_empty_owner_name_raises(self) -> None:
        with pytest.raises(ValueError, match="owner_name"):
            InsiderTransaction(
                filing_date="2024-01-15",
                transaction_date="2024-01-15",
                owner_name="",
                owner_cik="123",
                owner_title="",
                is_director=False,
                is_officer=False,
                is_ten_pct_owner=False,
                transaction_code="P",
                shares=100,
                price_per_share=10.0,
                shares_owned_after=100,
                direct_or_indirect="D",
            )

    def test_frozen(self) -> None:
        txn = InsiderTransaction(
            filing_date="2024-01-15",
            transaction_date="2024-01-15",
            owner_name="Test",
            owner_cik="123",
            owner_title="",
            is_director=False,
            is_officer=False,
            is_ten_pct_owner=False,
            transaction_code="P",
            shares=100,
            price_per_share=10.0,
            shares_owned_after=100,
            direct_or_indirect="D",
        )
        with pytest.raises(AttributeError):
            txn.shares = 200  # type: ignore[misc]


class TestBeneficialOwnerModel:
    def test_create_valid(self) -> None:
        bo = BeneficialOwner(
            filing_date="2024-01-15",
            owner_name="RC Ventures LLC",
            owner_cik="1822844",
            form_type="SC 13D/A",
            shares_beneficially_owned=36300000,
            percent_of_class=11.9,
            sole_voting_power=36300000,
            shared_voting_power=0,
            sole_dispositive_power=36300000,
            shared_dispositive_power=0,
            filing_url="https://sec.gov/example",
        )
        assert bo.shares_beneficially_owned == 36300000

    def test_empty_owner_name_raises(self) -> None:
        with pytest.raises(ValueError, match="owner_name"):
            BeneficialOwner(
                filing_date="2024-01-15",
                owner_name="",
                owner_cik="123",
                form_type="SC 13D",
                shares_beneficially_owned=0,
                percent_of_class=0.0,
                sole_voting_power=0,
                shared_voting_power=0,
                sole_dispositive_power=0,
                shared_dispositive_power=0,
                filing_url="",
            )

    def test_empty_form_type_raises(self) -> None:
        with pytest.raises(ValueError, match="form_type"):
            BeneficialOwner(
                filing_date="2024-01-15",
                owner_name="Test",
                owner_cik="123",
                form_type="",
                shares_beneficially_owned=0,
                percent_of_class=0.0,
                sole_voting_power=0,
                shared_voting_power=0,
                sole_dispositive_power=0,
                shared_dispositive_power=0,
                filing_url="",
            )
