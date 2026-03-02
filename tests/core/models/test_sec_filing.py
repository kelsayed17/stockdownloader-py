"""Tests for SecFiling model."""

from __future__ import annotations

import pytest

from stockdownloader.core.models.regulatory import SecFiling


def _make_filing(**overrides) -> SecFiling:
    defaults = dict(
        accession_number="0001326380-24-000013",
        filing_date="2024-03-26",
        report_date="2024-02-03",
        form="10-K",
        primary_document="gme-20240203.htm",
        description="Annual Report",
        filing_url="https://www.sec.gov/Archives/edgar/data/1326380/000132638024000013/gme-20240203.htm",
    )
    defaults.update(overrides)
    return SecFiling(**defaults)


class TestCreation:
    def test_create_valid_filing(self) -> None:
        f = _make_filing()
        assert f.accession_number == "0001326380-24-000013"
        assert f.filing_date == "2024-03-26"
        assert f.report_date == "2024-02-03"
        assert f.form == "10-K"
        assert f.primary_document == "gme-20240203.htm"
        assert f.description == "Annual Report"
        assert "sec.gov" in f.filing_url

    def test_frozen(self) -> None:
        f = _make_filing()
        with pytest.raises(AttributeError):
            f.form = "8-K"  # type: ignore[misc]

    def test_equality(self) -> None:
        f1 = _make_filing()
        f2 = _make_filing()
        assert f1 == f2
        assert hash(f1) == hash(f2)


class TestValidation:
    def test_empty_accession_number_raises(self) -> None:
        with pytest.raises(ValueError, match="accession_number"):
            _make_filing(accession_number="")

    def test_empty_filing_date_raises(self) -> None:
        with pytest.raises(ValueError, match="filing_date"):
            _make_filing(filing_date="")

    def test_empty_form_raises(self) -> None:
        with pytest.raises(ValueError, match="form"):
            _make_filing(form="")


class TestIsMaterial:
    @pytest.mark.parametrize(
        "form",
        ["10-K", "10-Q", "8-K", "SC 13D", "SC 13D/A", "DEF 14A", "DEFA14A"],
    )
    def test_material_forms(self, form: str) -> None:
        f = _make_filing(form=form)
        assert f.is_material is True

    @pytest.mark.parametrize("form", ["4", "S-1", "13F-HR", "SD"])
    def test_non_material_forms(self, form: str) -> None:
        f = _make_filing(form=form)
        assert f.is_material is False


class TestStr:
    def test_contains_date_and_form(self) -> None:
        f = _make_filing()
        s = str(f)
        assert "2024-03-26" in s
        assert "10-K" in s
        assert "Annual Report" in s
