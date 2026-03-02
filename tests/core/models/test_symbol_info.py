"""Tests for SymbolInfo model and registry."""

from __future__ import annotations

from datetime import date

import pytest

from stockdownloader.core.models.symbol import (
    SymbolInfo,
    SYMBOL_REGISTRY,
    get_symbol_info,
    get_variants,
    get_family,
    get_all_tickers,
)


def _make_info(**overrides: object) -> SymbolInfo:
    defaults: dict = {
        "symbol": "TEST",
        "cusip": "123456789",
        "ipo_date": date(2020, 1, 15),
        "name": "Test Corp",
        "exchange": "NYSE",
    }
    defaults.update(overrides)
    return SymbolInfo(**defaults)


# ------------------------------------------------------------------
# Tests: Creation and immutability
# ------------------------------------------------------------------


class TestCreation:
    def test_create_valid_symbol_info(self) -> None:
        info = _make_info()
        assert info.symbol == "TEST"
        assert info.cusip == "123456789"
        assert info.ipo_date == date(2020, 1, 15)
        assert info.name == "Test Corp"
        assert info.exchange == "NYSE"

    def test_frozen(self) -> None:
        info = _make_info()
        with pytest.raises(AttributeError):
            info.symbol = "OTHER"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = _make_info()
        b = _make_info()
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality(self) -> None:
        a = _make_info(symbol="A")
        b = _make_info(symbol="B")
        assert a != b

    def test_default_exchange_empty(self) -> None:
        info = SymbolInfo(
            symbol="X", cusip="", ipo_date=date(2020, 1, 1), name="X Corp",
        )
        assert info.exchange == ""


# ------------------------------------------------------------------
# Tests: Validation
# ------------------------------------------------------------------


class TestValidation:
    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol"):
            _make_info(symbol="")

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            _make_info(name="")


# ------------------------------------------------------------------
# Tests: Derived properties
# ------------------------------------------------------------------


class TestProperties:
    def test_ipo_year(self) -> None:
        info = _make_info(ipo_date=date(2015, 6, 1))
        assert info.ipo_year == 2015

    def test_ftd_start_year_after_2004(self) -> None:
        info = _make_info(ipo_date=date(2018, 3, 1))
        assert info.ftd_start_year == 2018

    def test_ftd_start_year_before_2004(self) -> None:
        info = _make_info(ipo_date=date(1999, 1, 1))
        assert info.ftd_start_year == 2004

    def test_ftd_start_year_exactly_2004(self) -> None:
        info = _make_info(ipo_date=date(2004, 6, 1))
        assert info.ftd_start_year == 2004

    def test_ownership_start_year_after_2003(self) -> None:
        info = _make_info(ipo_date=date(2020, 1, 1))
        assert info.ownership_start_year == 2020

    def test_ownership_start_year_before_2003(self) -> None:
        info = _make_info(ipo_date=date(1999, 2, 13))
        assert info.ownership_start_year == 2003

    def test_ownership_start_year_exactly_2003(self) -> None:
        info = _make_info(ipo_date=date(2003, 6, 1))
        assert info.ownership_start_year == 2003


# ------------------------------------------------------------------
# Tests: Registry
# ------------------------------------------------------------------


class TestRegistry:
    _EXPECTED_SYMBOLS = ("GME", "AAPL", "TSLA", "AMZN", "GOOGL", "GOOG", "NVDA", "SPY")

    def test_known_symbols_present(self) -> None:
        for sym in self._EXPECTED_SYMBOLS:
            assert sym in SYMBOL_REGISTRY, f"{sym} missing from registry"

    def test_gme_metadata(self) -> None:
        gme = SYMBOL_REGISTRY["GME"]
        assert gme.cusip == "36467W109"
        assert gme.ipo_date == date(2002, 2, 13)
        assert gme.name == "GameStop Corp"
        assert gme.exchange == "NYSE"

    def test_aapl_cusip(self) -> None:
        assert SYMBOL_REGISTRY["AAPL"].cusip == "037833100"

    def test_tsla_ipo(self) -> None:
        assert SYMBOL_REGISTRY["TSLA"].ipo_date == date(2010, 6, 29)

    def test_spy_included(self) -> None:
        spy = SYMBOL_REGISTRY["SPY"]
        assert spy.cusip == "78462F103"
        assert spy.ipo_date == date(1993, 1, 29)

    def test_registry_values_are_symbol_info(self) -> None:
        for info in SYMBOL_REGISTRY.values():
            assert isinstance(info, SymbolInfo)

    def test_registry_count(self) -> None:
        assert len(SYMBOL_REGISTRY) >= len(self._EXPECTED_SYMBOLS)


# ------------------------------------------------------------------
# Tests: Lookup function
# ------------------------------------------------------------------


class TestLookup:
    def test_known_symbol(self) -> None:
        info = get_symbol_info("GME")
        assert info is not None
        assert info.symbol == "GME"

    def test_case_insensitive_lower(self) -> None:
        info = get_symbol_info("gme")
        assert info is not None
        assert info.symbol == "GME"

    def test_case_insensitive_mixed(self) -> None:
        info = get_symbol_info("Aapl")
        assert info is not None
        assert info.symbol == "AAPL"

    def test_unknown_returns_none(self) -> None:
        assert get_symbol_info("ZZZZZ") is None

    def test_empty_string_returns_none(self) -> None:
        assert get_symbol_info("") is None


# ------------------------------------------------------------------
# Tests: Variant support
# ------------------------------------------------------------------


class TestSymbolInfoVariants:
    def test_default_parent_is_none(self):
        info = get_symbol_info("GME")
        assert info is not None
        assert info.parent is None
        assert info.security_type == "common"

    def test_gmews_registered(self):
        info = get_symbol_info("GMEWS")
        assert info is not None
        assert info.parent == "GME"
        assert info.security_type == "warrant"
        assert info.cusip == "36467W117"

    def test_get_variants_returns_children(self):
        variants = get_variants("GME")
        symbols = [v.symbol for v in variants]
        assert "GMEWS" in symbols

    def test_get_variants_for_leaf_returns_empty(self):
        assert get_variants("GMEWS") == []

    def test_get_variants_unknown_returns_empty(self):
        assert get_variants("ZZZZZZ") == []

    def test_get_family_from_parent(self):
        family = get_family("GME")
        symbols = [f.symbol for f in family]
        assert "GME" in symbols
        assert "GMEWS" in symbols

    def test_get_family_from_child(self):
        family = get_family("GMEWS")
        symbols = [f.symbol for f in family]
        assert "GME" in symbols
        assert "GMEWS" in symbols

    def test_get_family_unknown_returns_empty(self):
        assert get_family("ZZZZZZ") == []

    def test_parent_field_preserved_frozen(self):
        info = SymbolInfo("TEST", "000000000", date(2020, 1, 1),
                          "Test Corp", "NYSE", parent="GME",
                          security_type="warrant")
        assert info.parent == "GME"
        assert info.security_type == "warrant"


# ------------------------------------------------------------------
# Tests: Aliases
# ------------------------------------------------------------------


class TestAliases:
    def test_default_aliases_empty(self):
        info = get_symbol_info("GME")
        assert info is not None
        assert info.aliases == ()

    def test_gmews_has_aliases(self):
        info = get_symbol_info("GMEWS")
        assert info is not None
        assert "GME WS" in info.aliases
        assert "GME-WS" in info.aliases
        assert "GME.WS" in info.aliases

    def test_get_all_tickers_parent(self):
        tickers = get_all_tickers("GME")
        assert tickers == ["GME"]

    def test_get_all_tickers_variant(self):
        tickers = get_all_tickers("GMEWS")
        assert tickers[0] == "GMEWS"
        assert "GME WS" in tickers
        assert "GME-WS" in tickers
        assert "GME.WS" in tickers
        assert "GME/WS" in tickers
        assert "GME+WS" in tickers

    def test_get_all_tickers_unknown(self):
        tickers = get_all_tickers("ZZZZZZ")
        assert tickers == ["ZZZZZZ"]

    def test_aliases_frozen(self):
        info = get_symbol_info("GMEWS")
        assert info is not None
        assert isinstance(info.aliases, tuple)
