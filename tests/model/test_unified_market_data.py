"""Tests for UnifiedMarketData model."""

from decimal import Decimal

import pytest

from stockdownloader.model.options import OptionContract, OptionType, OptionsChain
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.quote_data import QuoteData
from stockdownloader.model.unified_market_data import (
    FinancialData,
    HistoricalData,
    UnifiedMarketData,
)


def _make_contract(option_type, volume):
    return OptionContract(
        contract_symbol="SYM",
        type=option_type,
        strike=Decimal("475"),
        expiration_date="2024-01-19",
        last_price=Decimal("1"),
        bid=Decimal("1"),
        ask=Decimal("1"),
        volume=volume,
        open_interest=0,
        implied_volatility=Decimal("0"),
        delta=Decimal("0"),
        gamma=Decimal("0"),
        theta=Decimal("0"),
        vega=Decimal("0"),
        in_the_money=False,
    )


def _make_price(date, volume):
    return PriceData(
        date=date,
        open=Decimal("475"),
        high=Decimal("476"),
        low=Decimal("474"),
        close=Decimal("475"),
        adj_close=Decimal("475"),
        volume=volume,
    )


@pytest.fixture
def unified():
    return UnifiedMarketData("SPY")


def test_symbol_is_required():
    with pytest.raises(ValueError):
        UnifiedMarketData(None)


def test_default_values_are_zero_or_empty(unified):
    assert unified.symbol == "SPY"
    assert unified.equity_volume == Decimal("0")
    assert unified.options_volume == 0
    assert unified.current_price == Decimal("0")
    assert not unified.is_complete()


def test_equity_volume_from_quote_data(unified):
    quote = QuoteData()
    quote.volume = Decimal("5000000")
    unified.quote = quote
    assert unified.equity_volume == Decimal("5000000")


def test_options_volume_from_chain(unified):
    chain = OptionsChain("SPY")
    chain.add_call("2024-01-19", _make_contract(OptionType.CALL, 300))
    chain.add_put("2024-01-19", _make_contract(OptionType.PUT, 200))
    chain.add_expiration_date("2024-01-19")
    unified.options_chain = chain

    assert unified.options_volume == 500
    assert unified.call_volume == 300
    assert unified.put_volume == 200


def test_total_combined_volume(unified):
    quote = QuoteData()
    quote.volume = Decimal("1000")
    unified.quote = quote

    chain = OptionsChain("SPY")
    chain.add_call("2024-01-19", _make_contract(OptionType.CALL, 500))
    chain.add_expiration_date("2024-01-19")
    unified.options_chain = chain

    assert Decimal("1500").compare(unified.total_combined_volume) == 0


def test_average_daily_volume(unified):
    history = [
        _make_price("2024-01-01", 100_000),
        _make_price("2024-01-02", 200_000),
        _make_price("2024-01-03", 150_000),
        _make_price("2024-01-04", 250_000),
        _make_price("2024-01-05", 300_000),
    ]
    unified.price_history = history

    # Last 3 days: 150000 + 250000 + 300000 = 700000 / 3 = 233333
    avg = unified.get_average_daily_volume(3)
    assert int(avg) > 230000
    assert int(avg) < 240000


def test_average_daily_volume_empty_history(unified):
    assert unified.get_average_daily_volume(5) == Decimal("0")


def test_current_price_from_quote(unified):
    quote = QuoteData()
    quote.last_trade_price_only = Decimal("475.50")
    unified.quote = quote
    assert unified.current_price == Decimal("475.50")


def test_current_price_falls_back_to_latest_price(unified):
    price = PriceData(
        date="2024-01-05",
        open=Decimal("470"),
        high=Decimal("476"),
        low=Decimal("469"),
        close=Decimal("474"),
        adj_close=Decimal("474"),
        volume=100_000,
    )
    unified.latest_price = price
    assert unified.current_price == Decimal("474")


def test_put_call_ratio_delegation(unified):
    chain = OptionsChain("SPY")
    chain.add_call("2024-01-19", _make_contract(OptionType.CALL, 100))
    chain.add_put("2024-01-19", _make_contract(OptionType.PUT, 200))
    chain.add_expiration_date("2024-01-19")
    unified.options_chain = chain

    ratio = unified.put_call_ratio
    assert Decimal("2.0000").compare(ratio) == 0


def test_completeness_checks(unified):
    assert not unified.has_quote_data()
    assert not unified.has_price_history()
    assert not unified.has_financial_data()
    assert not unified.has_options_chain()
    assert not unified.has_historical_data()

    unified.quote = QuoteData()
    assert unified.has_quote_data()

    unified.price_history = [_make_price("2024-01-01", 1000)]
    assert unified.has_price_history()

    unified.financials = FinancialData()
    assert unified.has_financial_data()

    chain = OptionsChain("SPY")
    chain.add_expiration_date("2024-01-19")
    unified.options_chain = chain
    assert unified.has_options_chain()

    unified.historical = HistoricalData("SPY")
    assert unified.has_historical_data()

    assert unified.is_complete()


def test_to_string_contains_key_info(unified):
    s = str(unified)
    assert "SPY" in s
