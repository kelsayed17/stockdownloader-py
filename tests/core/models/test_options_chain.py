"""Tests for OptionsChain model."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.options import OptionContract, OptionType, OptionsChain


def _make_contract(symbol, option_type, strike, exp, volume, oi):
    return OptionContract(
        contract_symbol=symbol,
        type=option_type,
        strike=Decimal(strike),
        expiration_date=exp,
        last_price=Decimal("5.00"),
        bid=Decimal("4.90"),
        ask=Decimal("5.10"),
        volume=volume,
        open_interest=oi,
        implied_volatility=Decimal("0.20"),
        delta=Decimal("0"),
        gamma=Decimal("0"),
        theta=Decimal("0"),
        vega=Decimal("0"),
        in_the_money=False,
    )


@pytest.fixture
def chain():
    c = OptionsChain("SPY")
    c.underlying_price = Decimal("475.00")

    c.add_expiration_date("2024-01-19")
    c.add_expiration_date("2024-02-16")

    # Add calls for first expiration
    c.add_call("2024-01-19", _make_contract("C1", OptionType.CALL, "470", "2024-01-19", 500, 10000))
    c.add_call("2024-01-19", _make_contract("C2", OptionType.CALL, "480", "2024-01-19", 300, 8000))
    c.add_call("2024-01-19", _make_contract("C3", OptionType.CALL, "490", "2024-01-19", 200, 5000))

    # Add puts for first expiration
    c.add_put("2024-01-19", _make_contract("P1", OptionType.PUT, "460", "2024-01-19", 400, 12000))
    c.add_put("2024-01-19", _make_contract("P2", OptionType.PUT, "470", "2024-01-19", 600, 15000))

    # Add contracts for second expiration
    c.add_call("2024-02-16", _make_contract("C4", OptionType.CALL, "480", "2024-02-16", 150, 3000))
    c.add_put("2024-02-16", _make_contract("P3", OptionType.PUT, "460", "2024-02-16", 250, 7000))

    return c


def test_basic_properties(chain):
    assert chain.underlying_symbol == "SPY"
    assert chain.underlying_price == Decimal("475.00")
    assert len(chain.expiration_dates) == 2


def test_duplicate_expiration_dates_ignored(chain):
    chain.add_expiration_date("2024-01-19")
    assert len(chain.expiration_dates) == 2


def test_calls_and_puts_by_expiration(chain):
    assert len(chain.get_calls("2024-01-19")) == 3
    assert len(chain.get_puts("2024-01-19")) == 2
    assert len(chain.get_calls("2024-02-16")) == 1
    assert len(chain.get_puts("2024-02-16")) == 1


def test_empty_expiration_returns_empty_list(chain):
    assert len(chain.get_calls("2099-12-31")) == 0
    assert len(chain.get_puts("2099-12-31")) == 0


def test_total_call_volume(chain):
    # 500 + 300 + 200 + 150 = 1150
    assert chain.total_call_volume == 1150


def test_total_put_volume(chain):
    # 400 + 600 + 250 = 1250
    assert chain.total_put_volume == 1250


def test_total_volume(chain):
    assert chain.total_volume == 2400


def test_total_open_interest(chain):
    # Calls: 10000 + 8000 + 5000 + 3000 = 26000
    assert chain.total_call_open_interest == 26000
    # Puts: 12000 + 15000 + 7000 = 34000
    assert chain.total_put_open_interest == 34000


def test_put_call_ratio(chain):
    # 1250 / 1150 = 1.0870
    ratio = chain.put_call_ratio
    assert float(ratio) > 1.0
    assert float(ratio) < 1.2


def test_put_call_ratio_zero_call_volume():
    empty = OptionsChain("TEST")
    assert empty.put_call_ratio == Decimal("0")


def test_find_nearest_strike(chain):
    nearest = chain.find_nearest_strike("2024-01-19", OptionType.CALL, Decimal("477"))
    assert nearest is not None
    assert nearest.strike == Decimal("480")


def test_find_nearest_strike_exact_match(chain):
    nearest = chain.find_nearest_strike("2024-01-19", OptionType.PUT, Decimal("470"))
    assert nearest is not None
    assert nearest.strike == Decimal("470")


def test_get_contracts_at_strike(chain):
    contracts = chain.get_contracts_at_strike("2024-01-19", Decimal("470"))
    assert len(contracts) == 2  # one call and one put at 470


def test_get_all_calls_and_puts(chain):
    assert len(chain.all_calls) == 4
    assert len(chain.all_puts) == 3
