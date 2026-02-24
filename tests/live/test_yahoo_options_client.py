"""Live tests for YahooOptionsClient (v7 options API)."""
from __future__ import annotations

import re
from decimal import Decimal

import pytest

from stockdownloader.data.yahoo_options_client import YahooOptionsClient
from stockdownloader.core.models.options import OptionContract, OptionsChain

pytestmark = pytest.mark.live

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture(scope="module")
def options_chain(yahoo_auth) -> OptionsChain:
    """Fetch options for a single expiration to keep it fast."""
    client = YahooOptionsClient(auth=yahoo_auth)
    # First get expiration dates via a minimal download
    initial = client.download_for_expiration("SPY", "2025-12-19")
    # If that specific date has no data, fall back to full download
    if not initial.all_calls and not initial.all_puts:
        return client.download("SPY")
    return initial


@pytest.fixture(scope="module")
def full_chain_metadata(yahoo_auth) -> OptionsChain:
    """Fetch just the first request to get expiration dates and underlying price."""
    client = YahooOptionsClient(auth=yahoo_auth)
    return client.download("SPY")


class TestYahooOptionsClientMetadata:

    def test_returns_options_chain_instance(self, full_chain_metadata):
        assert isinstance(full_chain_metadata, OptionsChain)

    def test_underlying_symbol_is_spy(self, full_chain_metadata):
        assert full_chain_metadata.underlying_symbol == "SPY"

    def test_underlying_price_is_positive_and_reasonable(self, full_chain_metadata):
        assert full_chain_metadata.underlying_price > Decimal("50")
        assert full_chain_metadata.underlying_price < Decimal("1500")

    def test_has_expiration_dates(self, full_chain_metadata):
        assert len(full_chain_metadata.expiration_dates) > 0

    def test_expiration_dates_are_formatted_correctly(self, full_chain_metadata):
        for date_str in full_chain_metadata.expiration_dates[:5]:
            assert _DATE_PATTERN.match(date_str), f"Bad date format: {date_str}"


class TestYahooOptionsClientContracts:

    def test_has_call_contracts(self, options_chain):
        assert len(options_chain.all_calls) > 0

    def test_has_put_contracts(self, options_chain):
        assert len(options_chain.all_puts) > 0

    def test_call_contracts_have_valid_structure(self, options_chain):
        calls = options_chain.all_calls[:5]
        for c in calls:
            assert isinstance(c, OptionContract)
            assert c.strike > Decimal("0")
            assert len(c.contract_symbol) > 0
            assert len(c.expiration_date) > 0

    def test_total_volume_is_positive(self, options_chain):
        assert options_chain.total_volume > 0

    def test_total_open_interest_is_non_negative(self, options_chain):
        call_oi = options_chain.total_call_open_interest
        put_oi = options_chain.total_put_open_interest
        assert call_oi + put_oi >= 0

    def test_put_call_ratio_is_positive(self, options_chain):
        pcr = options_chain.put_call_ratio
        assert pcr > Decimal("0")
