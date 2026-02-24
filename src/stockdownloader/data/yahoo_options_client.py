"""Downloads options chain data from Yahoo Finance v7 options API.
Captures full contract details including volume, open interest, IV, and greeks.

Endpoint: https://query1.finance.yahoo.com/v7/finance/options/{ticker}
Supports querying specific expiration dates via ?date={epoch} parameter.

When Yahoo omits Greeks (common outside market hours), the client can
self-compute delta, gamma, theta, and vega from the reported implied
volatility using the Black-Scholes model.  Enable this with
``compute_greeks=True`` (the default).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import requests

from stockdownloader.data.data_parsers import (
    get_boolean,
    get_decimal,
    get_long,
    get_raw_decimal,
    get_raw_long,
    get_string,
)
from stockdownloader.data.yahoo_base_client import YahooAuthHelper, YahooBaseClient
from stockdownloader.core.models import OptionContract, OptionsChain, OptionType

logger = logging.getLogger(__name__)
_OPTIONS_URL = "https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
_DATE_FMT = "%Y-%m-%d"
_NY_TZ = ZoneInfo("America/New_York")

# Risk-free rate approximation (US 10-year Treasury as of 2025)
_RISK_FREE_RATE = Decimal("0.04")


class YahooOptionsClient(YahooBaseClient):
    """Fetches options chain data from Yahoo Finance.

    Parameters
    ----------
    auth:
        Optional :class:`YahooAuthHelper` for cookie/crumb auth.
    compute_greeks:
        If ``True`` (default), automatically compute Greeks via
        Black-Scholes when Yahoo omits them (common outside market
        hours).  Requires ``implied_volatility > 0`` on the contract.
    """

    def __init__(
        self,
        auth: YahooAuthHelper | None = None,
        compute_greeks: bool = True,
    ) -> None:
        super().__init__(auth)
        self._compute_greeks = compute_greeks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> OptionsChain:
        """Download the full options chain for *ticker* (all available
        expirations).

        If ``compute_greeks=True`` and Yahoo omits Greeks, they are
        computed from IV using the Black-Scholes model.
        """
        chain = OptionsChain(ticker)

        self._ensure_authenticated()

        # First request gets expiration dates and the nearest expiration chain
        self._fetch_options_with_retry(
            url=_OPTIONS_URL.format(symbol=ticker) + f"?crumb={self._auth.crumb}",
            chain=chain,
            context=f"options chain download for {ticker}",
        )

        # Fetch remaining expirations
        for exp_date in chain.expiration_dates:
            if not chain.get_calls(exp_date) and not chain.get_puts(exp_date):
                epoch = self._date_to_epoch(exp_date)
                url = (
                    _OPTIONS_URL.format(symbol=ticker)
                    + f"?date={epoch}&crumb={self._auth.crumb}"
                )
                self._fetch_options_with_retry(
                    url=url,
                    chain=chain,
                    context=f"options chain for {ticker} exp:{exp_date}",
                )

        # Auto-compute Greeks if missing
        if self._compute_greeks:
            chain = _enrich_greeks_if_missing(chain)

        return chain

    def download_for_expiration(
        self, ticker: str, expiration_date: str
    ) -> OptionsChain:
        """Download options chain for a specific *expiration_date*
        (``YYYY-MM-DD``).
        """
        chain = OptionsChain(ticker)

        self._ensure_authenticated()

        epoch = self._date_to_epoch(expiration_date)
        url = (
            _OPTIONS_URL.format(symbol=ticker)
            + f"?date={epoch}&crumb={self._auth.crumb}"
        )
        self._fetch_options_with_retry(
            url=url,
            chain=chain,
            context=f"options chain for {ticker} exp:{expiration_date}",
        )

        if self._compute_greeks:
            chain = _enrich_greeks_if_missing(chain)

        return chain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_options_with_retry(
        self, url: str, chain: OptionsChain, context: str
    ) -> None:
        def _parse(text: str) -> bool:
            self._parse_options_json(text, chain)
            return True

        self._fetch_with_retry(url, _parse, context)

    def _parse_options_json(self, raw: str, chain: OptionsChain) -> None:
        try:
            root = json.loads(raw)
            option_chain = root.get("optionChain")
            if option_chain is None:
                return

            results = option_chain.get("result")
            if not results:
                return

            result = results[0]

            # Parse underlying price
            quote = result.get("quote")
            if quote:
                chain.underlying_price = get_decimal(quote, "regularMarketPrice")

            # Parse expiration dates
            expirations = result.get("expirationDates", [])
            for exp in expirations:
                date_str = _epoch_to_date(int(exp))
                chain.add_expiration_date(date_str)

            # Parse options contracts
            options = result.get("options")
            if not options:
                return

            option_data = options[0]

            exp_date: str | None = None
            if "expirationDate" in option_data:
                exp_date = _epoch_to_date(int(option_data["expirationDate"]))

            # Parse calls
            calls = option_data.get("calls", [])
            if calls and exp_date is not None:
                for el in calls:
                    contract = _parse_contract(el, OptionType.CALL, exp_date)
                    if contract is not None:
                        chain.add_call(exp_date, contract)

            # Parse puts
            puts = option_data.get("puts", [])
            if puts and exp_date is not None:
                for el in puts:
                    contract = _parse_contract(el, OptionType.PUT, exp_date)
                    if contract is not None:
                        chain.add_put(exp_date, contract)
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, ValueError) as exc:
            logger.warning("Error parsing options chain JSON: %s", exc, exc_info=True)

    @staticmethod
    def _date_to_epoch(date_str: str) -> int:
        dt = datetime.strptime(date_str, _DATE_FMT).replace(tzinfo=_NY_TZ)
        return int(dt.timestamp())


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _epoch_to_date(epoch: int) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .astimezone(_NY_TZ)
        .strftime(_DATE_FMT)
    )


def _parse_contract(
    obj: dict, option_type: OptionType, exp_date: str
) -> OptionContract | None:
    try:
        # Yahoo wraps some numeric fields in {"raw": value, "fmt": "..."}.
        # Use get_raw_* helpers for fields that may be wrapped (OI, Greeks, IV)
        # and plain get_* for fields Yahoo always sends as direct values.
        return OptionContract(
            contract_symbol=get_string(obj, "contractSymbol"),
            type=option_type,
            strike=get_raw_decimal(obj, "strike"),
            expiration_date=exp_date,
            last_price=get_raw_decimal(obj, "lastPrice"),
            bid=get_raw_decimal(obj, "bid"),
            ask=get_raw_decimal(obj, "ask"),
            volume=get_raw_long(obj, "volume"),
            open_interest=get_raw_long(obj, "openInterest"),
            implied_volatility=get_raw_decimal(obj, "impliedVolatility"),
            delta=get_raw_decimal(obj, "delta"),
            gamma=get_raw_decimal(obj, "gamma"),
            theta=get_raw_decimal(obj, "theta"),
            vega=get_raw_decimal(obj, "vega"),
            in_the_money=get_boolean(obj, "inTheMoney"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Skipping malformed option contract: %s", exc)
        return None


# ------------------------------------------------------------------
# Greeks Self-Compute (Black-Scholes fallback)
# ------------------------------------------------------------------


def _enrich_greeks_if_missing(chain: OptionsChain) -> OptionsChain:
    """Re-create contracts with computed Greeks when Yahoo omits them.

    Yahoo frequently omits delta/gamma/theta/vega outside of market
    hours but still provides implied_volatility.  When IV is non-zero
    and all Greeks are zero, we compute them via Black-Scholes.

    Returns a **new** ``OptionsChain`` with enriched contracts.  If
    Greeks are already present, the original chain is returned unchanged.
    """
    # Quick check: do any contracts already have non-zero gamma?
    all_contracts = chain.all_calls + chain.all_puts
    has_any_greeks = any(float(c.gamma) != 0 for c in all_contracts)
    has_any_iv = any(float(c.implied_volatility) > 0 for c in all_contracts)

    if has_any_greeks or not has_any_iv:
        # Greeks present, or no IV to compute from — nothing to do
        return chain

    # Lazy import to avoid circular dependency
    from stockdownloader.util.options import black_scholes as bsc

    spot = chain.underlying_price
    if spot <= Decimal(0):
        return chain

    logger.info(
        "Yahoo omitted Greeks for %s — computing from IV via Black-Scholes "
        "(%d contracts)",
        chain.underlying_symbol, len(all_contracts),
    )

    now = datetime.now(_NY_TZ)
    enriched = OptionsChain(chain.underlying_symbol)
    enriched.underlying_price = spot
    for exp in chain.expiration_dates:
        enriched.add_expiration_date(exp)

    computed_count = 0

    for exp in chain.expiration_dates:
        # Compute time-to-expiry in years
        try:
            exp_dt = datetime.strptime(exp, _DATE_FMT).replace(tzinfo=_NY_TZ)
            # Use 16:00 ET as expiration time
            exp_dt = exp_dt.replace(hour=16, minute=0, second=0)
            tte_days = (exp_dt - now).total_seconds() / 86400.0
            tte_years = Decimal(str(max(tte_days / 365.0, 0.001)))
        except ValueError:
            tte_years = Decimal("0.001")

        for contract in chain.get_calls(exp):
            new_c = _compute_greeks_for_contract(
                contract, spot, tte_years, bsc
            )
            enriched.add_call(exp, new_c)
            if new_c is not contract:
                computed_count += 1

        for contract in chain.get_puts(exp):
            new_c = _compute_greeks_for_contract(
                contract, spot, tte_years, bsc
            )
            enriched.add_put(exp, new_c)
            if new_c is not contract:
                computed_count += 1

    logger.info(
        "Computed Greeks for %d / %d contracts via Black-Scholes",
        computed_count, len(all_contracts),
    )
    return enriched


def _compute_greeks_for_contract(
    contract: OptionContract,
    spot: Decimal,
    tte_years: Decimal,
    bsc: object,
) -> OptionContract:
    """Compute and return a new contract with Black-Scholes Greeks.

    If the contract already has non-zero gamma or has zero IV, return
    the original contract unchanged.
    """
    iv = contract.implied_volatility
    if float(iv) <= 0 or float(contract.gamma) != 0:
        return contract

    try:
        computed_delta = bsc.delta(
            contract.type, spot, contract.strike,
            tte_years, _RISK_FREE_RATE, iv,
        )
        computed_gamma = bsc.gamma(
            spot, contract.strike, tte_years, _RISK_FREE_RATE, iv,
        )
        computed_theta = bsc.theta(
            contract.type, spot, contract.strike,
            tte_years, _RISK_FREE_RATE, iv,
        )
        computed_vega = bsc.vega(
            spot, contract.strike, tte_years, _RISK_FREE_RATE, iv,
        )

        return OptionContract(
            contract_symbol=contract.contract_symbol,
            type=contract.type,
            strike=contract.strike,
            expiration_date=contract.expiration_date,
            last_price=contract.last_price,
            bid=contract.bid,
            ask=contract.ask,
            volume=contract.volume,
            open_interest=contract.open_interest,
            implied_volatility=iv,
            delta=computed_delta,
            gamma=computed_gamma,
            theta=computed_theta,
            vega=computed_vega,
            in_the_money=contract.in_the_money,
        )
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        logger.debug(
            "Failed to compute Greeks for %s: %s",
            contract.contract_symbol, exc,
        )
        return contract
