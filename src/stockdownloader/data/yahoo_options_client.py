"""Downloads options chain data from Yahoo Finance v7 options API.
Captures full contract details including volume, open interest, IV, and greeks.

Endpoint: https://query1.finance.yahoo.com/v7/finance/options/{ticker}
Supports querying specific expiration dates via ?date={epoch} parameter.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper
from stockdownloader.model import OptionContract, OptionsChain, OptionType

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_OPTIONS_URL = "https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
_DATE_FMT = "%Y-%m-%d"
_NY_TZ = ZoneInfo("America/New_York")


class YahooOptionsClient:
    """Fetches options chain data from Yahoo Finance."""

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> OptionsChain:
        """Download the full options chain for *ticker* (all available
        expirations).
        """
        chain = OptionsChain(ticker)

        if self._auth.crumb is None:
            self._auth.authenticate()

        # First request gets expiration dates and the nearest expiration chain
        self._fetch_with_retry(
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
                self._fetch_with_retry(
                    url=url,
                    chain=chain,
                    context=f"options chain for {ticker} exp:{exp_date}",
                )

        return chain

    def download_for_expiration(
        self, ticker: str, expiration_date: str
    ) -> OptionsChain:
        """Download options chain for a specific *expiration_date*
        (``YYYY-MM-DD``).
        """
        chain = OptionsChain(ticker)

        if self._auth.crumb is None:
            self._auth.authenticate()

        epoch = self._date_to_epoch(expiration_date)
        url = (
            _OPTIONS_URL.format(symbol=ticker)
            + f"?date={epoch}&crumb={self._auth.crumb}"
        )
        self._fetch_with_retry(
            url=url,
            chain=chain,
            context=f"options chain for {ticker} exp:{expiration_date}",
        )

        return chain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self, url: str, chain: OptionsChain, context: str
    ) -> None:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._auth.session.get(url, timeout=15)
                self._parse_options_json(resp.text, chain)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "Retrying %s, attempt %d", context, attempt + 1
                    )
                else:
                    logger.warning(
                        "Failed %s after %d retries: %s",
                        context,
                        _MAX_RETRIES,
                        last_exc,
                    )

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
                chain.underlying_price = _get_decimal(quote, "regularMarketPrice")

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
        except Exception as exc:
            logger.warning("Error parsing options chain JSON: %s", exc)

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
        return OptionContract(
            contract_symbol=_get_string(obj, "contractSymbol"),
            option_type=option_type,
            strike=_get_decimal(obj, "strike"),
            expiration_date=exp_date,
            last_price=_get_decimal(obj, "lastPrice"),
            bid=_get_decimal(obj, "bid"),
            ask=_get_decimal(obj, "ask"),
            volume=_get_long(obj, "volume"),
            open_interest=_get_long(obj, "openInterest"),
            implied_volatility=_get_decimal(obj, "impliedVolatility"),
            delta=_get_decimal(obj, "delta"),
            gamma=_get_decimal(obj, "gamma"),
            theta=_get_decimal(obj, "theta"),
            vega=_get_decimal(obj, "vega"),
            in_the_money=_get_boolean(obj, "inTheMoney"),
        )
    except Exception as exc:
        logger.debug("Skipping malformed option contract: %s", exc)
        return None


def _get_decimal(obj: dict, field: str) -> Decimal:
    val = obj.get(field)
    if val is None:
        return Decimal(0)
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal(0)


def _get_long(obj: dict, field: str) -> int:
    val = obj.get(field)
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _get_string(obj: dict, field: str) -> str:
    val = obj.get(field)
    if val is None:
        return ""
    return str(val)


def _get_boolean(obj: dict, field: str) -> bool:
    val = obj.get(field)
    if val is None:
        return False
    return bool(val)
