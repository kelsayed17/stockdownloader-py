"""Downloads options chain data with full Greeks from Tradier Sandbox API.

Tradier's sandbox API provides real ORATS-computed Greeks (delta, gamma,
theta, vega), open interest, volume, and IV for free — no paid account
needed.  The sandbox is 15-minute delayed, which is fine for squeeze
analysis.

Endpoint: https://sandbox.tradier.com/v1/markets/options/chains
Auth: Bearer token via ``TRADIER_API_TOKEN`` env var.
Rate limit: 60 requests/minute.

Usage::

    client = TradierOptionsClient()
    chain = client.download("GME")
    # chain.all_calls[0].gamma  → Decimal('0.0342')

    # Or fetch a specific expiration
    chain = client.download_for_expiration("GME", "2025-03-21")
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests

from stockdownloader.core.models import OptionContract, OptionsChain, OptionType

logger = logging.getLogger(__name__)

_BASE_URL = "https://sandbox.tradier.com/v1/markets"
_OPTIONS_CHAIN_URL = _BASE_URL + "/options/chains"
_EXPIRATIONS_URL = _BASE_URL + "/options/expirations"
_QUOTES_URL = _BASE_URL + "/quotes"

# Rate limiting: 60 req/min → minimum 1s between requests
_MIN_INTERVAL = 1.0


class TradierOptionsClient:
    """Fetches options chains with full Greeks from the Tradier Sandbox API.

    Parameters
    ----------
    api_token:
        Bearer token for Tradier API. Falls back to ``TRADIER_API_TOKEN``
        env var, then to the sandbox default token.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/options_chain.json``.
    use_sandbox:
        If True (default), use sandbox.tradier.com.
        If False, use api.tradier.com (requires paid token).
    """

    def __init__(
        self,
        api_token: str | None = None,
        data_dir: str = "data",
        use_sandbox: bool = True,
    ) -> None:
        self._token = (
            api_token
            or os.environ.get("TRADIER_API_TOKEN")
            or os.environ.get("TRADIER_SANDBOX_TOKEN")
        )
        if not self._token:
            raise ValueError(
                "Tradier API token required.  Set TRADIER_API_TOKEN or "
                "TRADIER_SANDBOX_TOKEN env var.  Get a free token at: "
                "https://web.tradier.com/user/api"
            )
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time: float = 0.0

        if use_sandbox:
            self._base_url = "https://sandbox.tradier.com/v1/markets"
        else:
            self._base_url = "https://api.tradier.com/v1/markets"

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> OptionsChain:
        """Download the full options chain for *ticker* (all available
        expirations) with complete Greeks.
        """
        chain = OptionsChain(ticker)

        # Step 1: Get underlying price
        price = self._fetch_quote_price(ticker)
        if price is not None:
            chain.underlying_price = price

        # Step 2: Get all expiration dates
        expirations = self._fetch_expirations(ticker)
        if not expirations:
            logger.warning("No expirations found for %s on Tradier", ticker)
            return chain

        for exp in expirations:
            chain.add_expiration_date(exp)

        # Step 3: Fetch chain for each expiration
        for exp in expirations:
            self._rate_limit()
            self._fetch_chain_for_expiration(ticker, exp, chain)

        logger.info(
            "Tradier: downloaded %d calls + %d puts across %d expirations for %s",
            len(chain.all_calls), len(chain.all_puts),
            len(expirations), ticker,
        )
        return chain

    def download_for_expiration(
        self, ticker: str, expiration_date: str
    ) -> OptionsChain:
        """Download options chain for a specific *expiration_date*
        (``YYYY-MM-DD``) with complete Greeks.
        """
        chain = OptionsChain(ticker)

        price = self._fetch_quote_price(ticker)
        if price is not None:
            chain.underlying_price = price

        chain.add_expiration_date(expiration_date)
        self._fetch_chain_for_expiration(ticker, expiration_date, chain)

        return chain

    def download_nearest_expiration(self, ticker: str) -> OptionsChain:
        """Download only the nearest expiration for quick analysis."""
        chain = OptionsChain(ticker)

        price = self._fetch_quote_price(ticker)
        if price is not None:
            chain.underlying_price = price

        expirations = self._fetch_expirations(ticker)
        if not expirations:
            return chain

        nearest = expirations[0]
        chain.add_expiration_date(nearest)
        self._fetch_chain_for_expiration(ticker, nearest, chain)

        return chain

    # ------------------------------------------------------------------
    # Internal: HTTP requests
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce rate limit of 60 req/min."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

    def _get_json(self, url: str, params: dict[str, str]) -> dict | None:
        """Make a GET request and return parsed JSON, or None on failure."""
        self._rate_limit()
        self._last_request_time = time.time()

        try:
            resp = self._session.get(url, params=params, timeout=30)

            if resp.status_code == 401:
                logger.error("Tradier API: unauthorized (invalid token)")
                return None
            if resp.status_code == 429:
                logger.warning("Tradier API: rate limited, waiting 5s...")
                time.sleep(5)
                resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(
                    "Tradier API returned %d for %s",
                    resp.status_code, url,
                )
                return None

            return resp.json()

        except (requests.RequestException, json.JSONDecodeError) as exc:
            logger.warning("Tradier API request failed: %s", exc)
            return None

    def _fetch_quote_price(self, ticker: str) -> Decimal | None:
        """Fetch the current underlying price."""
        url = f"{self._base_url}/quotes"
        data = self._get_json(url, {"symbols": ticker, "greeks": "false"})
        if data is None:
            return None

        try:
            quotes = data.get("quotes", {})
            quote = quotes.get("quote", {})
            if isinstance(quote, list):
                quote = quote[0]
            last = quote.get("last")
            if last is not None:
                return Decimal(str(last))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.debug("Failed to parse Tradier quote: %s", exc)

        return None

    def _fetch_expirations(self, ticker: str) -> list[str]:
        """Fetch all available expiration dates for *ticker*."""
        url = f"{self._base_url}/options/expirations"
        data = self._get_json(url, {"symbol": ticker, "includeAllRoots": "true"})
        if data is None:
            return []

        try:
            expirations = data.get("expirations", {})
            dates = expirations.get("date", [])
            if isinstance(dates, str):
                dates = [dates]
            return [d for d in dates if isinstance(d, str)]
        except (KeyError, TypeError) as exc:
            logger.debug("Failed to parse Tradier expirations: %s", exc)
            return []

    def _fetch_chain_for_expiration(
        self, ticker: str, expiration: str, chain: OptionsChain
    ) -> None:
        """Fetch the options chain for a single expiration and add to *chain*."""
        url = f"{self._base_url}/options/chains"
        data = self._get_json(url, {
            "symbol": ticker,
            "expiration": expiration,
            "greeks": "true",
        })
        if data is None:
            return

        try:
            options_data = data.get("options", {})
            if options_data is None:
                return

            contracts = options_data.get("option", [])
            if isinstance(contracts, dict):
                contracts = [contracts]

            for raw in contracts:
                contract = self._parse_contract(raw, expiration)
                if contract is None:
                    continue

                if contract.type == OptionType.CALL:
                    chain.add_call(expiration, contract)
                else:
                    chain.add_put(expiration, contract)

        except (KeyError, TypeError) as exc:
            logger.debug(
                "Failed to parse Tradier chain for %s exp:%s: %s",
                ticker, expiration, exc,
            )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_contract(raw: dict, expiration: str) -> OptionContract | None:
        """Parse a Tradier options contract dict into an OptionContract.

        Tradier JSON format::

            {
              "symbol": "GME250321C00025000",
              "option_type": "call",
              "strike": 25.0,
              "bid": 1.20,
              "ask": 1.35,
              "last": 1.28,
              "volume": 1234,
              "open_interest": 5678,
              "greeks": {
                "delta": 0.4521,
                "gamma": 0.0342,
                "theta": -0.0523,
                "vega": 0.0821,
                "mid_iv": 0.8523
              }
            }
        """
        try:
            opt_type_str = raw.get("option_type", "").lower()
            if opt_type_str == "call":
                opt_type = OptionType.CALL
            elif opt_type_str == "put":
                opt_type = OptionType.PUT
            else:
                return None

            # Extract Greeks from nested object
            greeks = raw.get("greeks") or {}

            # Use mid_iv for implied volatility (Tradier's midpoint IV)
            iv = greeks.get("mid_iv", 0) or 0
            delta_val = greeks.get("delta", 0) or 0
            gamma_val = greeks.get("gamma", 0) or 0
            theta_val = greeks.get("theta", 0) or 0
            vega_val = greeks.get("vega", 0) or 0

            # Determine ITM status
            strike = raw.get("strike", 0)
            last_price_raw = raw.get("last") or raw.get("close") or 0
            root_symbol = raw.get("root_symbol", "")

            return OptionContract(
                contract_symbol=raw.get("symbol", ""),
                type=opt_type,
                strike=_to_decimal(strike),
                expiration_date=expiration,
                last_price=_to_decimal(last_price_raw),
                bid=_to_decimal(raw.get("bid", 0)),
                ask=_to_decimal(raw.get("ask", 0)),
                volume=_to_int(raw.get("volume", 0)),
                open_interest=_to_int(raw.get("open_interest", 0)),
                implied_volatility=_to_decimal(iv),
                delta=_to_decimal(delta_val),
                gamma=_to_decimal(gamma_val),
                theta=_to_decimal(theta_val),
                vega=_to_decimal(vega_val),
                in_the_money=bool(raw.get("in_the_money", False)),
            )

        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed Tradier contract: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _symbol_dir(self, symbol: str) -> Path:
        """Return per-symbol data directory, creating it if needed."""
        d = self._data_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_chain_cache(self, chain: OptionsChain) -> None:
        """Persist an options chain to JSON cache for offline use."""
        symbol = chain.underlying_symbol
        cache_file = self._symbol_dir(symbol) / "options_chain.json"

        data = {
            "symbol": symbol,
            "underlying_price": str(chain.underlying_price),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "tradier_sandbox",
            "expirations": chain.expiration_dates,
            "total_calls": len(chain.all_calls),
            "total_puts": len(chain.all_puts),
            "total_call_oi": chain.total_call_open_interest,
            "total_put_oi": chain.total_put_open_interest,
            "has_greeks": any(
                float(c.gamma) != 0 for c in chain.all_calls
            ) if chain.all_calls else False,
        }

        try:
            cache_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Saved Tradier chain cache for %s", symbol)
        except OSError as exc:
            logger.debug("Failed to save Tradier cache: %s", exc)

    def load_chain_cache(self, symbol: str) -> dict | None:
        """Load cached chain metadata."""
        cache_file = self._symbol_dir(symbol) / "options_chain.json"
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _to_decimal(val: Any) -> Decimal:
    """Convert a value to Decimal, defaulting to 0."""
    if val is None:
        return Decimal(0)
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)


def _to_int(val: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
