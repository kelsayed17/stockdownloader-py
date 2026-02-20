"""Per-symbol metadata registry.

Provides IPO dates, CUSIPs, and other static metadata for well-known
symbols.  Data clients use :func:`get_symbol_info` to auto-resolve
CUSIPs and compute sensible start dates for historical data fetches.

Unknown symbols return ``None`` — callers fall back to their existing
defaults (e.g. ``start_year=2004`` for FTD data).

Usage::

    from stockdownloader.model.symbol_info import get_symbol_info

    info = get_symbol_info("GME")
    if info is not None:
        print(info.cusip)           # "36467W109"
        print(info.ftd_start_year)  # 2004  (IPO 2002 but FTD floor is 2004)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    """Static metadata for a traded symbol.

    Attributes
    ----------
    symbol:
        Upper-cased ticker (e.g. ``"GME"``).
    cusip:
        9-character CUSIP identifier.  Empty string if unknown.
    ipo_date:
        Date the symbol first traded on a US exchange.
    name:
        Human-readable company name.
    exchange:
        Primary exchange (e.g. ``"NYSE"``, ``"NASDAQ"``).
        Empty string if unknown.
    """

    symbol: str
    cusip: str
    ipo_date: date
    name: str
    exchange: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")

    @property
    def ipo_year(self) -> int:
        """Calendar year of the IPO."""
        return self.ipo_date.year

    @property
    def ftd_start_year(self) -> int:
        """Earliest useful year for FTD data (floored at 2004).

        SEC FTD data begins Q1 2004 — no earlier data exists.
        """
        return max(self.ipo_date.year, 2004)

    @property
    def ownership_start_year(self) -> int:
        """Earliest useful year for 13F ownership data (floored at 2003).

        SEC bulk 13F data sets begin Q2 2013, but EFTS full-text search
        can find 13F-HR filings back to ~2001.  The legacy text parser
        handles pre-2013 non-XML formats.  We use 2003 as the floor
        because earlier filings are sparse and unreliable.
        """
        return max(self.ipo_date.year, 2003)


# ------------------------------------------------------------------
# Registry of well-known symbols
# ------------------------------------------------------------------

SYMBOL_REGISTRY: dict[str, SymbolInfo] = {}


def _register(*infos: SymbolInfo) -> None:
    """Add one or more :class:`SymbolInfo` entries to the registry."""
    for info in infos:
        SYMBOL_REGISTRY[info.symbol] = info


_register(
    SymbolInfo(
        symbol="GME",
        cusip="36467W109",
        ipo_date=date(2002, 2, 13),
        name="GameStop Corp",
        exchange="NYSE",
    ),
    SymbolInfo(
        symbol="AAPL",
        cusip="037833100",
        ipo_date=date(1980, 12, 12),
        name="Apple Inc",
        exchange="NASDAQ",
    ),
    SymbolInfo(
        symbol="TSLA",
        cusip="88160R101",
        ipo_date=date(2010, 6, 29),
        name="Tesla Inc",
        exchange="NASDAQ",
    ),
    SymbolInfo(
        symbol="AMZN",
        cusip="023135106",
        ipo_date=date(1997, 5, 15),
        name="Amazon.com Inc",
        exchange="NASDAQ",
    ),
    SymbolInfo(
        symbol="GOOGL",
        cusip="02079K305",
        ipo_date=date(2004, 8, 19),
        name="Alphabet Inc Class A",
        exchange="NASDAQ",
    ),
    SymbolInfo(
        symbol="GOOG",
        cusip="02079K107",
        ipo_date=date(2004, 8, 19),
        name="Alphabet Inc Class C",
        exchange="NASDAQ",
    ),
    SymbolInfo(
        symbol="NVDA",
        cusip="67066G104",
        ipo_date=date(1999, 1, 22),
        name="NVIDIA Corp",
        exchange="NASDAQ",
    ),
    SymbolInfo(
        symbol="SPY",
        cusip="78462F103",
        ipo_date=date(1993, 1, 29),
        name="SPDR S&P 500 ETF Trust",
        exchange="NYSE",
    ),
)


def get_symbol_info(symbol: str) -> SymbolInfo | None:
    """Look up metadata for *symbol*.

    Returns ``None`` for unknown symbols — callers should fall back
    to their existing defaults.
    """
    return SYMBOL_REGISTRY.get(symbol.upper())
