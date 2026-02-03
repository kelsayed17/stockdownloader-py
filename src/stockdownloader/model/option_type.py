"""Option contract type: CALL gives the right to buy, PUT gives the right to sell."""

from __future__ import annotations

from enum import Enum


class OptionType(Enum):
    """Option contract type: CALL gives the right to buy, PUT gives the right to sell."""

    CALL = "CALL"
    PUT = "PUT"
