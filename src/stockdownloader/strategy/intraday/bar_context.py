"""Computed values for the current bar, passed to entry logic.

:class:`BarContext` is a pure data transfer object with no business logic.
It aggregates all indicator values, session state, and derived flags
that entry functions need to decide whether to trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.session_state import SessionState
    from stockdownloader.util.intraday_indicators import (
        AnchoredVWAPBands,
        ExtendedSessionVWAP,
    )
    from stockdownloader.util.smc_indicators import StructureState


@dataclass(slots=True)
class BarContext:
    """All computed values for the current bar, passed to entry logic."""

    bar: IntradayPriceData
    prev_bar: IntradayPriceData | None
    state: SessionState
    bar_of_day: int
    dow: int
    # Indicators
    atr_val: Decimal
    atr_fast: Decimal
    adx_val: Decimal
    rsi_val: Decimal
    ema_fast: Decimal
    ema_slow: Decimal
    htf_trend: int
    cvd_norm: Decimal
    lrs_atr: Decimal
    rel_vol: Decimal
    tod_rvol: Decimal
    # VWAP
    vwap_bands: ExtendedSessionVWAP
    vwap_delta: Decimal
    vwap_accel: Decimal
    # AVWAP (anchored, persists across sessions — None when not used)
    avwap_bands: AnchoredVWAPBands | None
    # Market structure (SMC — None when not used)
    structure: StructureState | None
    # Derived
    sr_any: bool
    sr_score_count: int
    box_pos: Decimal
    clean_pb: bool
    is_good_time: bool
