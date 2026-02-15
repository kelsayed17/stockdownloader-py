"""Trail strategy implementations for intraday exit management.

Uses the Strategy pattern to replace hard-coded trail type branching
in the exit manager. Each strategy knows how to activate and ratchet
its trailing stop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.base_config import InfraExitConfig
    from stockdownloader.strategy.intraday.session_state import SessionState
    from stockdownloader.util.intraday_indicators import ExtendedSessionVWAP

_BE_BUF = Decimal("0.05")


class TrailStrategy(ABC):
    """Abstract base for trailing stop strategies."""

    @abstractmethod
    def activate(
        self,
        state: SessionState,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> None:
        """Activate the trail when breakeven is triggered.

        Must set ``state.trail_level`` and any trailing flags.
        """

    @abstractmethod
    def ratchet(
        self,
        state: SessionState,
        bar: IntradayPriceData,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> bool:
        """Ratchet the trail per-bar. Return True if trail stop is hit."""

    @property
    @abstractmethod
    def exit_reason(self) -> str:
        """Reason string for exit signals (e.g. 'atr_trail')."""


class AtrChandelierTrail(TrailStrategy):
    """ATR chandelier trailing stop — used by ORB trades.

    Trail level = peak favorable excursion - ATR * coefficient.
    Ratchets up (long) / down (short) as price moves favorably.
    """

    def activate(
        self,
        state: SessionState,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> None:
        state.trailing_atr = True
        if is_long:
            state.trail_level = max(
                state.orb_extreme - atr_val * config.orb_trail_atr,
                entry + _BE_BUF,
            )
        else:
            state.trail_level = min(
                state.orb_extreme + atr_val * config.orb_trail_atr,
                entry - _BE_BUF,
            )

    def ratchet(
        self,
        state: SessionState,
        bar: IntradayPriceData,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> bool:
        if not state.trailing_atr:
            return False

        if is_long:
            new_trail = max(
                state.orb_extreme - atr_val * config.orb_trail_atr,
                entry + _BE_BUF,
            )
            new_trail = max(new_trail, state.trail_level)
            if new_trail > state.trail_level:
                state.trail_level = new_trail
                state.stop_loss = new_trail
            return bar.low <= state.trail_level
        else:
            new_trail = min(
                state.orb_extreme + atr_val * config.orb_trail_atr,
                entry - _BE_BUF,
            )
            new_trail = min(new_trail, state.trail_level)
            if new_trail < state.trail_level:
                state.trail_level = new_trail
                state.stop_loss = new_trail
            return bar.high >= state.trail_level

    @property
    def exit_reason(self) -> str:
        return "atr_trail"


class VwapRatchetTrail(TrailStrategy):
    """VWAP ratcheting trailing stop — used by PB and ORR trades.

    Trail level tracks VWAP minus/plus a buffer. Ratchets only in
    the favorable direction (never retreats).
    """

    def activate(
        self,
        state: SessionState,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> None:
        state.trailing_vwap = True
        vwap = vwap_bands.vwap
        buf = atr_val * config.trail_buf
        if is_long:
            state.trail_level = max(vwap - buf, entry + _BE_BUF)
        else:
            state.trail_level = min(vwap + buf, entry - _BE_BUF)

    def ratchet(
        self,
        state: SessionState,
        bar: IntradayPriceData,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> bool:
        if not state.trailing_vwap:
            return False

        vwap = vwap_bands.vwap
        buf = atr_val * config.trail_buf

        if is_long:
            new_trail = max(vwap - buf, entry + _BE_BUF)
            new_trail = max(new_trail, state.trail_level)
            if new_trail > state.trail_level:
                state.trail_level = new_trail
                state.stop_loss = new_trail
                if not config.trail_keep_tp:
                    state.take_profit = ZERO
            return bar.low <= state.trail_level
        else:
            new_trail = min(vwap + buf, entry - _BE_BUF)
            new_trail = min(new_trail, state.trail_level)
            if new_trail < state.trail_level:
                state.trail_level = new_trail
                state.stop_loss = new_trail
                if not config.trail_keep_tp:
                    state.take_profit = ZERO
            return bar.high >= state.trail_level

    @property
    def exit_reason(self) -> str:
        return "vwap_trail"


class BreakevenTrail(TrailStrategy):
    """Simple breakeven stop — used by REV and PS trades.

    When breakeven is triggered, moves stop to entry +/- buffer.
    No further ratcheting.
    """

    def activate(
        self,
        state: SessionState,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> None:
        if is_long:
            state.stop_loss = entry + _BE_BUF
        else:
            state.stop_loss = entry - _BE_BUF

    def ratchet(
        self,
        state: SessionState,
        bar: IntradayPriceData,
        is_long: bool,
        entry: Decimal,
        atr_val: Decimal,
        vwap_bands: ExtendedSessionVWAP,
        config: InfraExitConfig,
    ) -> bool:
        # Simple breakeven has no per-bar ratcheting.
        # The hard SL check in the exit manager handles the stop.
        return False

    @property
    def exit_reason(self) -> str:
        return "breakeven"
