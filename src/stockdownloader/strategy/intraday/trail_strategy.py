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
    from stockdownloader.model.price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.base_strategy import InfraExitConfig
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
    """Step-wise profit-locking trail — used by REV and PS trades.

    When breakeven is triggered, moves stop to entry +/- buffer.
    Then ratchets the stop as profit grows in R-multiple steps:
      - After 1.0R MFE → stop to entry + 0.5R
      - After 1.5R MFE → stop to entry + 1.0R
      - After 2.0R MFE → stop to entry + 1.5R
    The stop never retreats — only ratchets favorably.
    Returns True when the bar breaches the ratcheted stop.
    """

    # (mfe_threshold_R, lock_level_R)
    _STEPS: list[tuple[Decimal, Decimal]] = [
        (Decimal("2.0"), Decimal("1.5")),
        (Decimal("1.5"), Decimal("1.0")),
        (Decimal("1.0"), Decimal("0.5")),
    ]

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
        risk = state.risk_amount
        if risk <= ZERO:
            return False

        # Compute MFE in R-multiples from peak favorable excursion
        if is_long:
            mfe_r = (state.orb_extreme - entry) / risk
        else:
            mfe_r = (entry - state.orb_extreme) / risk

        # Walk the step table (highest threshold first) and lock profit
        for threshold_r, lock_r in self._STEPS:
            if mfe_r >= threshold_r:
                if is_long:
                    new_sl = entry + risk * lock_r
                    if new_sl > state.stop_loss:
                        state.stop_loss = new_sl
                else:
                    new_sl = entry - risk * lock_r
                    if new_sl < state.stop_loss:
                        state.stop_loss = new_sl
                break

        # Check if bar breaches the ratcheted stop
        if is_long:
            return bar.low <= state.stop_loss
        else:
            return bar.high >= state.stop_loss

    @property
    def exit_reason(self) -> str:
        return "breakeven"
