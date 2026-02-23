"""Streaming volume indicators: Session VWAP, Anchored VWAP, and CVD."""
from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.math import ZERO
from stockdownloader.util.streaming import _quantize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


class StreamingSessionVWAP:
    """Incremental Session VWAP with weighted standard deviation.

    Instead of scanning all session bars from scratch at each index (O(k)
    per bar, O(k^2) per session), this maintains running sums and updates
    in O(1) per bar.

    Uses the identity for weighted variance:

        Var = (Sum(TP^2 * Vol) / Sum(Vol)) - VWAP^2

    so only three running accumulators are needed, no stored bar lists.

    Results are **bit-identical** to
    :func:`~stockdownloader.util.technical_indicators._compute_session_vwap_core`
    when evaluated sequentially.
    """

    __slots__ = (
        "_sum_tpv", "_sum_vol", "_sum_tp2v",
        "_current_session", "_last_index",
        "_history",  # list of (vwap, std_dev) tuples per index
    )

    def __init__(self) -> None:
        self._sum_tpv: Decimal = ZERO
        self._sum_vol: Decimal = ZERO
        self._sum_tp2v: Decimal = ZERO
        self._current_session: str = ""
        self._last_index: int = -1
        self._history: list[tuple[Decimal, Decimal]] = []

    def reset(self) -> None:
        self._sum_tpv = ZERO
        self._sum_vol = ZERO
        self._sum_tp2v = ZERO
        self._current_session = ""
        self._last_index = -1
        self._history.clear()

    def update(
        self, data: Sequence[PriceData], index: int,
    ) -> tuple[Decimal, Decimal]:
        """Return ``(vwap, std_dev)`` at *index*, updating incrementally.

        Fills forward from ``_last_index + 1`` to *index* if there are gaps.
        Resets accumulators on session boundary (new trading day).
        """
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO, ZERO

        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            session = bar.date[:10]

            # Session boundary detection -- reset accumulators
            if session != self._current_session:
                self._sum_tpv = ZERO
                self._sum_vol = ZERO
                self._sum_tp2v = ZERO
                self._current_session = session

            # Typical price
            tp = _quantize(
                (bar.high + bar.low + bar.close) / Decimal("3")
            )
            vol = Decimal(str(bar.volume))

            # Update running sums
            self._sum_tpv += tp * vol
            self._sum_vol += vol
            self._sum_tp2v += tp * tp * vol

            # Compute VWAP and std dev
            if self._sum_vol == ZERO:
                vwap_val = ZERO
                std_val = ZERO
            else:
                vwap_val = _quantize(self._sum_tpv / self._sum_vol)

                # Weighted variance: E[X^2] - (E[X])^2
                mean_tp2 = float(self._sum_tp2v / self._sum_vol)
                vwap_f = float(vwap_val)
                variance = max(0.0, mean_tp2 - vwap_f * vwap_f)
                std_val = _quantize(Decimal(str(math.sqrt(variance))))

            self._history.append((vwap_val, std_val))

        self._last_index = index
        return self._history[index]


class StreamingAnchoredVWAP:
    """Incremental Anchored VWAP that resets on event dates, not sessions.

    Mirrors :class:`StreamingSessionVWAP` but resets accumulators when the
    most recent anchor date changes (e.g., a new FOMC meeting) rather than
    on every new trading day.

    The ``_valid`` flag stays ``False`` until the first anchor date is
    encountered in the data, at which point accumulation begins.

    Results are **bit-identical** to
    :func:`~stockdownloader.util.technical_indicators._compute_session_vwap_core`
    when the anchor start index corresponds to the core function's start.
    """

    __slots__ = (
        "_sum_tpv", "_sum_vol", "_sum_tp2v",
        "_current_anchor", "_last_index",
        "_last_date",      # cache for per-day anchor lookups
        "_history",        # list of (vwap, std_dev) tuples per index
        "_anchor_type",
        "_valid",          # False until the first anchor is hit
    )

    def __init__(self, anchor_type: str = "fomc") -> None:
        self._sum_tpv: Decimal = ZERO
        self._sum_vol: Decimal = ZERO
        self._sum_tp2v: Decimal = ZERO
        self._current_anchor: str = ""
        self._last_index: int = -1
        self._last_date: str = ""
        self._history: list[tuple[Decimal, Decimal]] = []
        self._anchor_type: str = anchor_type
        self._valid: bool = False

    def reset(self) -> None:
        self._sum_tpv = ZERO
        self._sum_vol = ZERO
        self._sum_tp2v = ZERO
        self._current_anchor = ""
        self._last_index = -1
        self._last_date = ""
        self._history.clear()
        self._valid = False

    @property
    def current_anchor(self) -> str:
        """The anchor date string currently in use (empty if none yet)."""
        return self._current_anchor

    @property
    def valid(self) -> bool:
        """Whether the accumulator has reached its first anchor."""
        return self._valid

    def update(
        self, data: Sequence[PriceData], index: int,
    ) -> tuple[Decimal, Decimal]:
        """Return ``(avwap, std_dev)`` at *index*, updating incrementally.

        Fills forward from ``_last_index + 1`` to *index* if there are gaps.
        Resets accumulators when the anchor date changes (new event).
        Returns ``(ZERO, ZERO)`` while ``_valid`` is ``False``.
        """
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO, ZERO

        # Lazy import to avoid circular dependency
        from stockdownloader.util.config import get_anchor_date

        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            trading_date = bar.date[:10]

            # Only re-lookup anchor when the trading date changes
            if trading_date != self._last_date:
                anchor = get_anchor_date(trading_date, self._anchor_type)
                if anchor is not None and anchor != self._current_anchor:
                    # New anchor event -- reset accumulators
                    self._sum_tpv = ZERO
                    self._sum_vol = ZERO
                    self._sum_tp2v = ZERO
                    self._current_anchor = anchor
                    self._valid = True
                self._last_date = trading_date

            if not self._valid:
                self._history.append((ZERO, ZERO))
                continue

            # Typical price
            tp = _quantize(
                (bar.high + bar.low + bar.close) / Decimal("3")
            )
            vol = Decimal(str(bar.volume))

            # Update running sums
            self._sum_tpv += tp * vol
            self._sum_vol += vol
            self._sum_tp2v += tp * tp * vol

            # Compute VWAP and std dev
            if self._sum_vol == ZERO:
                vwap_val = ZERO
                std_val = ZERO
            else:
                vwap_val = _quantize(self._sum_tpv / self._sum_vol)

                # Weighted variance: E[X^2] - (E[X])^2
                mean_tp2 = float(self._sum_tp2v / self._sum_vol)
                vwap_f = float(vwap_val)
                variance = max(0.0, mean_tp2 - vwap_f * vwap_f)
                std_val = _quantize(Decimal(str(math.sqrt(variance))))

            self._history.append((vwap_val, std_val))

        self._last_index = index
        return self._history[index]


class StreamingCVD:
    """Incremental cumulative volume delta for intraday sessions.

    Replaces the O(k)-per-bar :func:`cvd_session` with an O(1) running
    sum that resets at each session boundary.
    """

    __slots__ = ("_cum", "_current_session", "_last_index", "_history")

    def __init__(self) -> None:
        self._cum: Decimal = ZERO
        self._current_session: str = ""
        self._last_index: int = -1
        self._history: list[Decimal] = []

    def reset(self) -> None:
        self._cum = ZERO
        self._current_session = ""
        self._last_index = -1
        self._history.clear()

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Return session CVD at *index*, updating incrementally."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO

        _half = Decimal("0.5")
        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            session = bar.date[:10]

            if session != self._current_session:
                self._cum = ZERO
                self._current_session = session

            bar_range = bar.high - bar.low
            if bar_range > ZERO:
                v_delta = (bar.close - bar.low) / bar_range
            else:
                v_delta = _half
            self._cum += (v_delta - _half) * Decimal(str(bar.volume))

            self._history.append(_quantize(self._cum))

        self._last_index = index
        return self._history[index]
