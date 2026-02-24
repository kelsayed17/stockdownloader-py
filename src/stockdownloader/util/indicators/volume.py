"""Volume indicators: Session VWAP, Anchored VWAP, CVD.

Section 1 — Data structures (SessionVWAP, ExtendedSessionVWAP, AnchoredVWAPBands)
Section 2 — Batch functions (session_vwap, session_vwap_bands, extended_session_vwap_bands, cvd_session, cvd_normalized)
Section 3 — Streaming classes (StreamingSessionVWAP, StreamingAnchoredVWAP, StreamingCVD)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import THREE, ZERO, quantize
from stockdownloader.util.indicators._core import (
    _compute_session_vwap_core,
    _find_session_start,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.core.models.price import PriceData

_HALF = Decimal("0.5")

__all__ = [
    # Data structures
    "SessionVWAP",
    "ExtendedSessionVWAP",
    "AnchoredVWAPBands",
    # Batch
    "session_vwap",
    "session_vwap_bands",
    "extended_session_vwap_bands",
    "cvd_session",
    "cvd_normalized",
    # Streaming
    "StreamingSessionVWAP",
    "StreamingAnchoredVWAP",
    "StreamingCVD",
]


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass(frozen=True, slots=True)
class SessionVWAP:
    """Session VWAP with standard-deviation bands."""

    vwap: Decimal
    std_dev: Decimal
    upper_1: Decimal
    lower_1: Decimal
    upper_05: Decimal
    lower_05: Decimal


@dataclass(frozen=True, slots=True)
class ExtendedSessionVWAP:
    """Session VWAP with 0.5 / 1 / 1.5 / 2 / 3 sigma bands."""

    vwap: Decimal
    std_dev: Decimal
    upper_05: Decimal
    lower_05: Decimal
    upper_1: Decimal
    lower_1: Decimal
    upper_15: Decimal
    lower_15: Decimal
    upper_2: Decimal
    lower_2: Decimal
    upper_3: Decimal
    lower_3: Decimal

    def band_pair(self, label: str = "2\u03c3") -> tuple[Decimal, Decimal]:
        """Return ``(upper, lower)`` band pair for the given sigma label.

        Supported labels: ``'0.5\u03c3'``, ``'1\u03c3'``, ``'1.5\u03c3'``, ``'2\u03c3'``, ``'3\u03c3'``.
        Defaults to ``'2\u03c3'`` if the label is unrecognized.
        """
        suffix_map: dict[str, str] = {
            "0.5\u03c3": "_05",
            "1\u03c3": "_1",
            "1.5\u03c3": "_15",
            "2\u03c3": "_2",
            "3\u03c3": "_3",
        }
        sfx = suffix_map.get(label, "_2")
        return (
            getattr(self, f"upper{sfx}"),
            getattr(self, f"lower{sfx}"),
        )

_EMPTY_VWAP = ExtendedSessionVWAP(
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
)


@dataclass(frozen=True, slots=True)
class AnchoredVWAPBands:
    """Anchored VWAP with sigma bands and event metadata.

    Unlike :class:`ExtendedSessionVWAP`, this persists across trading days
    and tracks the anchor event that initiated the accumulation.
    """

    avwap: Decimal
    std_dev: Decimal
    upper_1: Decimal          # +1 sigma
    lower_1: Decimal          # -1 sigma
    upper_2: Decimal          # +2 sigma
    lower_2: Decimal          # -2 sigma
    anchor_date: str          # The anchor date this AVWAP started from
    days_since_anchor: int    # Calendar days since anchor
    valid: bool               # False before first anchor


_EMPTY_AVWAP = AnchoredVWAPBands(
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
    "", 0, False,
)


# =========================================================================
# BATCH — SESSION VWAP
# =========================================================================

def session_vwap(
    data: Sequence[PriceData], end_index: int
) -> Decimal:
    """Calculate cumulative session VWAP, resetting at each new trading day.

    Walks backward from *end_index* to find the session start, then
    cumulates ``TP * Volume / Volume`` forward.
    """
    vwap_val, _ = _compute_session_vwap_core(data, end_index)
    return vwap_val

def session_vwap_bands(
    data: Sequence[PriceData], end_index: int
) -> SessionVWAP:
    """Calculate session VWAP with standard-deviation bands.

    sigma = sqrt( Sum((TP - VWAP)^2 * Volume) / Sum(Volume) )

    Returns a :class:`SessionVWAP` with VWAP, std_dev and +-1/+-0.5 sigma
    bands.
    """
    vwap_val, std_val = _compute_session_vwap_core(data, end_index)

    if vwap_val == ZERO and std_val == ZERO:
        return SessionVWAP(ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)

    half_std = quantize(std_val * Decimal('0.5'))

    return SessionVWAP(
        vwap=vwap_val,
        std_dev=std_val,
        upper_1=quantize(vwap_val + std_val),
        lower_1=quantize(vwap_val - std_val),
        upper_05=quantize(vwap_val + half_std),
        lower_05=quantize(vwap_val - half_std),
    )


# =========================================================================
# BATCH — EXTENDED SESSION VWAP BANDS
# =========================================================================

def extended_session_vwap_bands(
    data: Sequence[PriceData], end_index: int
) -> ExtendedSessionVWAP:
    """Session VWAP with 0.5 / 1 / 1.5 / 2 / 3 sigma bands.

    Delegates core VWAP computation to
    :func:`~stockdownloader.util.indicators._core._compute_session_vwap_core`.
    """
    if end_index < 0:
        return _EMPTY_VWAP

    vwap_val, std_val = _compute_session_vwap_core(data, end_index)

    if vwap_val == ZERO and std_val == ZERO:
        return _EMPTY_VWAP

    s05 = quantize(std_val * Decimal("0.5"))
    s15 = quantize(std_val * Decimal("1.5"))
    s2 = quantize(std_val * Decimal("2"))
    s3 = quantize(std_val * THREE)

    return ExtendedSessionVWAP(
        vwap=vwap_val,
        std_dev=std_val,
        upper_05=quantize(vwap_val + s05),
        lower_05=quantize(vwap_val - s05),
        upper_1=quantize(vwap_val + std_val),
        lower_1=quantize(vwap_val - std_val),
        upper_15=quantize(vwap_val + s15),
        lower_15=quantize(vwap_val - s15),
        upper_2=quantize(vwap_val + s2),
        lower_2=quantize(vwap_val - s2),
        upper_3=quantize(vwap_val + s3),
        lower_3=quantize(vwap_val - s3),
    )


# =========================================================================
# BATCH — CUMULATIVE VOLUME DELTA PROXY
# =========================================================================

def cvd_session(
    data: Sequence[PriceData],
    end_index: int,
) -> Decimal:
    """Cumulative volume delta proxy for the current session.

    For each bar: ``vDelta = (close - low) / (high - low)`` (clamped to 0.5
    when range is zero).  Session CVD = ``Sum((vDelta - 0.5) * volume)``.

    This is the same approximation as the PineScript ``cumVD``.
    """
    if end_index < 0:
        return ZERO

    session_start = _find_session_start(data, end_index)
    cum = ZERO

    for i in range(session_start, end_index + 1):
        bar = data[i]
        bar_range = bar.high - bar.low
        if bar_range > ZERO:
            v_delta = (bar.close - bar.low) / bar_range
        else:
            v_delta = _HALF
        cum += (v_delta - _HALF) * Decimal(str(bar.volume))

    return quantize(cum)

def cvd_normalized(
    data: Sequence[PriceData],
    end_index: int,
    vol_sma_period: int = 20,
) -> Decimal:
    """CVD normalized by ``volSMA * 20`` (PineScript ``cumVDnorm``)."""
    cvd = cvd_session(data, end_index)

    if end_index < vol_sma_period:
        return ZERO

    total = ZERO
    for i in range(end_index - vol_sma_period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    vol_sma = total / Decimal(str(vol_sma_period))

    denom = vol_sma * Decimal("20")
    if denom <= ZERO:
        return ZERO

    return quantize(cvd / denom)


# =========================================================================
# STREAMING — SESSION VWAP
# =========================================================================

class StreamingSessionVWAP:
    """Incremental Session VWAP with weighted standard deviation.

    Instead of scanning all session bars from scratch at each index (O(k)
    per bar, O(k^2) per session), this maintains running sums and updates
    in O(1) per bar.

    Uses the identity for weighted variance:

        Var = (Sum(TP^2 * Vol) / Sum(Vol)) - VWAP^2

    so only three running accumulators are needed, no stored bar lists.

    Results are **bit-identical** to
    :func:`~stockdownloader.util.indicators._core._compute_session_vwap_core`
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
            tp = quantize(
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
                vwap_val = quantize(self._sum_tpv / self._sum_vol)

                # Weighted variance: E[X^2] - (E[X])^2
                mean_tp2 = float(self._sum_tp2v / self._sum_vol)
                vwap_f = float(vwap_val)
                variance = max(0.0, mean_tp2 - vwap_f * vwap_f)
                std_val = quantize(Decimal(str(math.sqrt(variance))))

            self._history.append((vwap_val, std_val))

        self._last_index = index
        return self._history[index]


# =========================================================================
# STREAMING — ANCHORED VWAP
# =========================================================================

class StreamingAnchoredVWAP:
    """Incremental Anchored VWAP that resets on event dates, not sessions.

    Mirrors :class:`StreamingSessionVWAP` but resets accumulators when the
    most recent anchor date changes (e.g., a new FOMC meeting) rather than
    on every new trading day.

    The ``_valid`` flag stays ``False`` until the first anchor date is
    encountered in the data, at which point accumulation begins.

    Results are **bit-identical** to
    :func:`~stockdownloader.util.indicators._core._compute_session_vwap_core`
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
        from stockdownloader.core.config import get_anchor_date

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
            tp = quantize(
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
                vwap_val = quantize(self._sum_tpv / self._sum_vol)

                # Weighted variance: E[X^2] - (E[X])^2
                mean_tp2 = float(self._sum_tp2v / self._sum_vol)
                vwap_f = float(vwap_val)
                variance = max(0.0, mean_tp2 - vwap_f * vwap_f)
                std_val = quantize(Decimal(str(math.sqrt(variance))))

            self._history.append((vwap_val, std_val))

        self._last_index = index
        return self._history[index]


# =========================================================================
# STREAMING — CVD
# =========================================================================

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

            self._history.append(quantize(self._cum))

        self._last_index = index
        return self._history[index]
