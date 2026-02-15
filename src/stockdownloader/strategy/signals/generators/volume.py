"""Volume signal generators: OBV, MFI, Volume Surge."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stockdownloader.model.price_data import PriceData
    from stockdownloader.util.indicator_hub import IndicatorHub


class OBVGenerator(AtomicSignalGenerator):
    """Score based on OBV trend (accumulation vs distribution)."""

    def __init__(self, lookback: int = 5) -> None:
        self._lookback = lookback

    @property
    def name(self) -> str:
        return f"obv_{self._lookback}"

    @property
    def display_name(self) -> str:
        return f"OBV({self._lookback})"

    @property
    def category(self) -> str:
        return "volume"

    @property
    def warmup_period(self) -> int:
        return self._lookback + 2

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self._lookback + 1:
            return SignalResult.neutral()

        rising = hub.is_obv_rising(data, index, self._lookback)
        prev_rising = hub.is_obv_rising(data, index - 1, self._lookback)

        score = 0.6 if rising else -0.6
        fired = rising != prev_rising

        direction = SignalDirection.BULLISH if rising else SignalDirection.BEARISH
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"obv_rising": rising})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"lookback": [3, 5, 10]}


class MFIGenerator(AtomicSignalGenerator):
    """Score based on MFI zone (money flow accumulation/distribution)."""

    def __init__(self, period: int = 14, oversold: float = 20.0, overbought: float = 80.0) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def name(self) -> str:
        return f"mfi_{self._period}_{self._oversold}_{self._overbought}"

    @property
    def display_name(self) -> str:
        return f"MFI({self._period}) [{self._oversold}/{self._overbought}]"

    @property
    def category(self) -> str:
        return "volume"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self._period + 1:
            return SignalResult.neutral()

        mfi = float(hub.mfi(data, index, self._period))
        prev_mfi = float(hub.mfi(data, index - 1, self._period))

        if mfi <= self._oversold:
            score = 1.0
        elif mfi >= self._overbought:
            score = -1.0
        else:
            mid = (self._oversold + self._overbought) / 2.0
            half = (self._overbought - self._oversold) / 2.0
            score = -(mfi - mid) / half if half > 0 else 0.0

        fired = (mfi > self._oversold and prev_mfi <= self._oversold) or \
                (mfi < self._overbought and prev_mfi >= self._overbought)

        direction = (SignalDirection.BULLISH if score > 0.1
                     else SignalDirection.BEARISH if score < -0.1
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"mfi": mfi})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [10, 14, 21], "oversold": [15.0, 20.0, 25.0],
                "overbought": [75.0, 80.0, 85.0]}


class VolumeSurgeGenerator(AtomicSignalGenerator):
    """Score based on current volume vs average volume ratio.

    High relative volume confirms conviction.  Direction from price change.
    """

    def __init__(self, period: int = 20, multiplier: float = 1.5) -> None:
        self._period = period
        self._multiplier = multiplier

    @property
    def name(self) -> str:
        return f"vol_surge_{self._period}_{self._multiplier}"

    @property
    def display_name(self) -> str:
        return f"Volume Surge({self._period}, \u00d7{self._multiplier})"

    @property
    def category(self) -> str:
        return "volume"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self._period + 1:
            return SignalResult.neutral()

        avg_vol = float(hub.average_volume(data, index, self._period))
        current_vol = float(data[index].volume)

        if avg_vol <= 0:
            return SignalResult.neutral()

        rvol = current_vol / avg_vol
        is_surge = rvol >= self._multiplier

        close = float(data[index].close)
        prev_close = float(data[index - 1].close)
        price_up = close > prev_close

        if is_surge:
            score = min(1.0, rvol / 3.0) if price_up else max(-1.0, -rvol / 3.0)
        else:
            score = 0.0

        direction = (SignalDirection.BULLISH if score > 0.05
                     else SignalDirection.BEARISH if score < -0.05
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=is_surge,
                            metadata={"rvol": rvol, "avg_vol": avg_vol, "volume": current_vol})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [10, 20, 30], "multiplier": [1.2, 1.5, 2.0, 2.5]}
