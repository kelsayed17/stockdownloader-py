"""Volatility signal generators: Bollinger Band Touch and Squeeze."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stockdownloader.core.models.price import PriceData
    from stockdownloader.util.indicators.hub import IndicatorHub


class BBTouchGenerator(AtomicSignalGenerator):
    """Score based on Bollinger Band %B.  Fires on band touch.

    %B < 0 (below lower band) -> +1.0 bullish (mean reversion)
    %B > 1 (above upper band) -> -1.0 bearish
    %B = 0.5 (at middle)      ->  0.0 neutral
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        self._period = period
        self._std_dev = std_dev

    @property
    def name(self) -> str:
        return f"bb_touch_{self._period}_{self._std_dev}"

    @property
    def display_name(self) -> str:
        return f"BB Touch({self._period}, {self._std_dev}\u03c3)"

    @property
    def category(self) -> str:
        return "volatility"

    @property
    def warmup_period(self) -> int:
        return self._period

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self._period:
            return SignalResult.neutral()

        bb = hub.bollinger_bands(data, index, self._period, self._std_dev)
        close = float(data[index].close)

        bw = float(bb.upper - bb.lower)
        if bw <= 0:
            return SignalResult.neutral()

        pct_b = (close - float(bb.lower)) / bw

        score = max(-1.0, min(1.0, 1.0 - 2.0 * pct_b))

        fired = pct_b <= 0.0 or pct_b >= 1.0

        direction = (
            SignalDirection.BULLISH if score > 0.1
            else SignalDirection.BEARISH if score < -0.1
            else SignalDirection.NEUTRAL
        )
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"pct_b": pct_b, "close": close,
                                      "upper": float(bb.upper), "lower": float(bb.lower)})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [15, 20, 25], "std_dev": [1.5, 2.0, 2.5]}


class BBSqueezeGenerator(AtomicSignalGenerator):
    """Detects Bollinger Band squeeze and expansion breakouts.

    Narrow bandwidth = squeeze (coiling volatility).
    Expansion above threshold = breakout.  Direction from price vs middle band.
    """

    def __init__(
        self, period: int = 20, squeeze_lookback: int = 120,
    ) -> None:
        self._period = period
        self._lookback = squeeze_lookback

    @property
    def name(self) -> str:
        return f"bb_squeeze_{self._period}_{self._lookback}"

    @property
    def display_name(self) -> str:
        return f"BB Squeeze({self._period}, lb={self._lookback})"

    @property
    def category(self) -> str:
        return "volatility"

    @property
    def warmup_period(self) -> int:
        return max(self._period, self._lookback) + 1

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self.warmup_period:
            return SignalResult.neutral()

        bb = hub.bollinger_bands(data, index, self._period)
        prev_bb = hub.bollinger_bands(data, index - 1, self._period)

        bw = float(bb.width)
        prev_bw = float(prev_bb.width)

        min_bw = bw
        for j in range(max(0, index - self._lookback), index):
            past_bb = hub.bollinger_bands(data, j, self._period)
            past_bw = float(past_bb.width)
            if past_bw < min_bw:
                min_bw = past_bw

        is_expanding = bw > prev_bw and prev_bw <= min_bw * 1.1
        close = float(data[index].close)
        mid = float(bb.middle)

        if is_expanding:
            score = 1.0 if close > mid else -1.0
            fired = True
        else:
            score = 0.0
            fired = False

        direction = (
            SignalDirection.BULLISH if score > 0.05
            else SignalDirection.BEARISH if score < -0.05
            else SignalDirection.NEUTRAL
        )
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"bw": bw, "min_bw": min_bw, "expanding": is_expanding})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [15, 20, 25], "squeeze_lookback": [60, 90, 120, 150]}
