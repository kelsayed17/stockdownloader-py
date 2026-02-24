"""Momentum signal generators: RSI, MACD, Stochastic, CCI, Williams %R."""

from __future__ import annotations

from decimal import Decimal
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


class RSISignalGenerator(AtomicSignalGenerator):
    """Score based on RSI position relative to oversold/overbought zones.

    Score mapping:
      RSI <= oversold:      +1.0 (maximally bullish)
      RSI >= overbought:    -1.0 (maximally bearish)
      Between zones:        linear interpolation

    Fires when RSI crosses above oversold or below overbought.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self._period = period
        self._oversold = Decimal(str(oversold))
        self._overbought = Decimal(str(overbought))

    @property
    def name(self) -> str:
        return f"rsi_{self._period}_{self._oversold}_{self._overbought}"

    @property
    def display_name(self) -> str:
        return f"RSI({self._period}) [{self._oversold}/{self._overbought}]"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(
        self,
        data: Sequence[PriceData],
        index: int,
        hub: IndicatorHub,
    ) -> SignalResult:
        if index < self._period + 1:
            return SignalResult.neutral()

        current_rsi = hub.rsi(data, index, self._period)
        prev_rsi = hub.rsi(data, index - 1, self._period)

        rsi_f = float(current_rsi)
        prev_f = float(prev_rsi)
        os_f = float(self._oversold)
        ob_f = float(self._overbought)

        if rsi_f <= os_f:
            score = 1.0
        elif rsi_f >= ob_f:
            score = -1.0
        else:
            midpoint = (os_f + ob_f) / 2.0
            half_range = (ob_f - os_f) / 2.0
            score = -(rsi_f - midpoint) / half_range if half_range > 0 else 0.0

        fired = False
        if rsi_f > os_f and prev_f <= os_f:
            fired = True
        elif rsi_f < ob_f and prev_f >= ob_f:
            fired = True

        direction = (
            SignalDirection.BULLISH if score > 0.1
            else SignalDirection.BEARISH if score < -0.1
            else SignalDirection.NEUTRAL
        )
        return SignalResult(
            score=score,
            direction=direction,
            fired=fired,
            metadata={"rsi": rsi_f, "prev_rsi": prev_f},
        )

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {
            "period": [7, 10, 14, 21],
            "oversold": [20.0, 25.0, 30.0, 35.0],
            "overbought": [65.0, 70.0, 75.0, 80.0],
        }


class MACDCrossoverGenerator(AtomicSignalGenerator):
    """Score based on MACD histogram normalized by ATR.  Fires on crossover."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self._fast = fast
        self._slow = slow
        self._signal = signal

    @property
    def name(self) -> str:
        return f"macd_{self._fast}_{self._slow}_{self._signal}"

    @property
    def display_name(self) -> str:
        return f"MACD({self._fast}/{self._slow}/{self._signal})"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return self._slow + self._signal + 1

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self.warmup_period:
            return SignalResult.neutral()

        line = float(hub.macd_line(data, index, self._fast, self._slow))
        sig = float(hub.macd_signal(data, index, self._fast, self._slow, self._signal))
        hist = line - sig

        prev_line = float(hub.macd_line(data, index - 1, self._fast, self._slow))
        prev_sig = float(hub.macd_signal(data, index - 1, self._fast, self._slow, self._signal))

        atr = float(hub.atr(data, index, 14))
        score = max(-1.0, min(1.0, hist / atr)) if atr > 0 else 0.0

        bullish_cross = line > sig and prev_line <= prev_sig
        bearish_cross = line < sig and prev_line >= prev_sig
        fired = bullish_cross or bearish_cross

        direction = (
            SignalDirection.BULLISH if score > 0.05
            else SignalDirection.BEARISH if score < -0.05
            else SignalDirection.NEUTRAL
        )
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"macd_line": line, "macd_signal": sig, "histogram": hist})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"fast": [8, 10, 12, 15], "slow": [20, 26, 30, 35], "signal": [5, 7, 9, 12]}


class StochasticGenerator(AtomicSignalGenerator):
    """Score based on Stochastic %K zone and K/D crossover."""

    def __init__(self, oversold: float = 20.0, overbought: float = 80.0) -> None:
        self._oversold = oversold
        self._overbought = overbought

    @property
    def name(self) -> str:
        return f"stoch_{self._oversold}_{self._overbought}"

    @property
    def display_name(self) -> str:
        return f"Stochastic [{self._oversold}/{self._overbought}]"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return 15

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < 15:
            return SignalResult.neutral()

        stoch = hub.stochastic(data, index)
        prev_stoch = hub.stochastic(data, index - 1)

        k = float(stoch.percent_k)
        d = float(stoch.percent_d)
        prev_k = float(prev_stoch.percent_k)
        prev_d = float(prev_stoch.percent_d)

        if k <= self._oversold:
            score = 1.0
        elif k >= self._overbought:
            score = -1.0
        else:
            mid = (self._oversold + self._overbought) / 2.0
            half = (self._overbought - self._oversold) / 2.0
            score = -(k - mid) / half if half > 0 else 0.0

        bullish_cross = k > d and prev_k <= prev_d and k < self._oversold + 10
        bearish_cross = k < d and prev_k >= prev_d and k > self._overbought - 10
        fired = bullish_cross or bearish_cross

        direction = (
            SignalDirection.BULLISH if score > 0.1
            else SignalDirection.BEARISH if score < -0.1
            else SignalDirection.NEUTRAL
        )
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"stoch_k": k, "stoch_d": d})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"oversold": [15.0, 20.0, 25.0], "overbought": [75.0, 80.0, 85.0]}


class CCIGenerator(AtomicSignalGenerator):
    """Score based on CCI zone (+-100/+-200).  Fires on threshold cross."""

    def __init__(self, period: int = 20, oversold: float = -100.0, overbought: float = 100.0) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def name(self) -> str:
        return f"cci_{self._period}_{self._oversold}_{self._overbought}"

    @property
    def display_name(self) -> str:
        return f"CCI({self._period}) [{self._oversold}/{self._overbought}]"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self._period + 1:
            return SignalResult.neutral()

        cci = float(hub.cci(data, index, self._period))
        prev_cci = float(hub.cci(data, index - 1, self._period))

        score = max(-1.0, min(1.0, -cci / 200.0))

        fired = (cci > self._oversold and prev_cci <= self._oversold) or \
                (cci < self._overbought and prev_cci >= self._overbought)

        direction = (SignalDirection.BULLISH if score > 0.1
                     else SignalDirection.BEARISH if score < -0.1
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"cci": cci})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [14, 20, 30], "oversold": [-150.0, -100.0, -50.0],
                "overbought": [50.0, 100.0, 150.0]}


class WilliamsRGenerator(AtomicSignalGenerator):
    """Score based on Williams %R zone.  Fires on threshold cross."""

    def __init__(self, period: int = 14, oversold: float = -80.0, overbought: float = -20.0) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def name(self) -> str:
        return f"willr_{self._period}_{self._oversold}_{self._overbought}"

    @property
    def display_name(self) -> str:
        return f"Williams %R({self._period}) [{self._oversold}/{self._overbought}]"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self._period + 1:
            return SignalResult.neutral()

        wr = float(hub.williams_r(data, index, self._period))
        prev_wr = float(hub.williams_r(data, index - 1, self._period))

        score = max(-1.0, min(1.0, -wr / 50.0 - 1.0))

        fired = (wr > self._oversold and prev_wr <= self._oversold) or \
                (wr < self._overbought and prev_wr >= self._overbought)

        direction = (SignalDirection.BULLISH if score > 0.1
                     else SignalDirection.BEARISH if score < -0.1
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"williams_r": wr})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [10, 14, 21], "oversold": [-85.0, -80.0, -75.0],
                "overbought": [-25.0, -20.0, -15.0]}
