"""Trend signal generators: SMA Crossover, ADX, EMA, Ichimoku, SAR, VWAP, SMA Position."""

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
    from stockdownloader.indicators.hub import IndicatorHub


class SMACrossoverGenerator(AtomicSignalGenerator):
    """Score based on SMA fast/slow difference.  Fires on golden/death cross."""

    def __init__(self, short_period: int = 9, long_period: int = 21) -> None:
        self._short = short_period
        self._long = long_period

    @property
    def name(self) -> str:
        return f"sma_cross_{self._short}_{self._long}"

    @property
    def display_name(self) -> str:
        return f"SMA Cross({self._short}/{self._long})"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return self._long + 1

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self.warmup_period:
            return SignalResult.neutral()

        fast = float(hub.sma(data, index, self._short))
        slow = float(hub.sma(data, index, self._long))
        prev_fast = float(hub.sma(data, index - 1, self._short))
        prev_slow = float(hub.sma(data, index - 1, self._long))

        atr = float(hub.atr(data, index, 14))
        diff = fast - slow
        score = max(-1.0, min(1.0, diff / atr)) if atr > 0 else 0.0

        golden = fast > slow and prev_fast <= prev_slow
        death = fast < slow and prev_fast >= prev_slow
        fired = golden or death

        direction = (
            SignalDirection.BULLISH if score > 0.05
            else SignalDirection.BEARISH if score < -0.05
            else SignalDirection.NEUTRAL
        )
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"sma_fast": fast, "sma_slow": slow})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"short_period": [5, 9, 12, 15, 20], "long_period": [21, 30, 50, 100, 200]}


class ADXTrendGenerator(AtomicSignalGenerator):
    """Score based on ADX strength and +DI/-DI directional bias.

    Strong trend (ADX > threshold) with +DI > -DI -> bullish.
    Weak trend (ADX < weak_threshold) -> neutral (range-bound filter).
    """

    def __init__(
        self, period: int = 14, strength: float = 25.0, weak: float = 20.0,
    ) -> None:
        self._period = period
        self._strength = strength
        self._weak = weak

    @property
    def name(self) -> str:
        return f"adx_{self._period}_{self._strength}_{self._weak}"

    @property
    def display_name(self) -> str:
        return f"ADX({self._period}) [{self._weak}/{self._strength}]"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return self._period * 2 + 1

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self.warmup_period:
            return SignalResult.neutral()

        adx_result = hub.adx(data, index, self._period)
        adx_val = float(adx_result.adx)
        plus_di = float(adx_result.plus_di)
        minus_di = float(adx_result.minus_di)

        prev_adx = hub.adx(data, index - 1, self._period)
        prev_val = float(prev_adx.adx)

        di_diff = plus_di - minus_di
        di_sum = plus_di + minus_di
        di_ratio = di_diff / di_sum if di_sum > 0 else 0.0

        if adx_val >= self._strength:
            strength_mult = min(1.0, adx_val / 50.0)
        elif adx_val <= self._weak:
            strength_mult = 0.0
        else:
            strength_mult = (adx_val - self._weak) / (self._strength - self._weak)

        score = max(-1.0, min(1.0, di_ratio * strength_mult))

        fired = adx_val >= self._strength and prev_val < self._strength

        direction = (
            SignalDirection.BULLISH if score > 0.05
            else SignalDirection.BEARISH if score < -0.05
            else SignalDirection.NEUTRAL
        )
        return SignalResult(
            score=score, direction=direction, fired=fired,
            metadata={"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di},
        )

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [14], "strength": [20.0, 25.0, 30.0], "weak": [15.0, 20.0]}


class EMATrendGenerator(AtomicSignalGenerator):
    """Score based on price distance from EMA.  State-based (no crossover fire)."""

    def __init__(self, period: int = 200) -> None:
        self._period = period

    @property
    def name(self) -> str:
        return f"ema_trend_{self._period}"

    @property
    def display_name(self) -> str:
        return f"EMA Trend({self._period})"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(
        self, data: Sequence[PriceData], index: int, hub: IndicatorHub,
    ) -> SignalResult:
        if index < self._period:
            return SignalResult.neutral()

        ema = float(hub.ema(data, index, self._period))
        close = float(data[index].close)
        atr = float(hub.atr(data, index, 14))

        if atr <= 0 or ema <= 0:
            return SignalResult.neutral()

        distance = (close - ema) / atr
        score = max(-1.0, min(1.0, distance / 3.0))

        prev_ema = float(hub.ema(data, index - 1, self._period))
        prev_close = float(data[index - 1].close)
        crossed_above = close > ema and prev_close <= prev_ema
        crossed_below = close < ema and prev_close >= prev_ema
        fired = crossed_above or crossed_below

        direction = (
            SignalDirection.BULLISH if score > 0.05
            else SignalDirection.BEARISH if score < -0.05
            else SignalDirection.NEUTRAL
        )
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"ema": ema, "close": close, "distance_atr": distance})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [50, 100, 150, 200]}


class IchimokuGenerator(AtomicSignalGenerator):
    """Score based on price vs Ichimoku cloud and TK cross."""

    @property
    def name(self) -> str:
        return "ichimoku"

    @property
    def display_name(self) -> str:
        return "Ichimoku Cloud"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return 52 + 26 + 1

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self.warmup_period:
            return SignalResult.neutral()

        ichi = hub.ichimoku(data, index)
        prev_ichi = hub.ichimoku(data, index - 1)

        above_cloud = ichi.price_above_cloud
        prev_above = prev_ichi.price_above_cloud

        tk_bullish = ichi.tenkan_sen > ichi.kijun_sen
        close = float(data[index].close)
        span_a = float(ichi.senkou_span_a)
        span_b = float(ichi.senkou_span_b)

        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        atr = float(hub.atr(data, index, 14))

        if atr <= 0:
            return SignalResult.neutral()

        if close > cloud_top:
            dist = (close - cloud_top) / atr
            score = min(1.0, 0.5 + dist / 6.0)
        elif close < cloud_bot:
            dist = (cloud_bot - close) / atr
            score = max(-1.0, -0.5 - dist / 6.0)
        else:
            score = 0.0

        if tk_bullish:
            score = min(1.0, score + 0.2)
        else:
            score = max(-1.0, score - 0.2)

        fired = above_cloud != prev_above

        direction = (SignalDirection.BULLISH if score > 0.1
                     else SignalDirection.BEARISH if score < -0.1
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"above_cloud": above_cloud, "tk_bullish": tk_bullish})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {}


class SARGenerator(AtomicSignalGenerator):
    """Score based on Parabolic SAR vs price.  Fires on SAR flip."""

    @property
    def name(self) -> str:
        return "sar"

    @property
    def display_name(self) -> str:
        return "Parabolic SAR"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return 3

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < 3:
            return SignalResult.neutral()

        bullish = hub.is_sar_bullish(data, index)
        prev_bullish = hub.is_sar_bullish(data, index - 1)

        sar = float(hub.parabolic_sar(data, index))
        close = float(data[index].close)
        atr = float(hub.atr(data, index, 14))

        if atr <= 0:
            score = 0.5 if bullish else -0.5
        else:
            distance = (close - sar) / atr
            score = max(-1.0, min(1.0, distance / 3.0))

        fired = bullish != prev_bullish

        direction = SignalDirection.BULLISH if bullish else SignalDirection.BEARISH
        return SignalResult(score=score, direction=direction, fired=fired,
                            metadata={"sar": sar, "bullish": bullish})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {}


class VWAPGenerator(AtomicSignalGenerator):
    """Score based on price distance from VWAP.  State-based."""

    def __init__(self, period: int = 20) -> None:
        self._period = period

    @property
    def name(self) -> str:
        return f"vwap_{self._period}"

    @property
    def display_name(self) -> str:
        return f"VWAP({self._period})"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self._period:
            return SignalResult.neutral()

        vwap = float(hub.vwap(data, index, self._period))
        close = float(data[index].close)
        atr = float(hub.atr(data, index, 14))

        if atr <= 0 or vwap <= 0:
            return SignalResult.neutral()

        distance = (close - vwap) / atr
        score = max(-1.0, min(1.0, distance / 2.0))

        prev_close = float(data[index - 1].close)
        prev_vwap = float(hub.vwap(data, index - 1, self._period))
        crossed = (close > vwap and prev_close <= prev_vwap) or \
                  (close < vwap and prev_close >= prev_vwap)

        direction = (SignalDirection.BULLISH if score > 0.05
                     else SignalDirection.BEARISH if score < -0.05
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=crossed,
                            metadata={"vwap": vwap, "close": close})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [10, 20, 30]}


class SMAPositionGenerator(AtomicSignalGenerator):
    """Score based on price position relative to SMA.  State filter."""

    def __init__(self, period: int = 200) -> None:
        self._period = period

    @property
    def name(self) -> str:
        return f"sma_pos_{self._period}"

    @property
    def display_name(self) -> str:
        return f"SMA Position({self._period})"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return self._period

    def evaluate(self, data: Sequence[PriceData], index: int, hub: IndicatorHub) -> SignalResult:
        if index < self._period:
            return SignalResult.neutral()

        sma = float(hub.sma(data, index, self._period))
        close = float(data[index].close)
        atr = float(hub.atr(data, index, 14))

        if atr <= 0 or sma <= 0:
            return SignalResult.neutral()

        distance = (close - sma) / atr
        score = max(-1.0, min(1.0, distance / 3.0))

        prev_close = float(data[index - 1].close)
        prev_sma = float(hub.sma(data, index - 1, self._period))
        crossed = (close > sma and prev_close <= prev_sma) or \
                  (close < sma and prev_close >= prev_sma)

        direction = (SignalDirection.BULLISH if score > 0.05
                     else SignalDirection.BEARISH if score < -0.05
                     else SignalDirection.NEUTRAL)
        return SignalResult(score=score, direction=direction, fired=crossed,
                            metadata={"sma": sma, "close": close, "distance_atr": distance})

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {"period": [50, 100, 200]}
