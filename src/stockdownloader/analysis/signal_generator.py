"""Generates trading alerts by analyzing all available technical indicators
and producing confluence-based buy/sell signals with options recommendations.

The signal generator scores indicators across four dimensions:
- Trend (EMA crossovers, SMA(200) position, Ichimoku cloud, Parabolic SAR)
- Momentum (RSI, MACD, Stochastic, Williams %R, CCI, ROC)
- Volume (OBV, MFI, volume vs average)
- Volatility (Bollinger Bands, ATR)

Options recommendations include:
- When to buy calls (bullish signals) or puts (bearish signals)
- Suggested strike prices based on Fibonacci levels and delta targets
- Suggested expiration (DTE) based on ATR and signal strength
- Estimated premium using Black-Scholes
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.model.alert_result import AlertResult, Direction, OptionsRecommendation, Action
from stockdownloader.model.indicator_values import IndicatorValues
from stockdownloader.model.option_type import OptionType
from stockdownloader.util import black_scholes_calculator as bsc
from stockdownloader.util import technical_indicators as ti

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData

_ZERO = Decimal("0")
_RISK_FREE_RATE = Decimal("0.05")
_VOLATILITY_LOOKBACK = 20


def generate_alert(
    symbol: str,
    data: list[PriceData],
    index: int | None = None,
) -> AlertResult:
    """Generate a full trading alert for the given bar index in the data.

    Args:
        symbol: Ticker symbol.
        data: List of ``PriceData`` bars.
        index: Bar index to evaluate. Defaults to the last bar.

    Returns:
        An ``AlertResult`` with direction, confluence score, indicators, and
        options recommendations.
    """
    if index is None:
        index = len(data) - 1

    if index < 200:
        # Not enough data for full analysis
        return _create_neutral_alert(symbol, data, index)

    current = IndicatorValues.compute(data, index)
    previous = IndicatorValues.compute(data, index - 1)

    bullish: list[str] = []
    bearish: list[str] = []

    # === TREND INDICATORS ===
    _score_trend_indicators(current, previous, bullish, bearish)

    # === MOMENTUM INDICATORS ===
    _score_momentum_indicators(current, previous, bullish, bearish)

    # === VOLUME INDICATORS ===
    _score_volume_indicators(current, data, index, bullish, bearish)

    # === VOLATILITY INDICATORS ===
    _score_volatility_indicators(current, previous, bullish, bearish)

    # Calculate confluence
    total_indicators = len(bullish) + len(bearish)
    if total_indicators == 0:
        total_indicators = 1

    bullish_pct = len(bullish) / total_indicators
    bearish_pct = len(bearish) / total_indicators

    if bullish_pct >= 0.75:
        direction = Direction.STRONG_BUY
        confluence_score = bullish_pct
    elif bullish_pct >= 0.55:
        direction = Direction.BUY
        confluence_score = bullish_pct
    elif bearish_pct >= 0.75:
        direction = Direction.STRONG_SELL
        confluence_score = bearish_pct
    elif bearish_pct >= 0.55:
        direction = Direction.SELL
        confluence_score = bearish_pct
    else:
        direction = Direction.NEUTRAL
        confluence_score = max(bullish_pct, bearish_pct)

    # Generate options recommendations
    close_prices = [bar.close for bar in data]
    vol = bsc.estimate_volatility(
        close_prices, min(index + 1, _VOLATILITY_LOOKBACK)
    )

    call_rec = _generate_call_recommendation(current, direction, vol, data, index)
    put_rec = _generate_put_recommendation(current, direction, vol, data, index)

    # Support & Resistance
    sr = ti.support_resistance(data, index, 100, 5)

    return AlertResult(
        symbol=symbol,
        date=current.date,
        current_price=current.close,
        direction=direction,
        confluence_score=confluence_score,
        total_indicators=total_indicators,
        bullish_indicators=bullish,
        bearish_indicators=bearish,
        call_recommendation=call_rec,
        put_recommendation=put_rec,
        support_levels=sr.support_levels,
        resistance_levels=sr.resistance_levels,
        indicators=current,
    )


# =========================================================================
# TREND SCORING
# =========================================================================

def _score_trend_indicators(
    current: IndicatorValues,
    previous: IndicatorValues,
    bullish: list[str],
    bearish: list[str],
) -> None:
    # EMA(12) vs EMA(26) crossover
    if current.ema12 > current.ema26:
        bullish.append("EMA(12) > EMA(26) -- short-term uptrend")
    elif current.ema12 < current.ema26:
        bearish.append("EMA(12) < EMA(26) -- short-term downtrend")

    # Price vs SMA(200) -- long-term trend
    if current.sma200 > _ZERO:
        if current.close > current.sma200:
            bullish.append("Price above SMA(200) -- long-term uptrend")
        else:
            bearish.append("Price below SMA(200) -- long-term downtrend")

    # SMA(50) vs SMA(200) -- golden/death cross
    if current.sma50 > _ZERO and current.sma200 > _ZERO:
        if current.sma50 > current.sma200 and previous.sma50 <= previous.sma200:
            bullish.append("Golden Cross -- SMA(50) crossed above SMA(200)")
        elif current.sma50 < current.sma200 and previous.sma50 >= previous.sma200:
            bearish.append("Death Cross -- SMA(50) crossed below SMA(200)")

    # Ichimoku Cloud
    if current.ichimoku_span_a > _ZERO:
        if current.price_above_cloud:
            bullish.append("Ichimoku -- price above cloud (bullish)")
        else:
            bearish.append("Ichimoku -- price below cloud (bearish)")

        # Tenkan/Kijun cross
        if (
            current.ichimoku_tenkan > current.ichimoku_kijun
            and previous.ichimoku_tenkan <= previous.ichimoku_kijun
        ):
            bullish.append("Ichimoku TK Cross -- Tenkan above Kijun (bullish)")
        elif (
            current.ichimoku_tenkan < current.ichimoku_kijun
            and previous.ichimoku_tenkan >= previous.ichimoku_kijun
        ):
            bearish.append("Ichimoku TK Cross -- Tenkan below Kijun (bearish)")

    # Parabolic SAR
    if current.sar_bullish and not previous.sar_bullish:
        bullish.append("Parabolic SAR flipped bullish (SAR below price)")
    elif not current.sar_bullish and previous.sar_bullish:
        bearish.append("Parabolic SAR flipped bearish (SAR above price)")

    # ADX trend strength
    if float(current.adx14) > 25:
        if current.plus_di > current.minus_di:
            bullish.append("ADX > 25 with +DI > -DI -- strong uptrend")
        else:
            bearish.append("ADX > 25 with -DI > +DI -- strong downtrend")


# =========================================================================
# MOMENTUM SCORING
# =========================================================================

def _score_momentum_indicators(
    current: IndicatorValues,
    previous: IndicatorValues,
    bullish: list[str],
    bearish: list[str],
) -> None:
    # RSI
    rsi = float(current.rsi14)
    prev_rsi = float(previous.rsi14)
    if rsi > 30 and prev_rsi <= 30:
        bullish.append(f"RSI(14) = {rsi:.1f} -- recovering from oversold")
    elif rsi < 70 and prev_rsi >= 70:
        bearish.append(f"RSI(14) = {rsi:.1f} -- falling from overbought")
    elif rsi < 30:
        bullish.append(f"RSI(14) = {rsi:.1f} -- oversold (potential reversal up)")
    elif rsi > 70:
        bearish.append(f"RSI(14) = {rsi:.1f} -- overbought (potential reversal down)")

    # MACD crossover
    if current.macd_line > current.macd_signal and previous.macd_line <= previous.macd_signal:
        bullish.append("MACD bullish crossover (MACD > Signal)")
    elif current.macd_line < current.macd_signal and previous.macd_line >= previous.macd_signal:
        bearish.append("MACD bearish crossover (MACD < Signal)")

    # MACD histogram direction
    if current.macd_histogram > _ZERO and current.macd_histogram > previous.macd_histogram:
        bullish.append("MACD histogram expanding positive")
    elif current.macd_histogram < _ZERO and current.macd_histogram < previous.macd_histogram:
        bearish.append("MACD histogram expanding negative")

    # Stochastic
    stoch_k = float(current.stoch_k)
    if stoch_k < 20:
        bullish.append(f"Stochastic %K = {stoch_k:.1f} -- oversold")
    elif stoch_k > 80:
        bearish.append(f"Stochastic %K = {stoch_k:.1f} -- overbought")

    # Williams %R
    will_r = float(current.williams_r)
    if will_r < -80:
        bullish.append(f"Williams %R = {will_r:.1f} -- oversold")
    elif will_r > -20:
        bearish.append(f"Williams %R = {will_r:.1f} -- overbought")

    # CCI
    cci = float(current.cci20)
    if cci < -100:
        bullish.append(f"CCI(20) = {cci:.1f} -- oversold")
    elif cci > 100:
        bearish.append(f"CCI(20) = {cci:.1f} -- overbought")

    # ROC
    roc = float(current.roc12)
    if roc > 0 and float(previous.roc12) <= 0:
        bullish.append("ROC(12) crossed positive -- momentum turning up")
    elif roc < 0 and float(previous.roc12) >= 0:
        bearish.append("ROC(12) crossed negative -- momentum turning down")


# =========================================================================
# VOLUME SCORING
# =========================================================================

def _score_volume_indicators(
    current: IndicatorValues,
    data: list[PriceData],
    index: int,
    bullish: list[str],
    bearish: list[str],
) -> None:
    # OBV
    if current.obv_rising:
        bullish.append("OBV rising -- accumulation (buying pressure)")
    else:
        bearish.append("OBV falling -- distribution (selling pressure)")

    # MFI
    mfi = float(current.mfi14)
    if mfi < 20:
        bullish.append(f"MFI(14) = {mfi:.1f} -- oversold (volume-weighted)")
    elif mfi > 80:
        bearish.append(f"MFI(14) = {mfi:.1f} -- overbought (volume-weighted)")

    # Volume vs average
    avg_vol = current.avg_volume20
    if avg_vol > _ZERO:
        vol_ratio = (
            Decimal(str(current.volume))
            / avg_vol
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if float(vol_ratio) > 1.5:
            # High volume confirms current trend
            if current.close > data[index - 1].close:
                bullish.append(f"Volume {float(vol_ratio):.1f}x avg -- confirming upward move")
            else:
                bearish.append(f"Volume {float(vol_ratio):.1f}x avg -- confirming downward move")


# =========================================================================
# VOLATILITY SCORING
# =========================================================================

def _score_volatility_indicators(
    current: IndicatorValues,
    previous: IndicatorValues,
    bullish: list[str],
    bearish: list[str],
) -> None:
    # Bollinger Bands
    if float(current.bb_percent_b) <= 0:
        bullish.append("Price at/below lower Bollinger Band -- oversold")
    elif float(current.bb_percent_b) >= 1:
        bearish.append("Price at/above upper Bollinger Band -- overbought")

    # Price vs VWAP
    if current.vwap > _ZERO:
        if current.close > current.vwap:
            bullish.append("Price above VWAP -- bullish intraday bias")
        else:
            bearish.append("Price below VWAP -- bearish intraday bias")


# =========================================================================
# OPTIONS RECOMMENDATIONS
# =========================================================================

def _generate_call_recommendation(
    current: IndicatorValues,
    direction: Direction,
    vol: Decimal,
    data: list[PriceData],
    index: int,
) -> OptionsRecommendation:
    price = current.close

    if direction in (Direction.STRONG_BUY, Direction.BUY):
        # Recommend buying calls
        target_delta = (
            Decimal("0.50") if direction == Direction.STRONG_BUY else Decimal("0.35")
        )

        strike = _select_call_strike(current, price)
        dte = _select_dte(current.atr14, price, direction)

        time_to_expiry = (
            Decimal(str(dte)) / Decimal("365")
        ).quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
        premium = bsc.price(
            OptionType.CALL, price, strike, time_to_expiry, _RISK_FREE_RATE, vol
        )

        rationale = _build_call_rationale(current, direction)

        return OptionsRecommendation(
            type=OptionType.CALL,
            action=Action.BUY,
            suggested_strike=strike,
            suggested_dte=dte,
            estimated_premium=premium,
            target_delta=target_delta,
            rationale=rationale,
        )

    elif direction in (Direction.STRONG_SELL, Direction.SELL):
        # Bearish scenario: sell calls (covered call / income)
        strike = (price * Decimal("1.05")).to_integral_value(rounding="ROUND_CEILING")
        dte = 30

        time_to_expiry = (
            Decimal(str(dte)) / Decimal("365")
        ).quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
        premium = bsc.price(
            OptionType.CALL, price, strike, time_to_expiry, _RISK_FREE_RATE, vol
        )

        return OptionsRecommendation(
            type=OptionType.CALL,
            action=Action.SELL,
            suggested_strike=strike,
            suggested_dte=dte,
            estimated_premium=premium,
            target_delta=Decimal("0.30"),
            rationale="Bearish outlook -- sell OTM covered call for income",
        )

    else:
        return OptionsRecommendation(
            type=OptionType.CALL,
            action=Action.HOLD,
            suggested_strike=price,
            suggested_dte=0,
            estimated_premium=_ZERO,
            target_delta=_ZERO,
            rationale="Neutral -- wait for clearer signal",
        )


def _generate_put_recommendation(
    current: IndicatorValues,
    direction: Direction,
    vol: Decimal,
    data: list[PriceData],
    index: int,
) -> OptionsRecommendation:
    price = current.close

    if direction in (Direction.STRONG_SELL, Direction.SELL):
        # Recommend buying puts
        target_delta = (
            Decimal("0.50") if direction == Direction.STRONG_SELL else Decimal("0.35")
        )

        strike = _select_put_strike(current, price)
        dte = _select_dte(current.atr14, price, direction)

        time_to_expiry = (
            Decimal(str(dte)) / Decimal("365")
        ).quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
        premium = bsc.price(
            OptionType.PUT, price, strike, time_to_expiry, _RISK_FREE_RATE, vol
        )

        rationale = _build_put_rationale(current, direction)

        return OptionsRecommendation(
            type=OptionType.PUT,
            action=Action.BUY,
            suggested_strike=strike,
            suggested_dte=dte,
            estimated_premium=premium,
            target_delta=target_delta,
            rationale=rationale,
        )

    elif direction in (Direction.STRONG_BUY, Direction.BUY):
        # Bullish scenario: sell puts (cash-secured put for income)
        strike = (price * Decimal("0.95")).to_integral_value(rounding="ROUND_FLOOR")
        dte = 30

        time_to_expiry = (
            Decimal(str(dte)) / Decimal("365")
        ).quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
        premium = bsc.price(
            OptionType.PUT, price, strike, time_to_expiry, _RISK_FREE_RATE, vol
        )

        return OptionsRecommendation(
            type=OptionType.PUT,
            action=Action.SELL,
            suggested_strike=strike,
            suggested_dte=dte,
            estimated_premium=premium,
            target_delta=Decimal("-0.30"),
            rationale="Bullish outlook -- sell OTM cash-secured put for income",
        )

    else:
        return OptionsRecommendation(
            type=OptionType.PUT,
            action=Action.HOLD,
            suggested_strike=price,
            suggested_dte=0,
            estimated_premium=_ZERO,
            target_delta=_ZERO,
            rationale="Neutral -- wait for clearer signal",
        )


def _select_call_strike(current: IndicatorValues, price: Decimal) -> Decimal:
    """For strong signals, use ATM. Otherwise use nearest Fibonacci resistance or 2-3% OTM."""
    if (
        current.fib_382 > _ZERO
        and current.fib_382 > price
        and float(
            abs(current.fib_382 - price)
            / price
        ) < 0.05
    ):
        return current.fib_382.to_integral_value(rounding="ROUND_CEILING")
    # Default: ATM or slightly OTM
    return price.to_integral_value(rounding="ROUND_CEILING")


def _select_put_strike(current: IndicatorValues, price: Decimal) -> Decimal:
    """Use nearest Fibonacci support or ATM."""
    if (
        current.fib_618 > _ZERO
        and current.fib_618 < price
        and float(
            (price - current.fib_618)
            / price
        ) < 0.05
    ):
        return current.fib_618.to_integral_value(rounding="ROUND_FLOOR")
    return price.to_integral_value(rounding="ROUND_FLOOR")


def _select_dte(atr: Decimal, price: Decimal, direction: Direction) -> int:
    """Strong signals get shorter DTE (more gamma), weaker signals get longer DTE."""
    if direction in (Direction.STRONG_BUY, Direction.STRONG_SELL):
        return 30  # Optimal theta/gamma balance

    if price > _ZERO:
        atr_pct = float(
            atr / price
        )
        if atr_pct > 0.03:
            return 45  # High volatility: longer DTE to avoid theta decay

    return 30


def _build_call_rationale(current: IndicatorValues, direction: Direction) -> str:
    parts: list[str] = []
    if float(current.rsi14) < 40:
        parts.append("RSI oversold")
    if current.macd_histogram > _ZERO:
        parts.append("MACD bullish")
    if current.price_above_cloud:
        parts.append("above Ichimoku cloud")
    if current.obv_rising:
        parts.append("OBV rising")
    if current.sar_bullish:
        parts.append("SAR bullish")

    strength = "Strong" if direction == Direction.STRONG_BUY else "Moderate"
    return f"{strength} bullish confluence: {', '.join(parts)}"


def _build_put_rationale(current: IndicatorValues, direction: Direction) -> str:
    parts: list[str] = []
    if float(current.rsi14) > 60:
        parts.append("RSI overbought")
    if current.macd_histogram < _ZERO:
        parts.append("MACD bearish")
    if not current.price_above_cloud:
        parts.append("below Ichimoku cloud")
    if not current.obv_rising:
        parts.append("OBV falling")
    if not current.sar_bullish:
        parts.append("SAR bearish")

    strength = "Strong" if direction == Direction.STRONG_SELL else "Moderate"
    return f"{strength} bearish confluence: {', '.join(parts)}"


def _create_neutral_alert(
    symbol: str, data: list[PriceData], index: int
) -> AlertResult:
    bar = data[index]
    noop_call = OptionsRecommendation(
        type=OptionType.CALL,
        action=Action.HOLD,
        suggested_strike=bar.close,
        suggested_dte=0,
        estimated_premium=_ZERO,
        target_delta=_ZERO,
        rationale="Insufficient data for analysis (need 200+ bars)",
    )
    noop_put = OptionsRecommendation(
        type=OptionType.PUT,
        action=Action.HOLD,
        suggested_strike=bar.close,
        suggested_dte=0,
        estimated_premium=_ZERO,
        target_delta=_ZERO,
        rationale="Insufficient data for analysis (need 200+ bars)",
    )

    return AlertResult(
        symbol=symbol,
        date=bar.date,
        current_price=bar.close,
        direction=Direction.NEUTRAL,
        confluence_score=0.0,
        total_indicators=0,
        bullish_indicators=[],
        bearish_indicators=[],
        call_recommendation=noop_call,
        put_recommendation=noop_put,
        support_levels=[],
        resistance_levels=[],
        indicators=IndicatorValues.compute(data, index) if index >= 52 else None,
    )
