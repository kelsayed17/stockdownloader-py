"""Snapshot of all computed technical indicator values at a given point in time."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


@dataclass(frozen=True)
class IndicatorValues:
    """Snapshot of all computed technical indicator values at a given point in time.

    Used by multi-indicator strategies and the signal generator to make decisions
    based on the full suite of available indicators rather than just price alone.
    """

    # Price data
    date: str
    close: Decimal
    high: Decimal
    low: Decimal
    open: Decimal
    volume: int

    # Moving Averages
    sma20: Decimal
    sma50: Decimal
    sma200: Decimal
    ema12: Decimal
    ema26: Decimal

    # Momentum
    rsi14: Decimal
    macd_line: Decimal
    macd_signal: Decimal
    macd_histogram: Decimal
    roc12: Decimal
    williams_r: Decimal

    # Bollinger Bands
    bb_upper: Decimal
    bb_middle: Decimal
    bb_lower: Decimal
    bb_width: Decimal
    bb_percent_b: Decimal

    # Stochastic
    stoch_k: Decimal
    stoch_d: Decimal

    # Volatility
    atr14: Decimal

    # Volume
    obv: Decimal
    obv_rising: bool
    mfi14: Decimal
    avg_volume20: Decimal

    # Trend
    adx14: Decimal
    plus_di: Decimal
    minus_di: Decimal
    parabolic_sar: Decimal
    sar_bullish: bool

    # CCI
    cci20: Decimal

    # VWAP
    vwap: Decimal

    # Ichimoku
    ichimoku_tenkan: Decimal
    ichimoku_kijun: Decimal
    ichimoku_span_a: Decimal
    ichimoku_span_b: Decimal
    price_above_cloud: bool

    # Fibonacci
    fib_high: Decimal
    fib_low: Decimal
    fib_236: Decimal
    fib_382: Decimal
    fib_500: Decimal
    fib_618: Decimal
    fib_786: Decimal

    @classmethod
    def compute(cls, data: list[PriceData], index: int) -> IndicatorValues:
        """Compute all indicator values at the given index in the price data.

        Requires at least 200 bars for SMA(200); partial values are returned
        for indicators that have enough data.
        """
        from stockdownloader.util.moving_average_calculator import sma as _sma, ema as _ema
        from stockdownloader.util import technical_indicators as ti

        bar = data[index]

        # Moving averages
        sma20 = _sma(data, index, 20) if index >= 19 else Decimal("0")
        sma50 = _sma(data, index, 50) if index >= 49 else Decimal("0")
        sma200 = _sma(data, index, 200) if index >= 199 else Decimal("0")
        ema12 = _ema(data, index, 12) if index >= 12 else Decimal("0")
        ema26 = _ema(data, index, 26) if index >= 26 else Decimal("0")

        # RSI
        rsi14 = ti.rsi(data, index, 14)

        # MACD
        macd_l = ti.macd_line(data, index, 12, 26)
        macd_s = ti.macd_signal(data, index, 12, 26, 9)
        macd_h = ti.macd_histogram(data, index, 12, 26, 9)

        # ROC
        roc = ti.roc(data, index, 12)

        # Williams %R
        will_r = ti.williams_r(data, index, 14)

        # Bollinger Bands
        bb = ti.bollinger_bands(data, index)
        bb_pct_b = ti.bollinger_percent_b(data, index, 20) if index >= 19 else Decimal("0")

        # Stochastic
        stoch = ti.stochastic(data, index)

        # ATR
        atr = ti.atr(data, index, 14)

        # OBV
        obv_val = ti.obv(data, index)
        obv_ris = ti.is_obv_rising(data, index, 5)

        # MFI
        mfi_val = ti.mfi(data, index, 14)

        # Average Volume
        avg_vol = ti.average_volume(data, index, 20)

        # ADX
        adx_result = ti.adx(data, index, 14)

        # Parabolic SAR
        psar = ti.parabolic_sar(data, index)
        sar_bull = ti.is_sar_bullish(data, index)

        # CCI
        cci_val = ti.cci(data, index, 20)

        # VWAP
        vwap_val = ti.vwap(data, index, 20)

        # Ichimoku
        ichimoku = ti.ichimoku(data, index)

        # Fibonacci
        fib = ti.fibonacci_retracement(data, index, 50)

        return cls(
            date=bar.date,
            close=bar.close,
            high=bar.high,
            low=bar.low,
            open=bar.open,
            volume=bar.volume,
            sma20=sma20,
            sma50=sma50,
            sma200=sma200,
            ema12=ema12,
            ema26=ema26,
            rsi14=rsi14,
            macd_line=macd_l,
            macd_signal=macd_s,
            macd_histogram=macd_h,
            roc12=roc,
            williams_r=will_r,
            bb_upper=bb.upper,
            bb_middle=bb.middle,
            bb_lower=bb.lower,
            bb_width=bb.width,
            bb_percent_b=bb_pct_b,
            stoch_k=stoch.percent_k,
            stoch_d=stoch.percent_d,
            atr14=atr,
            obv=obv_val,
            obv_rising=obv_ris,
            mfi14=mfi_val,
            avg_volume20=avg_vol,
            adx14=adx_result.adx,
            plus_di=adx_result.plus_di,
            minus_di=adx_result.minus_di,
            parabolic_sar=psar,
            sar_bullish=sar_bull,
            cci20=cci_val,
            vwap=vwap_val,
            ichimoku_tenkan=ichimoku.tenkan_sen,
            ichimoku_kijun=ichimoku.kijun_sen,
            ichimoku_span_a=ichimoku.senkou_span_a,
            ichimoku_span_b=ichimoku.senkou_span_b,
            price_above_cloud=ichimoku.price_above_cloud,
            fib_high=fib.high,
            fib_low=fib.low,
            fib_236=fib.level_236,
            fib_382=fib.level_382,
            fib_500=fib.level_500,
            fib_618=fib.level_618,
            fib_786=fib.level_786,
        )

    def summary(self) -> str:
        """Summary string for display in reports."""
        return (
            f"{self.date}  Close: ${self.close}  Vol: {self.volume:,}\n"
            f"  SMA(20/50/200): {_fmt(self.sma20)} / {_fmt(self.sma50)} / {_fmt(self.sma200)}\n"
            f"  RSI(14): {_fmt(self.rsi14)}  MACD: {_fmt(self.macd_line)}  ADX: {_fmt(self.adx14)}\n"
            f"  BB: [{_fmt(self.bb_lower)} - {_fmt(self.bb_middle)} - {_fmt(self.bb_upper)}]  %B: {_fmt(self.bb_percent_b)}\n"
            f"  Stoch: %K={_fmt(self.stoch_k)} %D={_fmt(self.stoch_d)}  ATR: {_fmt(self.atr14)}\n"
            f"  OBV Rising: {'Yes' if self.obv_rising else 'No'}  MFI: {_fmt(self.mfi14)}  CCI: {_fmt(self.cci20)}\n"
            f"  VWAP: {_fmt(self.vwap)}  SAR: {_fmt(self.parabolic_sar)} ({'Bullish' if self.sar_bullish else 'Bearish'})\n"
            f"  Ichimoku: {'Above' if self.price_above_cloud else 'Below'} cloud  Williams %R: {_fmt(self.williams_r)}\n"
            f"  Fib Levels: 23.6%={_fmt(self.fib_236)}  38.2%={_fmt(self.fib_382)}  50%={_fmt(self.fib_500)}  61.8%={_fmt(self.fib_618)}"
        )


def _fmt(v: Decimal | None) -> str:
    """Format a Decimal value to 2 decimal places, or 'N/A' if None."""
    if v is None:
        return "N/A"
    return str(v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
