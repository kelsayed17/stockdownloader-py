"""Cached indicator calculator for strategies.

Wraps all functions from the unified :mod:`indicators` package with
per-``(indicator, index, params)`` caching.

**Performance**: For indicators that dominate backtest runtime (OBV, EMA,
ATR, ADX, RSI, MACD, Parabolic SAR), the hub uses *streaming accumulators*
that maintain running state and update in O(1) per bar instead of
recomputing from scratch (O(n)).  This drops Multi-Indicator backtests
from ~1400s to ~10s on 40K bars.

Each strategy holds its own hub instance.  The cache auto-clears when
the underlying data reference changes (new backtest run).

Usage::

    from stockdownloader.indicators.hub import IndicatorHub

    class MyStrategy(TradingStrategy):
        def __init__(self):
            self._hub = IndicatorHub()

        def evaluate(self, data, current_index):
            rsi = self._hub.rsi(data, current_index, period=14)
            bb = self._hub.bollinger_bands(data, current_index, period=20)
            # Second call at same index+params is a cache hit:
            rsi_prev = self._hub.rsi(data, current_index - 1, period=14)
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

# Batch indicator functions — imported from specific modules
from stockdownloader.indicators.core import sma as _sma
from stockdownloader.indicators import momentum, volatility, trend
from stockdownloader.indicators.hub_intraday import IntraDayHubMixin

# Streaming accumulators — imported from specific modules
from stockdownloader.indicators.momentum import (
    StreamingMACD,
    StreamingOBV,
    StreamingRSI,
)
from stockdownloader.indicators.volatility import (
    StreamingATR,
    StreamingEMA,
)
from stockdownloader.indicators.trend import (
    StreamingADX,
    StreamingSAR,
)
from stockdownloader.indicators.volume import (
    StreamingAnchoredVWAP,
    StreamingCVD,
    StreamingSessionVWAP,
)
from stockdownloader.indicators.htf import StreamingHTFResample
from stockdownloader.indicators.smc import StreamingStructureTracker

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.core.models.price import PriceData


class IndicatorHub(IntraDayHubMixin):
    """Per-strategy indicator cache with streaming acceleration.

    Delegates to the raw indicator functions on first call, then serves
    subsequent identical requests from an in-memory dict.

    For high-cost indicators (OBV, EMA, ATR, ADX, RSI, MACD, SAR), the
    hub maintains streaming accumulators that update incrementally in O(1)
    per bar.  These are keyed by parameter tuple (e.g., EMA period) and
    lazily created on first use.

    Cache is keyed by ``("indicator_name", index, param1, param2, ...)``.
    Cache auto-clears when the ``id(data)`` changes (different list object).
    """

    __slots__ = (
        "_cache", "_data_id",
        # Streaming accumulators for O(n)-per-call indicators
        "_s_obv",      # StreamingOBV (OBV scans from bar 0)
        "_s_sar",      # dict[(af_start, af_step, af_max), StreamingSAR]
        "_s_rsi",      # dict[period, StreamingRSI]
        "_s_ema",      # dict[period, StreamingEMA]
        "_s_atr",      # dict[period, StreamingATR]
        "_s_adx",      # dict[period, StreamingADX]
        "_s_macd",     # dict[(fast, slow, signal), StreamingMACD]
        "_s_vwap",     # StreamingSessionVWAP (session VWAP resets per day)
        "_s_avwap",    # dict[anchor_type, StreamingAnchoredVWAP]
        "_s_cvd",      # StreamingCVD (session CVD resets per day)
        "_s_htf",      # dict[factor, StreamingHTFResample]
        "_s_structure", # dict[params_tuple, StreamingStructureTracker]
    )

    def __init__(self) -> None:
        self._cache: dict[tuple, Any] = {}
        self._data_id: int | None = None
        # Streaming accumulators for indicators that scan from bar 0
        self._s_obv: StreamingOBV | None = None
        self._s_sar: dict[tuple, StreamingSAR] = {}
        self._s_rsi: dict[int, StreamingRSI] = {}
        self._s_ema: dict[int, StreamingEMA] = {}
        self._s_atr: dict[int, StreamingATR] = {}
        self._s_adx: dict[int, StreamingADX] = {}
        self._s_macd: dict[tuple, StreamingMACD] = {}
        self._s_vwap: StreamingSessionVWAP | None = None
        self._s_avwap: dict[str, StreamingAnchoredVWAP] = {}
        self._s_cvd: StreamingCVD | None = None
        self._s_htf: dict[int, StreamingHTFResample] = {}
        self._s_structure: dict[tuple, StreamingStructureTracker] = {}

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------

    def _ensure_bound(self, data: Sequence[PriceData]) -> None:
        """Clear cache and streaming state if data reference changed."""
        did = id(data)
        if self._data_id != did:
            self._cache.clear()
            self._reset_streaming()
            self._data_id = did

    def _reset_streaming(self) -> None:
        """Reset all streaming accumulators."""
        self._s_obv = None
        self._s_sar.clear()
        self._s_rsi.clear()
        self._s_ema.clear()
        self._s_atr.clear()
        self._s_adx.clear()
        self._s_macd.clear()
        self._s_vwap = None
        self._s_avwap.clear()
        self._s_cvd = None
        self._s_htf.clear()
        self._s_structure.clear()

    def _get(self, key: tuple, data: Sequence[PriceData], fn, *args, **kwargs) -> Any:
        """Cache-through: return cached value or compute and store."""
        self._ensure_bound(data)
        if key not in self._cache:
            self._cache[key] = fn(*args, **kwargs)
        return self._cache[key]

    def clear(self) -> None:
        """Manually clear all cached values and streaming state."""
        self._cache.clear()
        self._reset_streaming()

    @property
    def cache_size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    # ==================================================================
    # Moving Averages (streaming EMA, regular-cache SMA)
    # ==================================================================

    def sma(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int,
    ) -> Decimal:
        """Simple Moving Average."""
        return self._get(
            ("sma", index, period),
            data, _sma, data, index, period,
        )

    def ema(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int,
    ) -> Decimal:
        """Exponential Moving Average (streaming)."""
        self._ensure_bound(data)
        if period not in self._s_ema:
            self._s_ema[period] = StreamingEMA(period)
        return self._s_ema[period].update(data, index)

    # ==================================================================
    # Momentum (streaming RSI, streaming MACD)
    # ==================================================================

    def rsi(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 14,
    ) -> Decimal:
        """Relative Strength Index (streaming)."""
        self._ensure_bound(data)
        if period not in self._s_rsi:
            self._s_rsi[period] = StreamingRSI(period)
        return self._s_rsi[period].update(data, index)

    def _get_streaming_macd(
        self,
        data: Sequence[PriceData],
        index: int,
        fast: int,
        slow: int,
        signal: int,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Get or create StreamingMACD and update to index.

        Returns ``(line, signal_line, histogram)``.
        """
        self._ensure_bound(data)
        key = (fast, slow, signal)
        if key not in self._s_macd:
            self._s_macd[key] = StreamingMACD(fast, slow, signal)
        return self._s_macd[key].update(data, index)

    def macd_line(
        self,
        data: Sequence[PriceData],
        index: int,
        fast: int = 12,
        slow: int = 26,
    ) -> Decimal:
        """MACD line (fast EMA - slow EMA) (streaming)."""
        line, _, _ = self._get_streaming_macd(data, index, fast, slow, 9)
        return line

    def macd_signal(
        self,
        data: Sequence[PriceData],
        index: int,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Decimal:
        """MACD signal line (streaming)."""
        _, sig, _ = self._get_streaming_macd(data, index, fast, slow, signal)
        return sig

    def macd_histogram(
        self,
        data: Sequence[PriceData],
        index: int,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Decimal:
        """MACD histogram (streaming)."""
        _, _, hist = self._get_streaming_macd(data, index, fast, slow, signal)
        return hist

    def roc(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 12,
    ) -> Decimal:
        """Rate of Change."""
        return self._get(
            ("roc", index, period),
            data, momentum.roc, data, index, period,
        )

    def williams_r(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 14,
    ) -> Decimal:
        """Williams %R."""
        return self._get(
            ("williams_r", index, period),
            data, momentum.williams_r, data, index, period,
        )

    # ==================================================================
    # Bollinger Bands
    # ==================================================================

    def bollinger_bands(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 20,
        num_std_dev: float = 2.0,
    ) -> volatility.BollingerBands:
        """Bollinger Bands."""
        return self._get(
            ("bollinger_bands", index, period, num_std_dev),
            data, volatility.bollinger_bands, data, index, period, num_std_dev,
        )

    def bollinger_percent_b(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 20,
    ) -> Decimal:
        """Bollinger %B (composed from cached bollinger_bands)."""
        key = ("bollinger_percent_b", index, period)
        self._ensure_bound(data)
        if key not in self._cache:
            bb = self.bollinger_bands(data, index, period, 2.0)
            self._cache[key] = volatility._bollinger_percent_b_from_bands(
                data[index].close, bb,
            )
        return self._cache[key]

    # ==================================================================
    # Stochastic
    # ==================================================================

    def stochastic(
        self,
        data: Sequence[PriceData],
        index: int,
        k_period: int = 14,
        d_period: int = 3,
    ) -> momentum.Stochastic:
        """Stochastic Oscillator (%K, %D)."""
        return self._get(
            ("stochastic", index, k_period, d_period),
            data, momentum.stochastic, data, index, k_period, d_period,
        )

    # ==================================================================
    # Volatility (streaming ATR)
    # ==================================================================

    def atr(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 14,
    ) -> Decimal:
        """Average True Range (streaming)."""
        self._ensure_bound(data)
        if period not in self._s_atr:
            self._s_atr[period] = StreamingATR(period)
        return self._s_atr[period].update(data, index)

    def true_range(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> Decimal:
        """True Range for a single bar."""
        from stockdownloader.indicators.core import true_range as _true_range
        return self._get(
            ("true_range", index),
            data, _true_range, data, index,
        )

    def standard_deviation(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int,
    ) -> Decimal:
        """Standard deviation of close prices."""
        from stockdownloader.indicators.core import standard_deviation as _std
        return self._get(
            ("standard_deviation", index, period),
            data, _std, data, index, period,
        )

    # ==================================================================
    # Volume (streaming OBV)
    # ==================================================================

    def obv(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> Decimal:
        """On-Balance Volume (streaming)."""
        self._ensure_bound(data)
        if self._s_obv is None:
            self._s_obv = StreamingOBV()
        return self._s_obv.update(data, index)

    def is_obv_rising(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int = 5,
    ) -> bool:
        """Whether OBV is rising over *lookback* bars (streaming)."""
        self._ensure_bound(data)
        if self._s_obv is None:
            self._s_obv = StreamingOBV()
        # Ensure OBV computed up to index
        self._s_obv.update(data, index)
        return self._s_obv.is_rising(index, lookback)

    def average_volume(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 20,
    ) -> Decimal:
        """Average volume over *period* bars."""
        return self._get(
            ("average_volume", index, period),
            data, momentum.average_volume, data, index, period,
        )

    def mfi(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 14,
    ) -> Decimal:
        """Money Flow Index."""
        return self._get(
            ("mfi", index, period),
            data, momentum.mfi, data, index, period,
        )

    # ==================================================================
    # Trend (streaming ADX, streaming SAR)
    # ==================================================================

    def adx(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 14,
    ) -> trend.ADXResult:
        """Average Directional Index (ADX, +DI, -DI) (streaming)."""
        self._ensure_bound(data)
        if period not in self._s_adx:
            self._s_adx[period] = StreamingADX(period)
        vals = self._s_adx[period].update(data, index)
        return trend.ADXResult(adx=vals[0], plus_di=vals[1], minus_di=vals[2])

    def parabolic_sar(
        self,
        data: Sequence[PriceData],
        index: int,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> Decimal:
        """Parabolic SAR with configurable acceleration factors (streaming)."""
        self._ensure_bound(data)
        key = (af_start, af_step, af_max)
        if key not in self._s_sar:
            self._s_sar[key] = StreamingSAR(af_start, af_step, af_max)
        return self._s_sar[key].update(data, index)

    def is_sar_bullish(
        self,
        data: Sequence[PriceData],
        index: int,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> bool:
        """Whether Parabolic SAR indicates uptrend (SAR below price) (streaming)."""
        self._ensure_bound(data)
        key = (af_start, af_step, af_max)
        if key not in self._s_sar:
            self._s_sar[key] = StreamingSAR(af_start, af_step, af_max)
        self._s_sar[key].update(data, index)
        return self._s_sar[key].is_bullish(data, index)

    def ichimoku(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> trend.IchimokuCloud:
        """Ichimoku Cloud components."""
        return self._get(
            ("ichimoku", index),
            data, trend.ichimoku, data, index,
        )

    # ==================================================================
    # CCI
    # ==================================================================

    def cci(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 20,
    ) -> Decimal:
        """Commodity Channel Index."""
        return self._get(
            ("cci", index, period),
            data, momentum.cci, data, index, period,
        )

    # ==================================================================
    # VWAP (daily)
    # ==================================================================

    def vwap(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int = 20,
    ) -> Decimal:
        """Volume-Weighted Average Price (rolling lookback)."""
        return self._get(
            ("vwap", index, lookback),
            data, trend.vwap, data, index, lookback,
        )

    # ==================================================================
    # Fibonacci
    # ==================================================================

    def fibonacci_retracement(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int = 50,
    ) -> trend.FibonacciLevels:
        """Fibonacci retracement levels."""
        return self._get(
            ("fibonacci_retracement", index, lookback),
            data, trend.fibonacci_retracement, data, index, lookback,
        )

    # ==================================================================
    # Support & Resistance
    # ==================================================================

    def support_resistance(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int,
        window: int,
    ) -> trend.SupportResistance:
        """Detect support and resistance levels."""
        return self._get(
            ("support_resistance", index, lookback, window),
            data, trend.support_resistance, data, index, lookback, window,
        )

