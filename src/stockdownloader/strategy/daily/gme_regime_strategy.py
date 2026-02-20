"""GME Quantitative Regime Prediction strategy for backtesting.

Python implementation of the PineScript GME-QRP indicator, adapted for
the :class:`BacktestEngine` (long-only with BUY/SELL signals).

Encodes five statistically validated patterns:

1. **Volatility regime detection** — 4 regimes via rolling realized vol
2. **Volume anomaly detection** — z-score > threshold on rolling window
3. **OBV divergence** — price/OBV divergence signals potential reversals
4. **Volatility clustering** — consecutive large moves signal continuation
5. **ATR-scaled dynamic exits** — wider stops in high-vol regimes

Entry:
    BUY when Normal/High regime + volume anomaly + OBV rising + RSI < 70
    + (volatility cluster starting OR regime transitioning up)

Exit:
    SELL when Extreme regime (risk-off) OR OBV bearish divergence
    OR RSI > 80 OR dynamic ATR stop hit
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


# Regime constants
_LOW = 0
_NORMAL = 1
_HIGH = 2
_EXTREME = 3


class GmeRegimeStrategy(TradingStrategy):
    """Regime-adaptive prediction strategy based on GME quantitative analysis.

    Parameters
    ----------
    vol_window:
        Rolling window for realized volatility (default 20).
    vol_anomaly_window:
        Lookback for volume z-score (default 60).
    vol_anomaly_z:
        Z-score threshold for volume anomaly (default 3.0).
    obv_lookback:
        OBV smoothing + divergence lookback (default 14).
    cluster_threshold:
        Large-move threshold as multiple of avg |return| (default 2.0).
    cluster_count:
        Min consecutive large moves to activate cluster (default 2).
    regime_high_mult:
        Multiplier on median vol for High regime threshold (default 1.5).
    regime_extreme_mult:
        Multiplier on median vol for Extreme regime threshold (default 2.5).
    rsi_period:
        RSI period (default 14).
    atr_period:
        ATR period for dynamic stops (default 14).
    stop_mult:
        ATR multiplier for stop-loss (default 2.0).
    """

    def __init__(
        self,
        vol_window: int = 20,
        vol_anomaly_window: int = 60,
        vol_anomaly_z: float = 3.0,
        obv_lookback: int = 14,
        cluster_threshold: float = 2.0,
        cluster_count: int = 2,
        regime_high_mult: float = 1.5,
        regime_extreme_mult: float = 2.5,
        rsi_period: int = 14,
        atr_period: int = 14,
        stop_mult: float = 2.0,
    ) -> None:
        self._vol_window = vol_window
        self._vol_anomaly_window = vol_anomaly_window
        self._vol_anomaly_z = vol_anomaly_z
        self._obv_lookback = obv_lookback
        self._cluster_threshold = cluster_threshold
        self._cluster_count = cluster_count
        self._regime_high_mult = regime_high_mult
        self._regime_extreme_mult = regime_extreme_mult
        self._rsi_period = rsi_period
        self._atr_period = atr_period
        self._stop_mult = stop_mult

        self._hub = IndicatorHub()

        # State for volatility clustering counter
        self._cluster_len = 0

        # State for tracking position and stop
        self._in_position = False
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0

    @property
    def name(self) -> str:
        return "GME Regime Prediction"

    @property
    def warmup_period(self) -> int:
        # Need enough bars for: vol_anomaly_window + vol_window + some buffer
        return max(self._vol_anomaly_window, 200) + self._vol_window

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        """Evaluate the regime-adaptive prediction logic at *current_index*."""
        if current_index < self.warmup_period:
            return Signal.HOLD

        # === Core Indicators ===
        rsi = float(self._hub.rsi(data, current_index, self._rsi_period))
        atr = float(self._hub.atr(data, current_index, self._atr_period))
        obv_rising = self._hub.is_obv_rising(data, current_index, lookback=5)

        close = float(data[current_index].close)
        prev_close = float(data[current_index - 1].close)

        # === Realized Volatility & Regime Classification ===
        regime = self._classify_regime(data, current_index)
        prev_regime = self._classify_regime(data, current_index - 1)
        regime_up = regime > prev_regime and prev_regime != _LOW

        # === Volume Anomaly (z-score) ===
        vol_anomaly = self._check_volume_anomaly(data, current_index)

        # === Volatility Clustering ===
        cluster_active = self._update_cluster(data, current_index)

        # === OBV Divergence ===
        obv_bear_div = self._obv_bearish_divergence(data, current_index)

        # === EXIT LOGIC (check first if in position) ===
        if self._in_position:
            # Exit conditions:
            # 1. Extreme regime → risk-off
            # 2. OBV bearish divergence
            # 3. RSI overbought (> 80)
            # 4. Dynamic ATR stop hit
            if regime == _EXTREME:
                self._in_position = False
                return Signal.SELL

            if obv_bear_div:
                self._in_position = False
                return Signal.SELL

            if rsi > 80:
                self._in_position = False
                return Signal.SELL

            # Dynamic ATR stop (scaled by regime)
            stop_scale = self._regime_stop_scale(regime)
            dynamic_stop = atr * self._stop_mult * stop_scale
            self._stop_price = self._entry_price - dynamic_stop
            if close < self._stop_price:
                self._in_position = False
                return Signal.SELL

            return Signal.HOLD

        # === ENTRY LOGIC (not in position) ===
        # Must be in Normal or High regime
        if regime not in (_NORMAL, _HIGH):
            return Signal.HOLD

        # Volume anomaly required
        if not vol_anomaly:
            return Signal.HOLD

        # OBV must be rising
        if not obv_rising:
            return Signal.HOLD

        # RSI not overbought
        if rsi >= 70:
            return Signal.HOLD

        # Cluster starting OR regime transitioning up
        if not (cluster_active or regime_up):
            return Signal.HOLD

        # All conditions met → BUY
        self._in_position = True
        self._entry_price = close
        stop_scale = self._regime_stop_scale(regime)
        self._stop_price = close - (atr * self._stop_mult * stop_scale)
        return Signal.BUY

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _classify_regime(self, data: list[PriceData], index: int) -> int:
        """Compute rolling realized vol and classify into regime 0-3."""
        if index < self._vol_window + 1:
            return _NORMAL

        # Compute log returns for the rolling window
        log_returns: list[float] = []
        for j in range(index - self._vol_window + 1, index + 1):
            if j < 1:
                continue
            c = float(data[j].close)
            p = float(data[j - 1].close)
            if p > 0 and c > 0:
                log_returns.append(math.log(c / p))

        if len(log_returns) < 2:
            return _NORMAL

        # Realized vol (annualized)
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        realized_vol = math.sqrt(variance) * math.sqrt(252)

        # Median vol approximation: use a longer lookback
        hist_vols: list[float] = []
        lookback = min(200, index - self._vol_window)
        for k in range(max(0, index - lookback), index + 1):
            if k < self._vol_window + 1:
                continue
            lr: list[float] = []
            for j in range(k - self._vol_window + 1, k + 1):
                if j < 1:
                    continue
                c = float(data[j].close)
                p = float(data[j - 1].close)
                if p > 0 and c > 0:
                    lr.append(math.log(c / p))
            if len(lr) >= 2:
                m = sum(lr) / len(lr)
                v = sum((r - m) ** 2 for r in lr) / (len(lr) - 1)
                hist_vols.append(math.sqrt(v) * math.sqrt(252))

        if not hist_vols:
            return _NORMAL

        hist_vols.sort()
        median_vol = hist_vols[len(hist_vols) // 2]

        if median_vol <= 0:
            return _NORMAL

        # Classify
        extreme_threshold = median_vol * self._regime_extreme_mult
        high_threshold = median_vol * self._regime_high_mult
        low_threshold = median_vol * 0.5

        if realized_vol >= extreme_threshold:
            return _EXTREME
        if realized_vol >= high_threshold:
            return _HIGH
        if realized_vol <= low_threshold:
            return _LOW
        return _NORMAL

    def _check_volume_anomaly(self, data: list[PriceData], index: int) -> bool:
        """Check if current volume is a z-score anomaly."""
        if index < self._vol_anomaly_window:
            return False

        volumes: list[float] = []
        for j in range(index - self._vol_anomaly_window, index):
            volumes.append(float(data[j].volume))

        if not volumes:
            return False

        mean_v = sum(volumes) / len(volumes)
        if len(volumes) < 2:
            return False

        variance_v = sum((v - mean_v) ** 2 for v in volumes) / (len(volumes) - 1)
        std_v = math.sqrt(variance_v)

        if std_v <= 0:
            return False

        current_vol = float(data[index].volume)
        z_score = (current_vol - mean_v) / std_v
        return z_score > self._vol_anomaly_z

    def _update_cluster(self, data: list[PriceData], index: int) -> bool:
        """Update volatility cluster counter and return whether cluster is active."""
        if index < 21:
            self._cluster_len = 0
            return False

        # Current absolute log return
        c = float(data[index].close)
        p = float(data[index - 1].close)
        if p <= 0 or c <= 0:
            self._cluster_len = 0
            return False

        abs_ret = abs(math.log(c / p))

        # Average absolute return over 20 bars
        abs_returns: list[float] = []
        for j in range(index - 20, index):
            if j < 1:
                continue
            cj = float(data[j].close)
            pj = float(data[j - 1].close)
            if pj > 0 and cj > 0:
                abs_returns.append(abs(math.log(cj / pj)))

        if not abs_returns:
            self._cluster_len = 0
            return False

        avg_abs_ret = sum(abs_returns) / len(abs_returns)

        if abs_ret > avg_abs_ret * self._cluster_threshold:
            self._cluster_len += 1
        else:
            self._cluster_len = 0

        return self._cluster_len >= self._cluster_count

    def _obv_bearish_divergence(self, data: list[PriceData], index: int) -> bool:
        """Check if price is at recent high but OBV is not confirming."""
        lookback = self._obv_lookback
        if index < lookback:
            return False

        # Current high and OBV
        current_high = float(data[index].high)
        current_obv = float(self._hub.obv(data, index))

        # Highest high and highest OBV in lookback
        max_high = max(float(data[j].high) for j in range(index - lookback, index + 1))
        max_obv = max(float(self._hub.obv(data, j)) for j in range(index - lookback, index + 1))

        # Bearish divergence: price at/near high but OBV well below its high
        if max_obv == 0:
            return False
        return current_high >= max_high and current_obv < max_obv * 0.95

    @staticmethod
    def _regime_stop_scale(regime: int) -> float:
        """Return the ATR stop multiplier scale for the current regime."""
        if regime == _EXTREME:
            return 3.0
        if regime == _HIGH:
            return 2.0
        if regime == _LOW:
            return 1.0
        return 1.5  # Normal
