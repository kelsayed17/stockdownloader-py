"""Tests for EnsembleIntradayStrategy and DrawdownPositionScaler.

Validates regime-based strategy switching, drawdown scaling,
session boundary handling, and edge cases.
"""

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import (
    HOLD,
    IntradayAction,
    IntradaySignal,
)
from stockdownloader.strategy.regime.ensemble_strategy import (
    DrawdownPositionScaler,
    EnsembleIntradayStrategy,
)
from stockdownloader.strategy.regime.regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeClassification,
)
from stockdownloader.util.indicators.hub import IndicatorHub


# ---------------------------------------------------------------------------
# DrawdownPositionScaler
# ---------------------------------------------------------------------------


class TestDrawdownPositionScaler:
    def test_full_size_at_zero_dd(self):
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)
        assert scaler.scale_factor(0.0) == 1.0

    def test_half_size_at_half_dd(self):
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)
        assert scaler.scale_factor(5.0) == 0.5

    def test_zero_size_at_zero_dd_threshold(self):
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)
        assert scaler.scale_factor(10.0) == 0.0

    def test_zero_size_beyond_threshold(self):
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)
        assert scaler.scale_factor(15.0) == 0.0

    def test_linear_interpolation_first_half(self):
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)
        # At 2.5% DD → halfway between 1.0 and 0.5 → 0.75
        assert scaler.scale_factor(2.5) == pytest.approx(0.75, abs=0.01)

    def test_linear_interpolation_second_half(self):
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)
        # At 7.5% DD → halfway between 0.5 and 0.0 → 0.25
        assert scaler.scale_factor(7.5) == pytest.approx(0.25, abs=0.01)

    def test_custom_thresholds(self):
        scaler = DrawdownPositionScaler(1.0, 3.0, 8.0)
        assert scaler.scale_factor(0.5) == 1.0  # below full_dd
        assert scaler.scale_factor(3.0) == 0.5
        assert scaler.scale_factor(8.0) == 0.0

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError, match="Thresholds must be"):
            DrawdownPositionScaler(5.0, 3.0, 10.0)


# ---------------------------------------------------------------------------
# EnsembleIntradayStrategy helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 300) -> list[IntradayPriceData]:
    """Generate minimal synthetic data."""
    data = []
    bars_per_day = 78
    for i in range(n):
        day = i // bars_per_day
        bar = i % bars_per_day
        hour = 9 + bar // 12
        minute = 30 + (bar % 12) * 5
        if minute >= 60:
            hour += 1
            minute -= 60
        date_str = f"2025-01-{(day % 28) + 1:02d} {hour:02d}:{minute:02d}:00-05:00"
        price = Decimal(str(100 + i * 0.01))
        data.append(IntradayPriceData(
            date=date_str,
            open=price,
            high=price + Decimal("0.5"),
            low=price - Decimal("0.5"),
            close=price,
            adj_close=price,
            volume=1_000_000,
        ))
    return data


def _make_mock_strategy(name: str, signal: IntradaySignal = HOLD):
    """Create a mock IntradayTradingStrategy."""
    s = MagicMock()
    type(s).name = PropertyMock(return_value=name)
    type(s).warmup_period = PropertyMock(return_value=10)
    s.evaluate.return_value = signal
    return s


def _make_mock_detector(regime: MarketRegime = MarketRegime.WEAK_TREND):
    """Create a mock MarketRegimeDetector."""
    detector = MagicMock(spec=MarketRegimeDetector)
    detector.warmup_period = 200
    detector.classify.return_value = RegimeClassification(
        regime=regime,
        confidence=0.8,
        adx_value=25.0,
        bb_width_percentile=0.5,
        trend_slope=0.3,
        sma200_distance=2.0,
    )
    return detector


# ---------------------------------------------------------------------------
# EnsembleIntradayStrategy tests
# ---------------------------------------------------------------------------


class TestEnsembleStrategy:
    def test_construction(self):
        default = _make_mock_strategy("default")
        detector = _make_mock_detector()
        ensemble = EnsembleIntradayStrategy(
            sub_strategies={},
            regime_detector=detector,
            default_strategy=default,
        )
        assert ensemble.name == "Ensemble"

    def test_warmup_period_max_of_all(self):
        s1 = _make_mock_strategy("s1")
        type(s1).warmup_period = PropertyMock(return_value=50)
        s2 = _make_mock_strategy("s2")
        type(s2).warmup_period = PropertyMock(return_value=100)
        default = _make_mock_strategy("default")
        type(default).warmup_period = PropertyMock(return_value=30)
        detector = _make_mock_detector()
        detector.warmup_period = 200

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={
                MarketRegime.MEAN_REVERTING: s1,
                MarketRegime.STRONG_TREND_UP: s2,
            },
            regime_detector=detector,
            default_strategy=default,
        )
        assert ensemble.warmup_period == 200

    def test_selects_correct_strategy_for_regime(self):
        """Ensemble should delegate to the sub-strategy for the detected regime."""
        trend_signal = IntradaySignal(
            action=IntradayAction.ENTER_LONG,
            risk_per_share=Decimal("1.50"),
            reason="trend entry",
        )
        mean_rev_signal = IntradaySignal(
            action=IntradayAction.ENTER_SHORT,
            risk_per_share=Decimal("1.00"),
            reason="mean reversion entry",
        )

        trend_strategy = _make_mock_strategy("sma", trend_signal)
        mean_rev_strategy = _make_mock_strategy("rsi", mean_rev_signal)
        default = _make_mock_strategy("default")

        # Detect MEAN_REVERTING
        detector = _make_mock_detector(MarketRegime.MEAN_REVERTING)

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={
                MarketRegime.STRONG_TREND_UP: trend_strategy,
                MarketRegime.MEAN_REVERTING: mean_rev_strategy,
            },
            regime_detector=detector,
            default_strategy=default,
        )

        data = _make_data(250)
        signal = ensemble.evaluate(data, 210)

        # Should have used the mean_rev_strategy
        mean_rev_strategy.evaluate.assert_called()
        trend_strategy.evaluate.assert_not_called()

    def test_falls_back_to_default(self):
        """When no sub-strategy exists for detected regime, use default."""
        default_signal = IntradaySignal(
            action=IntradayAction.ENTER_LONG,
            risk_per_share=Decimal("2.00"),
        )
        default = _make_mock_strategy("default", default_signal)

        detector = _make_mock_detector(MarketRegime.HIGH_VOLATILITY)

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={
                MarketRegime.MEAN_REVERTING: _make_mock_strategy("rsi"),
            },
            regime_detector=detector,
            default_strategy=default,
        )

        data = _make_data(250)
        signal = ensemble.evaluate(data, 210)

        default.evaluate.assert_called()

    def test_hold_during_warmup(self):
        default = _make_mock_strategy("default")
        detector = _make_mock_detector()
        detector.warmup_period = 200

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={},
            regime_detector=detector,
            default_strategy=default,
        )

        data = _make_data(250)
        signal = ensemble.evaluate(data, 50)  # below warmup
        assert signal == HOLD

    def test_session_start_propagation(self):
        s1 = _make_mock_strategy("s1")
        default = _make_mock_strategy("default")
        detector = _make_mock_detector()

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={MarketRegime.WEAK_TREND: s1},
            regime_detector=detector,
            default_strategy=default,
        )

        ensemble.on_session_start("2025-01-15")
        s1.on_session_start.assert_called_with("2025-01-15")
        default.on_session_start.assert_called_with("2025-01-15")

    def test_drawdown_scaling_reduces_risk(self):
        """When in drawdown, risk_per_share should be scaled down."""
        entry_signal = IntradaySignal(
            action=IntradayAction.ENTER_LONG,
            risk_per_share=Decimal("2.00"),
            reason="test entry",
        )
        default = _make_mock_strategy("default", entry_signal)
        detector = _make_mock_detector()
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={},
            regime_detector=detector,
            default_strategy=default,
            drawdown_scaler=scaler,
        )

        # Simulate 5% drawdown
        ensemble.update_equity(Decimal("100000"))
        ensemble.update_equity(Decimal("95000"))

        data = _make_data(250)
        signal = ensemble.evaluate(data, 210)

        # At 5% DD, scale = 0.5, so risk should be ~1.00
        assert signal.action == IntradayAction.ENTER_LONG
        assert signal.risk_per_share == Decimal("1.00")

    def test_drawdown_scaling_blocks_at_zero_dd(self):
        """When drawdown exceeds zero threshold, should return HOLD."""
        entry_signal = IntradaySignal(
            action=IntradayAction.ENTER_LONG,
            risk_per_share=Decimal("2.00"),
        )
        default = _make_mock_strategy("default", entry_signal)
        detector = _make_mock_detector()
        scaler = DrawdownPositionScaler(0.0, 5.0, 10.0)

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={},
            regime_detector=detector,
            default_strategy=default,
            drawdown_scaler=scaler,
        )

        # Simulate 12% drawdown (beyond zero threshold)
        ensemble.update_equity(Decimal("100000"))
        ensemble.update_equity(Decimal("88000"))

        data = _make_data(250)
        signal = ensemble.evaluate(data, 210)

        assert signal == HOLD

    def test_update_equity_tracks_peak(self):
        default = _make_mock_strategy("default")
        detector = _make_mock_detector()
        ensemble = EnsembleIntradayStrategy(
            sub_strategies={},
            regime_detector=detector,
            default_strategy=default,
        )

        ensemble.update_equity(Decimal("100000"))
        ensemble.update_equity(Decimal("105000"))
        ensemble.update_equity(Decimal("103000"))

        # Peak should be 105000, DD should be ~1.9%
        assert ensemble._peak_equity == Decimal("105000")
        assert ensemble._current_dd_pct == pytest.approx(1.905, abs=0.01)

    def test_no_scaler_passes_signal_through(self):
        """Without a scaler, entry signals pass through unchanged."""
        entry_signal = IntradaySignal(
            action=IntradayAction.ENTER_LONG,
            risk_per_share=Decimal("2.00"),
        )
        default = _make_mock_strategy("default", entry_signal)
        detector = _make_mock_detector()

        ensemble = EnsembleIntradayStrategy(
            sub_strategies={},
            regime_detector=detector,
            default_strategy=default,
            drawdown_scaler=None,
        )

        data = _make_data(250)
        signal = ensemble.evaluate(data, 210)

        assert signal.action == IntradayAction.ENTER_LONG
        assert signal.risk_per_share == Decimal("2.00")
