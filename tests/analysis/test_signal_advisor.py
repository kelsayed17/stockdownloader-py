"""Tests for the SignalAdvisor — regime-aware advisory engine."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.analysis.signal_advisor import (
    AdvisorConfig,
    SignalAdvisor,
    _compute_levels,
    _is_downgraded,
    _map_direction,
    _position_size,
)
from stockdownloader.model.alert_result import (
    Action,
    AlertDirection,
    AlertResult,
    OptionsRecommendation,
)
from stockdownloader.model.indicator_values import IndicatorValues
from stockdownloader.model.options import OptionType
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.signal_advisory import (
    AdvisoryAction,
    SignalAdvisory as SignalAdvisoryModel,
)
from stockdownloader.strategy.regime.regime_detector import (
    MarketRegime,
    RegimeClassification,
)
from stockdownloader.strategy.regime.regime_strategy_map import RegimeStrategyMapper
from stockdownloader.util.big_decimal_math import ZERO


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_bar(date: str, close: float, volume: int = 1_000_000) -> PriceData:
    c = Decimal(str(round(close, 4)))
    h = c + Decimal("1.0")
    l = c - Decimal("1.0")
    return PriceData(
        date=date, open=c, high=h, low=l, close=c, adj_close=c, volume=volume,
    )


def _make_data(n: int = 250, start: float = 100.0, step: float = 0.2) -> list[PriceData]:
    """Generate uptrend data with enough bars for generate_alert (>200)."""
    data: list[PriceData] = []
    price = start
    for i in range(n):
        price += step
        data.append(_make_bar(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", price))
    return data


def _make_hold_rec(opt_type: OptionType) -> OptionsRecommendation:
    return OptionsRecommendation(
        type=opt_type,
        action=Action.HOLD,
        suggested_strike=Decimal("100"),
        suggested_dte=0,
        estimated_premium=ZERO,
        target_delta=ZERO,
        rationale="Neutral",
    )


def _make_buy_rec(opt_type: OptionType) -> OptionsRecommendation:
    return OptionsRecommendation(
        type=opt_type,
        action=Action.BUY,
        suggested_strike=Decimal("150"),
        suggested_dte=30,
        estimated_premium=Decimal("5.00"),
        target_delta=Decimal("0.50"),
        rationale="Bullish setup",
    )


def _fake_alert(
    direction: AlertDirection = AlertDirection.BUY,
    confluence: float = 0.70,
    total: int = 15,
    price: Decimal = Decimal("150.00"),
) -> AlertResult:
    bullish = ["RSI oversold"] if direction in (AlertDirection.STRONG_BUY, AlertDirection.BUY) else []
    bearish = ["MACD bearish"] if direction in (AlertDirection.STRONG_SELL, AlertDirection.SELL) else []
    return AlertResult(
        symbol="SPY",
        date="2024-06-15",
        current_price=price,
        direction=direction,
        confluence_score=confluence,
        total_indicators=total,
        bullish_indicators=bullish,
        bearish_indicators=bearish,
        call_recommendation=_make_buy_rec(OptionType.CALL) if direction in (AlertDirection.STRONG_BUY, AlertDirection.BUY) else _make_hold_rec(OptionType.CALL),
        put_recommendation=_make_hold_rec(OptionType.PUT),
        support_levels=[Decimal("145"), Decimal("140")],
        resistance_levels=[Decimal("155"), Decimal("160")],
        indicators=None,
    )


def _fake_regime(
    regime: MarketRegime = MarketRegime.STRONG_TREND_UP,
    confidence: float = 0.85,
) -> RegimeClassification:
    return RegimeClassification(
        regime=regime,
        confidence=confidence,
        adx_value=35.0,
        bb_width_percentile=0.5,
        trend_slope=1.2,
        sma200_distance=5.0,
    )


# ------------------------------------------------------------------
# _map_direction tests
# ------------------------------------------------------------------


class TestMapDirection:
    def test_neutral_always_hold(self) -> None:
        for regime in MarketRegime:
            assert _map_direction(AlertDirection.NEUTRAL, regime) == AdvisoryAction.HOLD

    def test_strong_buy_aligned(self) -> None:
        assert _map_direction(AlertDirection.STRONG_BUY, MarketRegime.STRONG_TREND_UP) == AdvisoryAction.STRONG_BUY

    def test_strong_buy_downgraded_weak_trend(self) -> None:
        assert _map_direction(AlertDirection.STRONG_BUY, MarketRegime.WEAK_TREND) == AdvisoryAction.BUY

    def test_strong_buy_blocked_downtrend(self) -> None:
        assert _map_direction(AlertDirection.STRONG_BUY, MarketRegime.STRONG_TREND_DOWN) == AdvisoryAction.HOLD

    def test_buy_blocked_downtrend(self) -> None:
        assert _map_direction(AlertDirection.BUY, MarketRegime.STRONG_TREND_DOWN) == AdvisoryAction.HOLD

    def test_sell_blocked_uptrend(self) -> None:
        assert _map_direction(AlertDirection.SELL, MarketRegime.STRONG_TREND_UP) == AdvisoryAction.HOLD

    def test_strong_sell_aligned(self) -> None:
        assert _map_direction(AlertDirection.STRONG_SELL, MarketRegime.STRONG_TREND_DOWN) == AdvisoryAction.STRONG_SELL

    def test_strong_sell_uptrend_blocked(self) -> None:
        assert _map_direction(AlertDirection.STRONG_SELL, MarketRegime.STRONG_TREND_UP) == AdvisoryAction.HOLD


# ------------------------------------------------------------------
# _is_downgraded tests
# ------------------------------------------------------------------


class TestIsDowngraded:
    def test_strong_buy_to_buy(self) -> None:
        assert _is_downgraded(AlertDirection.STRONG_BUY, AdvisoryAction.BUY) is True

    def test_buy_to_buy(self) -> None:
        assert _is_downgraded(AlertDirection.BUY, AdvisoryAction.BUY) is False

    def test_strong_sell_to_sell(self) -> None:
        assert _is_downgraded(AlertDirection.STRONG_SELL, AdvisoryAction.SELL) is True

    def test_buy_to_hold(self) -> None:
        assert _is_downgraded(AlertDirection.BUY, AdvisoryAction.HOLD) is True

    def test_neutral_to_hold(self) -> None:
        assert _is_downgraded(AlertDirection.NEUTRAL, AdvisoryAction.HOLD) is False


# ------------------------------------------------------------------
# _compute_levels tests
# ------------------------------------------------------------------


class TestComputeLevels:
    def test_buy_levels(self) -> None:
        cfg = AdvisorConfig(atr_stop_multiplier=2.0, atr_target_multiplier=3.0)
        sl, tp, rr = _compute_levels(100.0, 5.0, AdvisoryAction.BUY, cfg)
        assert sl == pytest.approx(90.0)   # 100 - 2*5
        assert tp == pytest.approx(115.0)  # 100 + 3*5
        assert rr == pytest.approx(1.5)    # 15 / 10

    def test_sell_levels(self) -> None:
        cfg = AdvisorConfig(atr_stop_multiplier=2.0, atr_target_multiplier=3.0)
        sl, tp, rr = _compute_levels(100.0, 5.0, AdvisoryAction.SELL, cfg)
        assert sl == pytest.approx(110.0)  # 100 + 2*5
        assert tp == pytest.approx(85.0)   # 100 - 3*5
        assert rr == pytest.approx(1.5)

    def test_hold_returns_flat(self) -> None:
        cfg = AdvisorConfig()
        sl, tp, rr = _compute_levels(100.0, 5.0, AdvisoryAction.HOLD, cfg)
        assert sl == 100.0
        assert tp == 100.0
        assert rr == 0.0

    def test_zero_atr(self) -> None:
        cfg = AdvisorConfig()
        sl, tp, rr = _compute_levels(100.0, 0.0, AdvisoryAction.BUY, cfg)
        assert rr == 0.0


# ------------------------------------------------------------------
# _position_size tests
# ------------------------------------------------------------------


class TestPositionSize:
    def test_full_confidence(self) -> None:
        assert _position_size(1.0, 2.0) == pytest.approx(2.0)

    def test_half_confidence(self) -> None:
        assert _position_size(0.5, 2.0) == pytest.approx(1.0)

    def test_capped(self) -> None:
        assert _position_size(1.5, 2.0) == pytest.approx(2.0)


# ------------------------------------------------------------------
# SignalAdvisor.evaluate integration tests (mocked dependencies)
# ------------------------------------------------------------------


class TestSignalAdvisorEvaluate:
    """Tests using mocked generate_alert and regime detector."""

    def _make_advisor(
        self,
        wf_scores: dict[str, tuple[float, float]] | None = None,
    ) -> SignalAdvisor:
        return SignalAdvisor(
            config=AdvisorConfig(),
            walk_forward_scores=wf_scores,
        )

    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_insufficient_data_returns_hold(self, mock_alert: MagicMock) -> None:
        advisor = self._make_advisor()
        data = _make_data(100)  # < 201 bars
        result = advisor.evaluate("SPY", data)
        assert result.action == AdvisoryAction.HOLD
        assert result.confidence == 0.0
        mock_alert.assert_not_called()

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_buy_aligned_with_uptrend(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        advisor = self._make_advisor()
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.action == AdvisoryAction.BUY
        assert result.confidence > 0.0
        assert result.regime == "strong_trend_up"
        assert result.symbol == "SPY"

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_buy_blocked_by_downtrend(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.65)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_DOWN)

        advisor = self._make_advisor()
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        # BUY + STRONG_TREND_DOWN → HOLD, then regime penalty may push below threshold
        assert result.action == AdvisoryAction.HOLD

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_strong_sell_aligned(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.STRONG_SELL, confluence=0.85)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_DOWN)

        advisor = self._make_advisor()
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.action == AdvisoryAction.STRONG_SELL
        assert result.confidence > 0.6

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_low_confidence_forced_hold(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.30)
        mock_classify.return_value = _fake_regime(MarketRegime.WEAK_TREND, confidence=0.4)

        advisor = SignalAdvisor(config=AdvisorConfig(min_confidence=0.5))
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.action == AdvisoryAction.HOLD

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_walk_forward_boost(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        # Good WF score → 1.15 boost
        advisor = self._make_advisor(wf_scores={"rsi": (120.0, 130.0)})
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.confidence > 0.70  # boosted beyond raw

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_walk_forward_penalty(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        # Bad WF score → 0.7 penalty
        advisor = self._make_advisor(wf_scores={"overfit": (30.0, 100.0)})
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.confidence < 0.70  # penalized below raw

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_stop_loss_and_target(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70, price=Decimal("150"))
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        advisor = self._make_advisor()
        data = _make_data(250, start=140.0, step=0.05)
        result = advisor.evaluate("SPY", data)

        # Stop should be below entry, target above
        assert result.stop_loss < result.entry_price
        assert result.take_profit > result.entry_price
        assert result.risk_reward > 0

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_options_conversion(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        alert = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_alert.return_value = alert
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        advisor = self._make_advisor()
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        # Call should be converted (BUY action), put is HOLD → None
        assert result.call_advisory is not None
        assert result.call_advisory.action == "BUY"
        assert result.put_advisory is None

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_position_sizing(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.80)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        cfg = AdvisorConfig(max_position_pct=2.0)
        advisor = SignalAdvisor(config=cfg)
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert 0 < result.position_size_pct <= 2.0

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_reasoning_populated(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        advisor = self._make_advisor()
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.reasoning.signal_confluence != ""
        assert "strong_trend_up" in result.reasoning.regime_alignment.lower()
        assert len(result.reasoning.support_levels) > 0

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_to_json_serializable(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        advisor = self._make_advisor()
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        # Should not raise
        import json
        parsed = json.loads(result.to_json())
        assert parsed["symbol"] == "SPY"
        assert parsed["action"] == "BUY"


# ------------------------------------------------------------------
# AdvisorConfig
# ------------------------------------------------------------------


class TestAdvisorConfig:
    def test_defaults(self) -> None:
        cfg = AdvisorConfig()
        assert cfg.atr_stop_multiplier == 2.0
        assert cfg.min_confidence == 0.4
        assert cfg.max_position_pct == 2.0

    def test_ml_defaults(self) -> None:
        cfg = AdvisorConfig()
        assert cfg.ml_model_path is None
        assert cfg.ml_boost_max == 0.3
        assert cfg.ml_penalty_max == 0.3
        assert cfg.ml_min_probability == 0.6

    def test_frozen(self) -> None:
        cfg = AdvisorConfig()
        with pytest.raises(AttributeError):
            cfg.min_confidence = 0.99  # type: ignore[misc]


# ------------------------------------------------------------------
# ML integration tests
# ------------------------------------------------------------------


class TestSignalAdvisorML:
    """Tests for ML confidence modifier in SignalAdvisor."""

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_no_ml_model_preserves_behavior(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        """Without ML model, confidence should be unaffected."""
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        advisor = SignalAdvisor(config=AdvisorConfig())
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        # Raw 0.70 * regime_factor 1.0 * wf_factor 1.0 * ml_factor 1.0 = 0.70
        assert result.confidence == pytest.approx(0.70, abs=0.01)

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_ml_boost_on_buy_agreement(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        """ML predicts profitable (prob=0.8) + BUY signal → boosted confidence."""
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        mock_predictor = MagicMock()
        mock_predictor.predict_proba.return_value = 0.8

        advisor = SignalAdvisor(
            config=AdvisorConfig(),
            ml_predictor=mock_predictor,
        )
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.confidence > 0.70  # boosted

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_ml_penalty_on_buy_disagreement(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        """ML predicts unprofitable (prob=0.2) + BUY signal → penalized."""
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        mock_predictor = MagicMock()
        mock_predictor.predict_proba.return_value = 0.2

        advisor = SignalAdvisor(
            config=AdvisorConfig(),
            ml_predictor=mock_predictor,
        )
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.confidence < 0.70  # penalized

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_ml_neutral_prob_no_effect(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        """ML returns 0.5 (uncertain) → no modification."""
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.70)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_UP)

        mock_predictor = MagicMock()
        mock_predictor.predict_proba.return_value = 0.5

        advisor = SignalAdvisor(
            config=AdvisorConfig(),
            ml_predictor=mock_predictor,
        )
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        # 0.5 is in the neutral zone → factor = 1.0
        assert result.confidence == pytest.approx(0.70, abs=0.01)

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_ml_boost_on_sell_agreement(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        """ML predicts unprofitable (prob=0.2) + SELL signal → boosted."""
        mock_alert.return_value = _fake_alert(AlertDirection.STRONG_SELL, confluence=0.85)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_DOWN)

        mock_predictor = MagicMock()
        mock_predictor.predict_proba.return_value = 0.2

        advisor = SignalAdvisor(
            config=AdvisorConfig(),
            ml_predictor=mock_predictor,
        )
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        assert result.confidence > 0.85  # boosted

    @patch("stockdownloader.analysis.signal_advisor.MarketRegimeDetector.classify")
    @patch("stockdownloader.analysis.signal_advisor.generate_alert")
    def test_ml_hold_action_no_adjustment(
        self, mock_alert: MagicMock, mock_classify: MagicMock,
    ) -> None:
        """HOLD action → ML factor is 1.0 regardless of prob."""
        mock_alert.return_value = _fake_alert(AlertDirection.BUY, confluence=0.20)
        mock_classify.return_value = _fake_regime(MarketRegime.STRONG_TREND_DOWN)

        mock_predictor = MagicMock()
        mock_predictor.predict_proba.return_value = 0.9

        advisor = SignalAdvisor(
            config=AdvisorConfig(min_confidence=0.0),
            ml_predictor=mock_predictor,
        )
        data = _make_data(250)
        result = advisor.evaluate("SPY", data)

        # BUY + STRONG_TREND_DOWN → forced HOLD by regime
        assert result.action == AdvisoryAction.HOLD
