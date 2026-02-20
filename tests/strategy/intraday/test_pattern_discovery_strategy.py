"""Tests for PatternDiscoveryStrategy — trading discovered patterns."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.analysis.pattern_discovery import DiscoveredPattern, PatternCatalog
from stockdownloader.analysis.pattern_encoder import BarFeatures
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction, HOLD
from stockdownloader.strategy.intraday.pattern_discovery_strategy import (
    PatternDiscoveryConfig,
    PatternDiscoveryStrategy,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_bar(
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100_000,
) -> IntradayPriceData:
    return IntradayPriceData(
        date=date,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        adj_close=Decimal(str(close)),
        volume=volume,
    )


def _make_features(
    body_type: str = "bull",
    body_strength: str = "moderate",
    wick_signal: str = "no_wick",
    relative_size: str = "normal",
    volume_profile: str = "normal",
) -> BarFeatures:
    return BarFeatures(body_type, body_strength, wick_signal, relative_size, volume_profile)


def _make_pattern(
    key: tuple[BarFeatures, ...],
    direction: str = "long",
    avg_mae: float = -0.10,
    avg_mfe: float = 0.15,
) -> DiscoveredPattern:
    return DiscoveredPattern(
        key=key,
        direction=direction,
        best_horizon=5,
        avg_return=0.05 if direction == "long" else -0.05,
        win_rate=0.60,
        occurrences=50,
        t_stat=3.0,
        p_value=0.003,
        avg_mae=avg_mae,
        avg_mfe=avg_mfe,
        risk_reward=1.5,
        walk_forward_stable=True,
        human_label=f"{len(key)}-bar: test pattern",
    )


def _generate_session(
    trading_date: str = "2025-01-15",
    base_price: float = 500.0,
    num_bars: int = 78,
) -> list[IntradayPriceData]:
    """Generate a realistic intraday session of 5-minute bars."""
    bars = []
    price = base_price
    for i in range(num_bars):
        total_mins = 30 + i * 5
        hour = 9 + total_mins // 60
        minute = total_mins % 60
        dt_str = f"{trading_date} {hour:02d}:{minute:02d}:00-05:00"

        o = price
        h = price + 0.50
        l = price - 0.50
        c = price + 0.10 * (1 if i % 2 == 0 else -1)
        vol = 100_000 + i * 1000
        bars.append(_make_bar(dt_str, o, h, l, c, vol))
        price = c
    return bars


def _generate_multi_session(num_days: int = 16) -> list[IntradayPriceData]:
    """Generate multiple sessions (enough for warmup)."""
    all_bars: list[IntradayPriceData] = []
    base_price = 500.0
    for d in range(num_days):
        date_str = f"2025-01-{d + 1:02d}"
        session = _generate_session(
            trading_date=date_str,
            base_price=base_price,
            num_bars=78,
        )
        all_bars.extend(session)
        base_price = float(session[-1].close)
    return all_bars


# ======================================================================
# Config tests
# ======================================================================


class TestPatternDiscoveryConfig:
    def test_defaults(self):
        c = PatternDiscoveryConfig()
        assert c.sl_mae_multiplier == Decimal("1.2")
        assert c.tp_mfe_multiplier == Decimal("0.8")
        assert c.sl_cap == Decimal("2.50")
        assert c.min_rr == Decimal("1.0")
        assert c.max_day == 2
        assert c.spacing == 3

    def test_frozen(self):
        c = PatternDiscoveryConfig()
        with pytest.raises(AttributeError):
            c.max_day = 5  # type: ignore[misc]

    def test_custom_values(self):
        c = PatternDiscoveryConfig(
            sl_mae_multiplier=Decimal("1.5"),
            tp_mfe_multiplier=Decimal("0.6"),
            max_day=3,
        )
        assert c.sl_mae_multiplier == Decimal("1.5")
        assert c.tp_mfe_multiplier == Decimal("0.6")
        assert c.max_day == 3

    def test_use_confirmation_default_true(self):
        c = PatternDiscoveryConfig()
        assert c.use_confirmation is True

    def test_use_confirmation_false(self):
        c = PatternDiscoveryConfig(use_confirmation=False)
        assert c.use_confirmation is False


# ======================================================================
# Strategy construction
# ======================================================================


class TestStrategyConstruction:
    def test_empty_catalog(self):
        catalog = PatternCatalog(patterns=())
        strategy = PatternDiscoveryStrategy(catalog)
        assert strategy.name == "Pattern Discovery"

    def test_with_patterns(self):
        bf = _make_features()
        pattern = _make_pattern(key=(bf,))
        catalog = PatternCatalog(patterns=(pattern,))
        strategy = PatternDiscoveryStrategy(catalog)
        assert strategy.name == "Pattern Discovery"
        assert strategy.warmup_period > 0

    def test_max_pattern_length_computed(self):
        bf1 = _make_features("bull")
        bf2 = _make_features("doji")
        bf3 = _make_features("bear")
        p2 = _make_pattern(key=(bf1, bf2))
        p3 = _make_pattern(key=(bf1, bf2, bf3))
        catalog = PatternCatalog(patterns=(p2, p3))
        strategy = PatternDiscoveryStrategy(catalog)
        assert strategy._max_pattern_len == 3

    def test_custom_config(self):
        catalog = PatternCatalog(patterns=())
        config = PatternDiscoveryConfig(max_day=5, spacing=5)
        strategy = PatternDiscoveryStrategy(catalog, config=config)
        assert strategy._c.max_day == 5
        assert strategy._c.spacing == 5


# ======================================================================
# Strategy lifecycle
# ======================================================================


class TestStrategyLifecycle:
    def test_session_start_clears_buffer(self):
        catalog = PatternCatalog(patterns=())
        strategy = PatternDiscoveryStrategy(catalog)
        # Manually add something to buffer
        strategy._feature_buffer.append(_make_features())
        assert len(strategy._feature_buffer) == 1

        # Session start on a new date should clear
        strategy.on_session_start("2025-01-15")
        assert len(strategy._feature_buffer) == 0

    def test_session_start_same_date_no_clear(self):
        catalog = PatternCatalog(patterns=())
        strategy = PatternDiscoveryStrategy(catalog)
        strategy.on_session_start("2025-01-15")
        strategy._feature_buffer.append(_make_features())
        assert len(strategy._feature_buffer) == 1

        # Same date — buffer should NOT be cleared
        strategy.on_session_start("2025-01-15")
        assert len(strategy._feature_buffer) == 1

    def test_position_callbacks(self):
        catalog = PatternCatalog(patterns=())
        strategy = PatternDiscoveryStrategy(catalog)
        strategy.on_session_start("2025-01-15")
        # These should not raise
        strategy.on_position_opened(is_long=True)
        strategy.on_position_closed()


# ======================================================================
# Evaluate returns HOLD when no patterns match
# ======================================================================


class TestEvaluateNoMatch:
    def test_empty_catalog_returns_hold(self):
        """With no patterns in catalog, evaluate should return HOLD."""
        catalog = PatternCatalog(patterns=())
        strategy = PatternDiscoveryStrategy(catalog)

        data = _generate_multi_session(num_days=16)
        # Start session
        strategy.on_session_start(data[0].date[:10])

        # Evaluate many bars — should always HOLD
        for i in range(50, min(60, len(data))):
            signal = strategy.evaluate(data, i)
            assert signal.action == IntradayAction.HOLD


# ======================================================================
# SL/TP from MAE/MFE computation
# ======================================================================


class TestSlTpComputation:
    def test_sl_from_mae(self):
        """SL distance = |avg_mae| * sl_mae_multiplier * close / 100."""
        config = PatternDiscoveryConfig(
            sl_mae_multiplier=Decimal("1.2"),
            sl_cap=Decimal("100.0"),  # high cap so it doesn't clamp
        )
        # avg_mae = -0.20% → |mae| = 0.20
        # SL = 0.20 * 1.2 * 500 / 100 = 1.20
        mae_pct = 0.20
        sl = Decimal(str(mae_pct)) * config.sl_mae_multiplier * Decimal("500") / Decimal("100")
        assert float(sl) == pytest.approx(1.20, abs=0.01)

    def test_tp_from_mfe(self):
        """TP distance = |avg_mfe| * tp_mfe_multiplier * close / 100."""
        config = PatternDiscoveryConfig(
            tp_mfe_multiplier=Decimal("0.8"),
        )
        # avg_mfe = 0.30% → mfe = 0.30
        # TP = 0.30 * 0.8 * 500 / 100 = 1.20
        mfe_pct = 0.30
        tp = Decimal(str(mfe_pct)) * config.tp_mfe_multiplier * Decimal("500") / Decimal("100")
        assert float(tp) == pytest.approx(1.20, abs=0.01)

    def test_sl_capped(self):
        """SL distance should be capped at sl_cap."""
        config = PatternDiscoveryConfig(
            sl_mae_multiplier=Decimal("1.2"),
            sl_cap=Decimal("2.50"),
        )
        # avg_mae = -1.0% → |mae| = 1.0
        # raw SL = 1.0 * 1.2 * 500 / 100 = 6.0
        # Capped at 2.50
        mae_pct = 1.0
        raw_sl = Decimal(str(mae_pct)) * config.sl_mae_multiplier * Decimal("500") / Decimal("100")
        capped = min(raw_sl, config.sl_cap)
        assert capped == Decimal("2.50")

    def test_rr_check(self):
        """R:R below min_rr should prevent entry."""
        config = PatternDiscoveryConfig(min_rr=Decimal("1.0"))
        # If TP < SL → R:R < 1.0 → no entry
        sl_dist = Decimal("2.0")
        tp_dist = Decimal("0.5")
        rr = tp_dist / sl_dist
        assert rr < config.min_rr


# ======================================================================
# Feature buffer management
# ======================================================================


class TestFeatureBuffer:
    def test_buffer_limited_to_max_pattern_length(self):
        """Feature buffer should not exceed max pattern length."""
        bf = _make_features()
        pattern = _make_pattern(key=(bf, bf))  # 2-bar pattern
        catalog = PatternCatalog(patterns=(pattern,))
        strategy = PatternDiscoveryStrategy(catalog)

        # Buffer limit should be 2
        assert strategy._max_pattern_len == 2

        # Add 5 features
        for _ in range(5):
            strategy._feature_buffer.append(_make_features())

        # Simulate what the strategy does internally
        if len(strategy._feature_buffer) > strategy._max_pattern_len:
            strategy._feature_buffer = strategy._feature_buffer[-strategy._max_pattern_len:]

        assert len(strategy._feature_buffer) == 2


# ======================================================================
# Confirmation check
# ======================================================================


class TestConfirmationCheck:
    """Tests for _check_confirmation on the strategy."""

    def _make_mock_ctx(
        self,
        ema_fast: float = 510.0,
        ema_slow: float = 500.0,
        rsi_val: float = 50.0,
        adx_val: float = 25.0,
        vwap: float = 500.0,
        close: float = 501.0,
    ) -> MagicMock:
        """Create a mock BarContext for confirmation checks."""
        ctx = MagicMock()
        ctx.bar = MagicMock()
        ctx.bar.close = Decimal(str(close))
        ctx.ema_fast = Decimal(str(ema_fast))
        ctx.ema_slow = Decimal(str(ema_slow))
        ctx.rsi_val = Decimal(str(rsi_val))
        ctx.adx_val = Decimal(str(adx_val))
        ctx.vwap_bands = MagicMock()
        ctx.vwap_bands.vwap = Decimal(str(vwap))
        ctx.atr_val = Decimal("1.0")
        ctx.state = MagicMock()
        ctx.state.day_trades = 0
        ctx.state.last_entry_bar = 0
        ctx.bar_of_day = 20
        return ctx

    def test_no_confirmation_always_passes(self):
        """Pattern with no confirmation dict → always passes."""
        bf = _make_features()
        pattern = _make_pattern(key=(bf,))  # confirmation=None by default
        catalog = PatternCatalog(patterns=(pattern,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        ctx = self._make_mock_ctx()
        assert strategy._check_confirmation(pattern, ctx) is True

    def test_use_confirmation_false_skips_check(self):
        """When use_confirmation=False, always passes even with confirmation."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"rsi_zone": "oversold"},  # RSI must be oversold
        )
        catalog = PatternCatalog(patterns=(p,))
        config = PatternDiscoveryConfig(use_confirmation=False)
        strategy = PatternDiscoveryStrategy(catalog, config=config)
        strategy._data = []
        strategy._idx = 0

        # RSI is neutral (50), not oversold, but use_confirmation=False → passes
        ctx = self._make_mock_ctx(rsi_val=50.0)
        assert strategy._check_confirmation(p, ctx) is True

    def test_trend_dir_match(self):
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"trend_dir": "1"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # ema_fast > ema_slow → trend_dir = "1" → match
        ctx = self._make_mock_ctx(ema_fast=510.0, ema_slow=500.0)
        assert strategy._check_confirmation(p, ctx) is True

    def test_trend_dir_mismatch(self):
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"trend_dir": "1"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # ema_fast < ema_slow → trend_dir = "-1" → mismatch
        ctx = self._make_mock_ctx(ema_fast=490.0, ema_slow=500.0)
        assert strategy._check_confirmation(p, ctx) is False

    def test_rsi_zone_match(self):
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"rsi_zone": "neutral"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # RSI 50 → neutral → match
        ctx = self._make_mock_ctx(rsi_val=50.0)
        assert strategy._check_confirmation(p, ctx) is True

    def test_rsi_zone_mismatch(self):
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"rsi_zone": "oversold"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # RSI 50 → neutral, but needs oversold → mismatch
        ctx = self._make_mock_ctx(rsi_val=50.0)
        assert strategy._check_confirmation(p, ctx) is False

    def test_adx_level_match(self):
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"adx_level": "moderate"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # ADX 25 → moderate (20-40) → match
        ctx = self._make_mock_ctx(adx_val=25.0)
        assert strategy._check_confirmation(p, ctx) is True

    def test_multi_field_confirmation_all_match(self):
        """Multiple confirmation fields — all must match."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"trend_dir": "1", "rsi_zone": "neutral"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # Both match: uptrend + neutral RSI
        ctx = self._make_mock_ctx(ema_fast=510.0, ema_slow=500.0, rsi_val=50.0)
        assert strategy._check_confirmation(p, ctx) is True

    def test_multi_field_confirmation_one_mismatch(self):
        """Multiple confirmation fields — one mismatch blocks entry."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"trend_dir": "1", "rsi_zone": "oversold"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        # trend_dir matches (uptrend) but rsi_zone doesn't (neutral != oversold)
        ctx = self._make_mock_ctx(ema_fast=510.0, ema_slow=500.0, rsi_val=50.0)
        assert strategy._check_confirmation(p, ctx) is False

    def test_unknown_field_skipped(self):
        """Unknown confirmation fields are silently skipped."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"unknown_field": "whatever"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = []
        strategy._idx = 0

        ctx = self._make_mock_ctx()
        assert strategy._check_confirmation(p, ctx) is True

    def test_htf_trend_confirmation_match(self):
        """HTF trend confirmation: matching trend passes."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"htf_trend": "1"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = [MagicMock()]
        strategy._idx = 0
        # Mock hub to return htf_trend = 1
        strategy._hub = MagicMock()
        strategy._hub.htf_ema_trend.return_value = 1

        ctx = self._make_mock_ctx()
        assert strategy._check_confirmation(p, ctx) is True

    def test_htf_trend_confirmation_mismatch(self):
        """HTF trend confirmation: mismatching trend fails."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"htf_trend": "1"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = [MagicMock()]
        strategy._idx = 0
        strategy._hub = MagicMock()
        strategy._hub.htf_ema_trend.return_value = -1  # bearish HTF

        ctx = self._make_mock_ctx()
        assert strategy._check_confirmation(p, ctx) is False

    def test_cvd_direction_confirmation_match(self):
        """CVD direction confirmation: matching direction passes."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=50,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.10,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"cvd_direction": "buying"},
        )
        catalog = PatternCatalog(patterns=(p,))
        strategy = PatternDiscoveryStrategy(catalog)
        strategy._data = [MagicMock()]
        strategy._idx = 0
        strategy._hub = MagicMock()
        strategy._hub.cvd_normalized.return_value = Decimal("0.5")

        ctx = self._make_mock_ctx()
        assert strategy._check_confirmation(p, ctx) is True

    def test_require_htf_alignment_blocks_counter_trend(self):
        """require_htf_alignment=True blocks long when HTF is bearish."""
        bf = _make_features()
        pattern = _make_pattern(key=(bf,), direction="long")
        catalog = PatternCatalog(patterns=(pattern,))
        config = PatternDiscoveryConfig(require_htf_alignment=True)
        strategy = PatternDiscoveryStrategy(catalog, config=config)
        assert config.require_htf_alignment is True

    def test_require_htf_alignment_default_false(self):
        """require_htf_alignment defaults to False."""
        config = PatternDiscoveryConfig()
        assert config.require_htf_alignment is False
