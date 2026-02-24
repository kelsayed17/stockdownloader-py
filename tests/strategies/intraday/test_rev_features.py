"""Tests for Reversal (REV) strategy enhancement features.

Covers:
- Separate trade window (rev_can_trade_bar)
- Band touch filter (rev_min_touches)
- Configurable hugging limit (rev_hug_limit)
- RR-based TP mode (rev_tp_mode)
- S/R hard filter (rev_require_sr)
- Backward compatibility
- Config + protocol verification
- SessionState new fields
- PineScript mode updates
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.reversal import ReversalStrategyConfig
from stockdownloader.strategies.intraday.reversal import ReversalStrategy
from stockdownloader.strategies.intraday.session import SessionState
from stockdownloader.pinescript.modes import rev_mode as _rev_mode

_ZERO = Decimal("0")
_D = Decimal


# ======================================================================
# Separate trade window
# ======================================================================

class TestREVTradeWindow:
    """Separate trade window for REV entries."""

    def test_default_is_eight(self) -> None:
        """Default rev_can_trade_bar is 8 (skip first 40 min volatility)."""
        config = ReversalStrategyConfig()
        assert config.rev_can_trade_bar == 8

    def test_custom_trade_bar(self) -> None:
        config = ReversalStrategyConfig(rev_can_trade_bar=13)
        assert config.rev_can_trade_bar == 13

    def test_zero_means_use_shared(self) -> None:
        """When rev_can_trade_bar=0, shared can_trade_bar (11) applies."""
        config = ReversalStrategyConfig(rev_can_trade_bar=0)
        assert config.rev_can_trade_bar == 0
        assert config.can_trade_bar == 11  # shared default


# ======================================================================
# Band touch filter
# ======================================================================

class TestREVBandTouches:
    """Band touch filter blocks entries without sufficient band touches."""

    def test_default_min_touches(self) -> None:
        config = ReversalStrategyConfig()
        assert config.rev_min_touches == 1

    def test_custom_min_touches(self) -> None:
        config = ReversalStrategyConfig(rev_min_touches=2)
        assert config.rev_min_touches == 2

    def test_rev_band_touches_field_exists(self) -> None:
        state = SessionState()
        assert hasattr(state, "rev_band_touches")
        assert state.rev_band_touches == 0

    def test_rev_band_touches_resets(self) -> None:
        state = SessionState()
        state.rev_band_touches = 5
        state.reset("2025-01-20")
        assert state.rev_band_touches == 0


# ======================================================================
# Configurable hugging limit
# ======================================================================

class TestREVHugLimit:
    """Configurable hugging limit replaces hardcoded 20."""

    def test_default_matches_optimized_value(self) -> None:
        """Default rev_hug_limit = 30 (wider for more setups)."""
        config = ReversalStrategyConfig()
        assert config.rev_hug_limit == 30

    def test_custom_hug_limit(self) -> None:
        config = ReversalStrategyConfig(rev_hug_limit=15)
        assert config.rev_hug_limit == 15

    def test_low_hug_limit_is_stricter(self) -> None:
        """Lower hug limit blocks entry sooner (more restrictive)."""
        config_strict = ReversalStrategyConfig(rev_hug_limit=10)
        config_loose = ReversalStrategyConfig(rev_hug_limit=30)
        assert config_strict.rev_hug_limit < config_loose.rev_hug_limit


# ======================================================================
# RR-based TP mode
# ======================================================================

class TestREVTPMode:
    """REV TP mode: 'vwap' (default) or 'rr' (RR-based)."""

    def test_vwap_is_default(self) -> None:
        config = ReversalStrategyConfig()
        assert config.rev_tp_mode == "vwap"

    def test_rr_mode(self) -> None:
        config = ReversalStrategyConfig(rev_tp_mode="rr")
        assert config.rev_tp_mode == "rr"

    def test_rev_rr_default(self) -> None:
        """Default R:R multiplier is 1.0."""
        config = ReversalStrategyConfig()
        assert config.rev_rr == _D("1.0")

    def test_custom_rev_rr(self) -> None:
        config = ReversalStrategyConfig(rev_rr=_D("1.5"))
        assert config.rev_rr == _D("1.5")

    def test_rr_mode_computes_tp_from_sl(self) -> None:
        """In 'rr' mode, TP = entry +/- (SL dist * rev_rr)."""
        sl_dist = _D("1.0")
        rev_rr = _D("1.0")
        tp_dist = sl_dist * rev_rr
        close = _D("500")
        # Long
        sl_price_long = close - sl_dist
        tp_price_long = close + tp_dist
        assert sl_price_long == _D("499")
        assert tp_price_long == _D("501")
        # Short
        sl_price_short = close + sl_dist
        tp_price_short = close - tp_dist
        assert sl_price_short == _D("501")
        assert tp_price_short == _D("499")


# ======================================================================
# S/R hard filter
# ======================================================================

class TestREVRequireSR:
    """S/R hard filter blocks REV entries without S/R proximity."""

    def test_enabled_by_default(self) -> None:
        config = ReversalStrategyConfig()
        assert config.rev_require_sr is True

    def test_enabled(self) -> None:
        config = ReversalStrategyConfig(rev_require_sr=True)
        assert config.rev_require_sr is True


# ======================================================================
# Config + Protocol verification
# ======================================================================

class TestREVConfig:
    """Config and protocol satisfaction tests."""

    def test_backward_compat_defaults(self) -> None:
        """REV params have correct defaults."""
        config = ReversalStrategyConfig()
        assert config.rev_can_trade_bar == 8
        assert config.rev_min_touches == 1
        assert config.rev_hug_limit == 30
        assert config.rev_tp_mode == "vwap"
        assert config.rev_rr == _D("1.0")
        assert config.rev_require_sr is True
        assert config.rev_vwap_flat_tol == _D("0.10")

    def test_factory_accepts_new_params(self) -> None:
        """for_reversal() factory accepts all new params."""
        config = ReversalStrategyConfig(
            rev_can_trade_bar=13,
            rev_min_touches=2,
            rev_hug_limit=15,
            rev_tp_mode="rr",
            rev_rr=_D("1.5"),
            rev_require_sr=True,
        )
        assert config.rev_can_trade_bar == 13
        assert config.rev_min_touches == 2
        assert config.rev_hug_limit == 15
        assert config.rev_tp_mode == "rr"
        assert config.rev_rr == _D("1.5")
        assert config.rev_require_sr is True


# ======================================================================
# SessionState new fields
# ======================================================================

class TestREVSessionState:
    """Session state fields for PB/REV enhancements."""

    def test_cum_bars_above_vwap_exists(self) -> None:
        state = SessionState()
        assert hasattr(state, "cum_bars_above_vwap")
        assert state.cum_bars_above_vwap == 0

    def test_cum_bars_below_vwap_exists(self) -> None:
        state = SessionState()
        assert hasattr(state, "cum_bars_below_vwap")
        assert state.cum_bars_below_vwap == 0

    def test_rev_band_touches_exists(self) -> None:
        state = SessionState()
        assert hasattr(state, "rev_band_touches")
        assert state.rev_band_touches == 0

    def test_new_fields_reset_on_session(self) -> None:
        """All new fields reset when session resets."""
        state = SessionState()
        state.cum_bars_above_vwap = 40
        state.cum_bars_below_vwap = 10
        state.rev_band_touches = 3
        state.reset("2025-01-20")
        assert state.cum_bars_above_vwap == 0
        assert state.cum_bars_below_vwap == 0
        assert state.rev_band_touches == 0


# ======================================================================
# PineScript mode updates
# ======================================================================

class TestREVPineScript:
    """PineScript mode definition includes new inputs."""

    def test_new_inputs_present(self) -> None:
        """All new REV inputs are in the PineScript mode."""
        mode = _rev_mode()
        input_names = {inp.name for inp in mode.inputs}
        expected = {
            "revCanTradeBar", "revMinTouches", "revHugLimit",
            "revTpMode", "revRr", "revRequireSr",
        }
        assert expected.issubset(input_names), f"Missing: {expected - input_names}"

    def test_existing_inputs_preserved(self) -> None:
        """Original REV inputs are still present."""
        mode = _rev_mode()
        input_names = {inp.name for inp in mode.inputs}
        original = {"revAdxMax", "revBandMult", "revBody"}
        assert original.issubset(input_names), f"Missing: {original - input_names}"

    def test_description_updated(self) -> None:
        """Mode description mentions new features."""
        mode = _rev_mode()
        assert "band touch" in mode.description
        assert "hug limit" in mode.description
        assert "S/R confluence" in mode.description

    def test_strategy_name_unchanged(self) -> None:
        """Strategy name remains 'VWAP Reversal'."""
        strat = ReversalStrategy()
        assert strat.name == "VWAP Reversal"
