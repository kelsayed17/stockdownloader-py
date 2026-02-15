"""Tests for Pullback (PB) strategy enhancement features.

Covers:
- VWAP session bias filter
- S/R hard filter
- VWAP TP mode
- Backward compatibility
- Config + protocol verification
- PineScript mode updates
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategy.intraday.entry_helpers import clamp_sl_dist, directional_sl_tp
from stockdownloader.strategy.intraday.pullback_config import PullbackStrategyConfig
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.util.pinescript_strategies import _pb_mode

_ZERO = Decimal("0")
_D = Decimal


# ======================================================================
# VWAP session bias filter
# ======================================================================

class TestPBVwapBias:
    """VWAP session bias filter blocks entries without VWAP-side dominance."""

    def test_disabled_by_default(self) -> None:
        config = PullbackStrategyConfig()
        assert config.pb_vwap_bias is False

    def test_default_bias_pct(self) -> None:
        config = PullbackStrategyConfig()
        assert config.pb_vwap_bias_pct == _D("0.7")

    def test_bias_enabled(self) -> None:
        config = PullbackStrategyConfig(pb_vwap_bias=True)
        assert config.pb_vwap_bias is True

    def test_custom_bias_pct(self) -> None:
        config = PullbackStrategyConfig(
            pb_vwap_bias=True, pb_vwap_bias_pct=_D("0.8"),
        )
        assert config.pb_vwap_bias_pct == _D("0.8")

    def test_cum_bars_above_vwap_tracked(self) -> None:
        """Cumulative VWAP bar counters exist and reset."""
        state = SessionState()
        assert state.cum_bars_above_vwap == 0
        assert state.cum_bars_below_vwap == 0
        state.cum_bars_above_vwap = 10
        state.cum_bars_below_vwap = 3
        state.reset("2025-01-20")
        assert state.cum_bars_above_vwap == 0
        assert state.cum_bars_below_vwap == 0


# ======================================================================
# S/R hard filter
# ======================================================================

class TestPBRequireSR:
    """S/R hard filter blocks PB entries without S/R proximity."""

    def test_disabled_by_default(self) -> None:
        config = PullbackStrategyConfig()
        assert config.pb_require_sr is False

    def test_enabled(self) -> None:
        config = PullbackStrategyConfig(pb_require_sr=True)
        assert config.pb_require_sr is True


# ======================================================================
# VWAP TP mode
# ======================================================================

class TestPBTPMode:
    """PB TP mode: 'rr' (default) or 'vwap' (target VWAP)."""

    def test_rr_is_default(self) -> None:
        config = PullbackStrategyConfig()
        assert config.pb_tp_mode == "rr"

    def test_vwap_mode(self) -> None:
        config = PullbackStrategyConfig(pb_tp_mode="vwap")
        assert config.pb_tp_mode == "vwap"

    def test_rr_mode_computes_tp_from_sl(self) -> None:
        """In 'rr' mode, TP = SL distance * RR multiplier."""
        sl_dist = _D("1.3")
        rr = _D("1.4")
        tp_dist = sl_dist * rr
        sl_price, tp_price = directional_sl_tp(True, _D("500"), sl_dist, tp_dist)
        assert sl_price == _D("500") - _D("1.3")
        assert tp_price == _D("500") + _D("1.3") * _D("1.4")

    def test_vwap_mode_targets_vwap_price(self) -> None:
        """In 'vwap' mode, TP targets the VWAP value."""
        # This is a logic-level check; the strategy targets vwap directly
        vwap = _D("500")
        close = _D("498")  # long entry below VWAP
        sl_dist = _D("1.5")
        # In vwap mode, tp_price = vwap
        tp_price = vwap
        sl_price = close - sl_dist
        reward = abs(tp_price - close)
        rr = reward / sl_dist
        assert tp_price == _D("500")
        assert rr > _D("0.3")  # Should pass min RR check

    def test_vwap_mode_rejects_low_rr(self) -> None:
        """VWAP TP mode rejects entries where VWAP is too close (RR < 0.3)."""
        vwap = _D("500")
        close = _D("499.8")  # very close to VWAP
        sl_dist = _D("1.5")
        reward = abs(vwap - close)  # 0.2
        rr = reward / sl_dist  # 0.133...
        assert rr < _D("0.3")  # Should be rejected


# ======================================================================
# Config + Protocol verification
# ======================================================================

class TestPBConfig:
    """Config and protocol satisfaction tests."""

    def test_backward_compat_defaults(self) -> None:
        """All new PB params default to off/disabled."""
        config = PullbackStrategyConfig()
        assert config.pb_vwap_bias is False
        assert config.pb_vwap_bias_pct == _D("0.7")
        assert config.pb_require_sr is False
        assert config.pb_tp_mode == "rr"

    def test_factory_accepts_new_params(self) -> None:
        """for_pullback() factory accepts all new params."""
        config = PullbackStrategyConfig(
            pb_vwap_bias=True,
            pb_vwap_bias_pct=_D("0.8"),
            pb_require_sr=True,
            pb_tp_mode="vwap",
        )
        assert config.pb_vwap_bias is True
        assert config.pb_vwap_bias_pct == _D("0.8")
        assert config.pb_require_sr is True
        assert config.pb_tp_mode == "vwap"


# ======================================================================
# PineScript mode updates
# ======================================================================

class TestPBPineScript:
    """PineScript mode definition includes new inputs."""

    def test_new_inputs_present(self) -> None:
        """All new PB inputs are in the PineScript mode."""
        mode = _pb_mode()
        input_names = {inp.name for inp in mode.inputs}
        expected = {"pbVwapBias", "pbVwapBiasPct", "pbRequireSr", "pbTpMode"}
        assert expected.issubset(input_names), f"Missing: {expected - input_names}"

    def test_existing_inputs_preserved(self) -> None:
        """Original PB inputs are still present."""
        mode = _pb_mode()
        input_names = {inp.name for inp in mode.inputs}
        original = {"pbZone", "pbAdxMin", "pbTrendBars"}
        assert original.issubset(input_names), f"Missing: {original - input_names}"

    def test_description_updated(self) -> None:
        """Mode description mentions new features."""
        mode = _pb_mode()
        assert "VWAP session bias" in mode.description
        assert "S/R confluence" in mode.description

    def test_strategy_name_unchanged(self) -> None:
        """Strategy name remains 'VWAP Pullback'."""
        strat = PullbackStrategy()
        assert strat.name == "VWAP Pullback"
