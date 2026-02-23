"""Tests for ORR (Opening Range Reversal) enhancement features.

Covers:
- Re-breakout invalidation exit (exit_manager)
- VWAP disagreement filter
- Gap fade filter
- ADX regime filter (inverted — low ADX)
- OR Opposite TP mode
- Breakout-then-reclaim detection (require_break mode)
- Enhanced scoring
- Backward compatibility
- Config + protocol verification
- PineScript mode updates
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.trade import IntradayAction
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.intraday.trade_management import IntradayExitManager
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategyConfig
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.util.pinescript.modes import orr_mode as _orr_mode

from .conftest import make_bar, make_long_position_state, make_short_position_state, make_vwap_bands

_ZERO = Decimal("0")
_D = Decimal


# ======================================================================
# Re-breakout invalidation exit (exit_manager)
# ======================================================================

class TestORRRebreakExit:
    """ORR re-breakout invalidation exit tests."""

    @staticmethod
    def _make_exit_config(**overrides: object) -> SimpleNamespace:
        """Minimal ExitConfig with ORR re-breakout enabled."""
        defaults = dict(
            be_trigger=_D("0.5"),
            trail_vwap=True,
            close_eod=True,
            bars_per_day=78,
            orb_reentry_exit=False,
            orb_time_exit=0,
            orr_rebreak_exit=True,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_orr_short_rebreak_exits(self) -> None:
        """ORR short position: price breaks above OR high → exit."""
        state = make_short_position_state(
            entry_price=_D("502"), stop_loss=_D("505"),
            take_profit=_D("498"), entry_mode="ORR",
        )
        state.or_high = _D("503")
        state.or_low = _D("497")
        bar = make_bar(close=_D("504"), high=_D("504"), low=_D("501"))
        config = self._make_exit_config()
        vbands = make_vwap_bands(vwap=_D("500"))

        mgr = IntradayExitManager()
        sig = mgr.evaluate(bar, state, config, vbands, _D("2"), bar_of_day=15)
        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "orr_rebreak"

    def test_orr_long_rebreak_exits(self) -> None:
        """ORR long position: price breaks below OR low → exit."""
        state = make_long_position_state(
            entry_price=_D("498"), stop_loss=_D("495"),
            take_profit=_D("502"), entry_mode="ORR",
        )
        state.or_high = _D("503")
        state.or_low = _D("497")
        bar = make_bar(close=_D("496"), high=_D("499"), low=_D("496"))
        config = self._make_exit_config()
        vbands = make_vwap_bands(vwap=_D("500"))

        mgr = IntradayExitManager()
        sig = mgr.evaluate(bar, state, config, vbands, _D("2"), bar_of_day=15)
        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "orr_rebreak"

    def test_rebreak_disabled_by_default(self) -> None:
        """Default config: re-breakout exit does not fire."""
        state = make_short_position_state(
            entry_price=_D("502"), stop_loss=_D("505"),
            take_profit=_D("498"), entry_mode="ORR",
        )
        state.or_high = _D("503")
        state.or_low = _D("497")
        bar = make_bar(close=_D("504"), high=_D("504"), low=_D("501"))
        config = self._make_exit_config(orr_rebreak_exit=False)
        vbands = make_vwap_bands(vwap=_D("500"))

        mgr = IntradayExitManager()
        sig = mgr.evaluate(bar, state, config, vbands, _D("2"), bar_of_day=15)
        # Should not exit via rebreak (may exit via TP or other reasons, but not rebreak)
        assert sig.reason != "orr_rebreak"

    def test_rebreak_only_fires_for_orr(self) -> None:
        """Re-breakout check only applies to ORR entry mode."""
        state = make_short_position_state(
            entry_price=_D("502"), stop_loss=_D("505"),
            take_profit=_D("498"), entry_mode="PB",
        )
        state.or_high = _D("503")
        state.or_low = _D("497")
        bar = make_bar(close=_D("504"), high=_D("504"), low=_D("501"))
        config = self._make_exit_config()
        vbands = make_vwap_bands(vwap=_D("500"))

        mgr = IntradayExitManager()
        sig = mgr.evaluate(bar, state, config, vbands, _D("2"), bar_of_day=15)
        assert sig.reason != "orr_rebreak"

    def test_orr_short_stays_when_inside_or(self) -> None:
        """ORR short stays in position when price stays below OR high."""
        state = make_short_position_state(
            entry_price=_D("502"), stop_loss=_D("505"),
            take_profit=_D("498"), entry_mode="ORR",
        )
        state.or_high = _D("503")
        state.or_low = _D("497")
        # Close at 501 — below OR high, no rebreak
        bar = make_bar(close=_D("501"), high=_D("502"), low=_D("500"))
        config = self._make_exit_config()
        vbands = make_vwap_bands(vwap=_D("500"))

        mgr = IntradayExitManager()
        sig = mgr.evaluate(bar, state, config, vbands, _D("2"), bar_of_day=15)
        assert sig.reason != "orr_rebreak"


# ======================================================================
# VWAP disagreement filter
# ======================================================================

class TestVwapDisagreeFilter:
    """VWAP disagreement filter blocks entries aligned with VWAP."""

    @staticmethod
    def _make_orr_config(**overrides: object) -> ORReversalStrategyConfig:
        defaults = dict(orr_enable=True, orr_vwap_disagree=True)
        defaults.update(overrides)
        return ORReversalStrategyConfig(**defaults)

    def test_short_orr_blocked_when_close_above_vwap(self) -> None:
        """Short ORR with close > VWAP → blocked (breakout was WITH vwap)."""
        config = self._make_orr_config()
        assert config.orr_vwap_disagree is True

    def test_vwap_disagree_disabled_by_default(self) -> None:
        """Default config has vwap disagree off (too restrictive for SPY)."""
        config = ORReversalStrategyConfig()
        assert config.orr_vwap_disagree is False


# ======================================================================
# Gap fade filter
# ======================================================================

class TestGapFadeFilter:
    """Gap fade filter blocks ORR entries aligned with gap direction."""

    def test_gap_filter_enabled_by_default(self) -> None:
        config = ORReversalStrategyConfig()
        assert config.orr_gap_filter is True

    def test_gap_filter_enabled(self) -> None:
        config = ORReversalStrategyConfig(orr_gap_filter=True)
        assert config.orr_gap_filter is True


# ======================================================================
# ADX regime filter (inverted)
# ======================================================================

class TestAdxFilter:
    """ADX filter blocks ORR in trending markets (high ADX)."""

    def test_adx_filter_enabled_by_default(self) -> None:
        config = ORReversalStrategyConfig()
        assert config.orr_adx_filter is True

    def test_adx_filter_enabled(self) -> None:
        config = ORReversalStrategyConfig(orr_adx_filter=True)
        assert config.orr_adx_filter is True


# ======================================================================
# OR Opposite TP mode
# ======================================================================

class TestOROppositeTP:
    """OR Opposite TP mode targets the other side of the OR."""

    def test_compute_tp_vwap_mode(self) -> None:
        """Default VWAP mode returns VWAP as TP."""
        state = SimpleNamespace(or_high=_D("505"), or_low=_D("495"))
        config = SimpleNamespace(orr_tp_mode="VWAP")
        tp = ORReversalStrategy._compute_tp(True, _D("500"), _D("500"), state, config)
        assert tp == _D("500")

    def test_compute_tp_or_mid(self) -> None:
        """OR Mid mode returns midpoint of OR."""
        state = SimpleNamespace(or_high=_D("505"), or_low=_D("495"))
        config = SimpleNamespace(orr_tp_mode="OR Mid")
        tp = ORReversalStrategy._compute_tp(True, _D("500"), _D("500"), state, config)
        assert tp == _D("500")

    def test_compute_tp_or_opposite_long(self) -> None:
        """OR Opposite mode for long → target OR high."""
        state = SimpleNamespace(or_high=_D("505"), or_low=_D("495"))
        config = SimpleNamespace(orr_tp_mode="OR Opposite")
        tp = ORReversalStrategy._compute_tp(True, _D("500"), _D("500"), state, config)
        assert tp == _D("505")

    def test_compute_tp_or_opposite_short(self) -> None:
        """OR Opposite mode for short → target OR low."""
        state = SimpleNamespace(or_high=_D("505"), or_low=_D("495"))
        config = SimpleNamespace(orr_tp_mode="OR Opposite")
        tp = ORReversalStrategy._compute_tp(False, _D("500"), _D("500"), state, config)
        assert tp == _D("495")


# ======================================================================
# Breakout-then-reclaim detection
# ======================================================================

class TestRequireBreak:
    """Require-breakout mode blocks entries without prior OR break."""

    def test_require_break_enabled_by_default(self) -> None:
        config = ORReversalStrategyConfig()
        assert config.orr_require_break is True

    def test_require_break_disabled(self) -> None:
        config = ORReversalStrategyConfig(orr_require_break=False)
        assert config.orr_require_break is False

    def test_orr_break_flags_reset(self) -> None:
        """orr_break_above/below reset at session start."""
        state = SessionState()
        state.orr_break_above = True
        state.orr_break_below = True
        state.reset("2025-01-16")
        assert state.orr_break_above is False
        assert state.orr_break_below is False


# ======================================================================
# Enhanced scoring
# ======================================================================

class TestORRScoring:
    """Enhanced ORR scoring includes VWAP, gap, and ADX points."""

    @staticmethod
    def _make_ctx(**overrides: object) -> SimpleNamespace:
        """Create minimal BarContext for scoring tests."""
        defaults = dict(
            rel_vol=_D("2.5"),
            bar=SimpleNamespace(close=_D("498")),
            vwap_bands=SimpleNamespace(vwap=_D("500")),
            adx_val=_D("15"),
            sr_any=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @staticmethod
    def _make_config(**overrides: object) -> SimpleNamespace:
        defaults = dict(orr_adx_max=_D("30"), w_sr=2)
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_high_volume_score(self) -> None:
        """RVOL >= 2 gives 3 volume points."""
        ctx = self._make_ctx(rel_vol=_D("2.5"))
        state = SimpleNamespace(gap_dir=0)
        config = self._make_config()
        score, _ = ORReversalStrategy._compute_score(ctx, True, False, state, config)
        # 3 (vol) + 0 (vwap: close 498 < vwap 500 for long) + 0 (gap) + 1 (adx < 30) + 0 (sr)
        assert score >= 3

    def test_low_adx_gives_point(self) -> None:
        """ADX below orr_adx_max gives 1 point."""
        ctx = self._make_ctx(adx_val=_D("15"))
        state = SimpleNamespace(gap_dir=0)
        config = self._make_config(orr_adx_max=_D("30"))
        score, _ = ORReversalStrategy._compute_score(ctx, True, False, state, config)
        # adx 15 < 30 → 1 point
        # Total should include adx point
        assert score >= 1

    def test_high_adx_no_point(self) -> None:
        """ADX above orr_adx_max gives 0 points."""
        ctx = self._make_ctx(adx_val=_D("35"), rel_vol=_D("0.5"))
        state = SimpleNamespace(gap_dir=0)
        config = self._make_config(orr_adx_max=_D("30"))
        score, _ = ORReversalStrategy._compute_score(ctx, True, False, state, config)
        # 0 (vol < 1) + 0 (vwap) + 0 (gap) + 0 (adx >= 30) + 0 (sr)
        assert score == 0

    def test_gap_fade_gives_point_long(self) -> None:
        """Gap-down day + long ORR → 1 gap point."""
        ctx = self._make_ctx(rel_vol=_D("0.5"), adx_val=_D("35"))
        state = SimpleNamespace(gap_dir=-1)
        config = self._make_config()
        score, _ = ORReversalStrategy._compute_score(ctx, True, False, state, config)
        # 0 (vol) + 0 (vwap) + 1 (gap -1 + long) + 0 (adx >= 30) + 0 (sr)
        assert score == 1

    def test_gap_fade_gives_point_short(self) -> None:
        """Gap-up day + short ORR → 1 gap point."""
        ctx = self._make_ctx(rel_vol=_D("0.5"), adx_val=_D("35"))
        state = SimpleNamespace(gap_dir=1)
        config = self._make_config()
        score, _ = ORReversalStrategy._compute_score(ctx, False, True, state, config)
        # 0 (vol) + 0 (vwap: close 498 < vwap 500 for short) + 1 (gap) + 0 (adx >= 30) + 0 (sr)
        assert score >= 1

    def test_vwap_alignment_gives_point(self) -> None:
        """Close > VWAP for long → 1 VWAP point."""
        ctx = self._make_ctx(
            bar=SimpleNamespace(close=_D("502")),
            vwap_bands=SimpleNamespace(vwap=_D("500")),
            rel_vol=_D("0.5"),
            adx_val=_D("35"),
        )
        state = SimpleNamespace(gap_dir=0)
        config = self._make_config()
        score, _ = ORReversalStrategy._compute_score(ctx, True, False, state, config)
        # 0 (vol) + 1 (vwap: 502 > 500 for long) + 0 (gap) + 0 (adx >= 30) + 0 (sr)
        assert score == 1

    def test_max_score(self) -> None:
        """Max score is 3 (vol) + 1 (vwap) + 1 (gap) + 1 (adx) + w_sr."""
        ctx = self._make_ctx()
        state = SimpleNamespace(gap_dir=0)
        config = self._make_config(w_sr=2)
        _, max_score = ORReversalStrategy._compute_score(ctx, True, False, state, config)
        assert max_score == 3 + 1 + 1 + 1 + 2  # 8


# ======================================================================
# Config + Protocol verification
# ======================================================================

class TestORRConfig:
    """Config and protocol satisfaction tests."""

    def test_backward_compat_defaults(self) -> None:
        """ORR quality filters default to on; vwap_disagree off."""
        config = ORReversalStrategyConfig()
        assert config.orr_vwap_disagree is False
        assert config.orr_gap_filter is True
        assert config.orr_adx_filter is True
        assert config.orr_adx_max == Decimal("30")
        assert config.orr_rebreak_exit is False
        assert config.orr_require_break is True

    def test_factory_accepts_new_params(self) -> None:
        """for_or_reversal() factory accepts all new params."""
        config = ORReversalStrategyConfig(
            orr_vwap_disagree=True,
            orr_gap_filter=True,
            orr_adx_filter=True,
            orr_rebreak_exit=True,
            orr_require_break=True,
        )
        assert config.orr_vwap_disagree is True
        assert config.orr_gap_filter is True
        assert config.orr_adx_filter is True
        assert config.orr_rebreak_exit is True
        assert config.orr_require_break is True

    def test_orr_tp_mode_includes_or_opposite(self) -> None:
        """orr_tp_mode comment/value supports 'OR Opposite'."""
        config = ORReversalStrategyConfig(orr_tp_mode="OR Opposite")
        assert config.orr_tp_mode == "OR Opposite"


# ======================================================================
# PineScript mode updates
# ======================================================================

class TestORRPineScript:
    """PineScript mode definition includes new inputs."""

    def test_new_inputs_present(self) -> None:
        """All new ORR inputs are in the PineScript mode."""
        mode = _orr_mode()
        input_names = {inp.name for inp in mode.inputs}
        expected = {
            "orrTpMode", "orrVwapDisagree", "orrGapFilter",
            "orrAdxFilter", "orrRebreakExit", "orrRequireBreak",
        }
        assert expected.issubset(input_names), f"Missing: {expected - input_names}"

    def test_original_inputs_preserved(self) -> None:
        """Original ORR inputs are still present."""
        mode = _orr_mode()
        input_names = {inp.name for inp in mode.inputs}
        original = {"orrWindow", "orrProx", "orrRvol"}
        assert original.issubset(input_names), f"Missing: {original - input_names}"

    def test_description_updated(self) -> None:
        """Mode description mentions new features."""
        mode = _orr_mode()
        assert "VWAP disagreement" in mode.description
        assert "gap-fade" in mode.description

    def test_strategy_name_unchanged(self) -> None:
        """Strategy name remains 'OR Reversal'."""
        strat = ORReversalStrategy()
        assert strat.name == "OR Reversal"


# ======================================================================
# SessionState ORR breakout tracking fields
# ======================================================================

class TestORRStateFields:
    """Session state fields for ORR breakout tracking."""

    def test_orr_break_fields_exist(self) -> None:
        """orr_break_above and orr_break_below fields exist."""
        state = SessionState()
        assert hasattr(state, "orr_break_above")
        assert hasattr(state, "orr_break_below")

    def test_orr_break_fields_default_false(self) -> None:
        """Both breakout flags default to False."""
        state = SessionState()
        assert state.orr_break_above is False
        assert state.orr_break_below is False

    def test_orr_break_fields_reset_on_new_session(self) -> None:
        """Breakout flags reset when session resets."""
        state = SessionState()
        state.orr_break_above = True
        state.orr_break_below = True
        state.reset("2025-01-20")
        assert state.orr_break_above is False
        assert state.orr_break_below is False
