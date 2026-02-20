"""Tests for standalone strategy config dataclasses.

Verifies that each strategy-specific config:
- Inherits from InfraExitConfig
- Is frozen (immutable)
- Has correct default values
- Supports ``dataclasses.replace()`` for overrides
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from stockdownloader.strategy.intraday.avwap_pullback_config import (
    AVWAPPullbackConfig,
)
from stockdownloader.strategy.intraday.base_config import InfraExitConfig
from stockdownloader.strategy.intraday.or_breakout_config import (
    ORBreakoutStrategyConfig,
)
from stockdownloader.strategy.intraday.or_reversal_config import (
    ORReversalStrategyConfig,
)
from stockdownloader.strategy.intraday.pattern_scalp_config import (
    PatternScalpStrategyConfig,
)
from stockdownloader.strategy.intraday.pullback_config import PullbackStrategyConfig
from stockdownloader.strategy.intraday.reversal_config import ReversalStrategyConfig
from stockdownloader.strategy.intraday.smc_structure_config import SMCStructureConfig

_D = Decimal


# ======================================================================
# InfraExitConfig base class
# ======================================================================


class TestInfraExitConfig:
    """Base config shared by all strategies."""

    def test_frozen(self) -> None:
        cfg = InfraExitConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.adx_len = 99  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = InfraExitConfig()
        assert cfg.adx_len == 14
        assert cfg.ema_fast == 9
        assert cfg.ema_slow == 21
        assert cfg.bars_per_day == 78
        assert cfg.can_trade_bar == 11
        assert cfg.be_trigger == _D("0.7")
        assert cfg.trail_buf == _D("0.15")
        assert cfg.orb_trail_atr == _D("0.8")


# ======================================================================
# PullbackStrategyConfig
# ======================================================================


class TestPullbackStrategyConfig:
    """Pullback strategy config tests."""

    def test_frozen(self) -> None:
        cfg = PullbackStrategyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.pb_zone = _D("99")  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(PullbackStrategyConfig, InfraExitConfig)

    def test_pb_defaults(self) -> None:
        cfg = PullbackStrategyConfig()
        assert cfg.pb_zone == _D("0.5")
        assert cfg.pb_body == _D("0.20")
        assert cfg.rr == _D("1.8")
        assert cfg.sl_atr == _D("1.3")
        assert cfg.sl_cap == _D("2.00")
        assert cfg.trend_bars == 7
        assert cfg.adx_thresh == _D("22")
        assert cfg.max_day == 1
        assert cfg.spacing == 5
        assert cfg.pb_tp_mode == "rr"
        assert cfg.pb_vwap_bias is True
        assert cfg.pb_require_sr is False

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(PullbackStrategyConfig(), rr=_D("2.0"))
        assert cfg.rr == _D("2.0")
        assert cfg.pb_zone == _D("0.5")  # unchanged

    def test_inherits_infra_defaults(self) -> None:
        cfg = PullbackStrategyConfig()
        assert cfg.bars_per_day == 78
        assert cfg.circuit == 3
        assert cfg.be_trigger == _D("0.7")  # PB uses base default


# ======================================================================
# ReversalStrategyConfig
# ======================================================================


class TestReversalStrategyConfig:
    """Reversal strategy config tests."""

    def test_frozen(self) -> None:
        cfg = ReversalStrategyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.rev_body = _D("99")  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(ReversalStrategyConfig, InfraExitConfig)

    def test_rev_defaults(self) -> None:
        cfg = ReversalStrategyConfig()
        assert cfg.rev_enable is True
        assert cfg.rev_band == "1σ"
        assert cfg.rev_body == _D("0.15")
        assert cfg.rev_sl_atr == _D("1.0")
        assert cfg.rev_shorts is False
        assert cfg.rev_min_rr == _D("1.0")
        assert cfg.rev_can_trade_bar == 8
        assert cfg.rev_min_touches == 1
        assert cfg.rev_hug_limit == 30
        assert cfg.rev_tp_mode == "vwap"
        assert cfg.rev_rr == _D("1.0")
        assert cfg.rev_require_sr is True
        assert cfg.rev_vwap_flat_tol == _D("0.10")

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(ReversalStrategyConfig(), rev_hug_limit=10)
        assert cfg.rev_hug_limit == 10
        assert cfg.rev_body == _D("0.15")  # unchanged


# ======================================================================
# ORBreakoutStrategyConfig
# ======================================================================


class TestORBreakoutStrategyConfig:
    """OR Breakout strategy config tests."""

    def test_frozen(self) -> None:
        cfg = ORBreakoutStrategyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.orb_window = 99  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(ORBreakoutStrategyConfig, InfraExitConfig)

    def test_orb_defaults(self) -> None:
        cfg = ORBreakoutStrategyConfig()
        assert cfg.orb_enable is True
        assert cfg.orb_window == 30
        assert cfg.orb_rvol == _D("2.0")
        assert cfg.orb_sl_mode == "OR Midpoint"
        assert cfg.orb_vwap_align is True
        assert cfg.orb_body_min == _D("0.25")
        assert cfg.orb_entry_mode == "aggressive"
        assert cfg.orb_tp_mode == "2x_or_range"
        assert cfg.orb_gap_filter is True
        assert cfg.orb_adx_filter is True

    def test_exit_fields_from_base(self) -> None:
        """ORB exit fields live in InfraExitConfig base."""
        cfg = ORBreakoutStrategyConfig()
        assert cfg.orb_reentry_exit is False
        assert cfg.orb_time_exit == 0
        assert cfg.orb_trail_atr == _D("0.8")

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(ORBreakoutStrategyConfig(), orb_rvol=_D("3.0"))
        assert cfg.orb_rvol == _D("3.0")


# ======================================================================
# ORReversalStrategyConfig
# ======================================================================


class TestORReversalStrategyConfig:
    """OR Reversal strategy config tests."""

    def test_frozen(self) -> None:
        cfg = ORReversalStrategyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.orr_window = 99  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(ORReversalStrategyConfig, InfraExitConfig)

    def test_orr_defaults(self) -> None:
        cfg = ORReversalStrategyConfig()
        assert cfg.orr_enable is True
        assert cfg.orr_window == 78
        assert cfg.orr_prox == _D("0.6")
        assert cfg.orr_sl_atr == _D("0.5")
        assert cfg.orr_tp_mode == "OR Mid"
        assert cfg.orr_min_rr == _D("0.5")
        assert cfg.orr_max_rr == _D("3.0")
        assert cfg.orr_rvol == _D("1.0")
        assert cfg.orr_vwap_disagree is False
        assert cfg.orr_gap_filter is True
        assert cfg.orr_adx_filter is True
        assert cfg.orr_adx_max == _D("30")
        assert cfg.orr_require_break is True

    def test_exit_field_from_base(self) -> None:
        """ORR rebreak exit lives in InfraExitConfig base."""
        cfg = ORReversalStrategyConfig()
        assert cfg.orr_rebreak_exit is False

    def test_ps_engulf_present(self) -> None:
        """ORR needs ps_engulf for pattern detection."""
        cfg = ORReversalStrategyConfig()
        assert cfg.ps_engulf == _D("0.30")

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(ORReversalStrategyConfig(), orr_tp_mode="OR Opposite")
        assert cfg.orr_tp_mode == "OR Opposite"


# ======================================================================
# PatternScalpStrategyConfig
# ======================================================================


class TestPatternScalpStrategyConfig:
    """Pattern Scalp strategy config tests."""

    def test_frozen(self) -> None:
        cfg = PatternScalpStrategyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.ps_window = 99  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(PatternScalpStrategyConfig, InfraExitConfig)

    def test_ps_defaults(self) -> None:
        cfg = PatternScalpStrategyConfig()
        assert cfg.ps_enable is True
        assert cfg.ps_window == 35
        assert cfg.ps_engulf == _D("0.25")
        assert cfg.ps_rvol == _D("1.5")
        assert cfg.ps_sma_filter is True
        assert cfg.ps_sl_mode == "ATR-Based"
        assert cfg.ps_tp_pct == _D("75.0")
        assert cfg.ps_min_rr == _D("1.5")

    def test_ps_atr_pct_override(self) -> None:
        """PS overrides ps_atr_pct from base to 20% for more manipulation days."""
        cfg = PatternScalpStrategyConfig()
        assert cfg.ps_atr_pct == _D("20.0")

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(PatternScalpStrategyConfig(), ps_tp_pct=_D("100.0"))
        assert cfg.ps_tp_pct == _D("100.0")


# ======================================================================
# Cross-config independence
# ======================================================================


class TestConfigIndependence:
    """Each config is independent and doesn't affect others."""

    def test_pb_has_no_rev_fields(self) -> None:
        cfg = PullbackStrategyConfig()
        assert not hasattr(cfg, "rev_body")
        assert not hasattr(cfg, "rev_band")

    def test_rev_has_no_pb_fields(self) -> None:
        cfg = ReversalStrategyConfig()
        assert not hasattr(cfg, "pb_zone")
        assert not hasattr(cfg, "rr")

    def test_orb_has_no_orr_fields(self) -> None:
        cfg = ORBreakoutStrategyConfig()
        assert not hasattr(cfg, "orr_prox")

    def test_ps_has_no_orb_fields(self) -> None:
        cfg = PatternScalpStrategyConfig()
        assert not hasattr(cfg, "orb_window")

    def test_avwap_has_no_pb_fields(self) -> None:
        cfg = AVWAPPullbackConfig()
        assert not hasattr(cfg, "pb_zone")
        assert not hasattr(cfg, "rr")

    def test_pb_has_no_avwap_fields(self) -> None:
        cfg = PullbackStrategyConfig()
        assert not hasattr(cfg, "avwap_zone_atr")
        assert not hasattr(cfg, "avwap_anchor_type")

    def test_smc_has_no_pb_fields(self) -> None:
        cfg = SMCStructureConfig()
        assert not hasattr(cfg, "pb_zone")
        assert not hasattr(cfg, "rr")

    def test_pb_has_no_smc_fields(self) -> None:
        cfg = PullbackStrategyConfig()
        assert not hasattr(cfg, "smc_zone_atr")
        assert not hasattr(cfg, "smc_swing_lookback")

    def test_all_share_infra_defaults(self) -> None:
        """All configs share the same infra defaults (except strategy overrides)."""
        configs = [
            PullbackStrategyConfig(),
            ReversalStrategyConfig(),
            ORBreakoutStrategyConfig(),
            ORReversalStrategyConfig(),
            PatternScalpStrategyConfig(),
            AVWAPPullbackConfig(),
            SMCStructureConfig(),
        ]
        for cfg in configs:
            assert cfg.bars_per_day == 78
            assert cfg.adx_len == 14
        # be_trigger per-strategy: PB=0.7, ORR=0.7, rest=base(0.7)
        assert PullbackStrategyConfig().be_trigger == _D("0.7")
        assert ORReversalStrategyConfig().be_trigger == _D("0.7")
        assert ReversalStrategyConfig().be_trigger == _D("0.7")
        assert ORBreakoutStrategyConfig().be_trigger == _D("0.7")
        assert PatternScalpStrategyConfig().be_trigger == _D("0.7")
        assert AVWAPPullbackConfig().be_trigger == _D("0.7")
        assert SMCStructureConfig().be_trigger == _D("0.7")
