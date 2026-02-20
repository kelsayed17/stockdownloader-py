"""Tests for the AVWAP Pullback strategy."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from stockdownloader.strategy.intraday.avwap_pullback_strategy import AVWAPPullbackConfig
from stockdownloader.strategy.intraday.avwap_pullback_strategy import AVWAPPullbackStrategy
from stockdownloader.strategy.intraday.base_strategy import InfraExitConfig


_D = Decimal


class TestAVWAPPullbackConfig:
    """AVWAP Pullback config tests."""

    def test_frozen(self) -> None:
        cfg = AVWAPPullbackConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.avwap_zone_atr = _D("99")  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(AVWAPPullbackConfig, InfraExitConfig)

    def test_avwap_defaults(self) -> None:
        cfg = AVWAPPullbackConfig()
        assert cfg.use_avwap is True
        assert cfg.avwap_anchor_type == "fomc"
        assert cfg.avwap_zone_atr == _D("0.8")
        assert cfg.avwap_min_days == 3
        assert cfg.avwap_max_days == 30
        assert cfg.avwap_body_min == _D("0.20")
        assert cfg.avwap_session_vwap_agree is True
        assert cfg.avwap_longs is True
        assert cfg.avwap_shorts is False
        assert cfg.avwap_trend_bars == 7
        assert cfg.avwap_htf_align is True
        assert cfg.avwap_sl_atr == _D("1.3")
        assert cfg.avwap_sl_cap == _D("2.00")
        assert cfg.avwap_rr == _D("2.0")
        assert cfg.avwap_tp_mode == "rr"
        assert cfg.max_day == 1
        assert cfg.spacing == 10
        assert cfg.w_vol == 2
        assert cfg.w_sr == 2
        assert cfg.w_rsi == 1
        assert cfg.w_time == 1
        assert cfg.min_score == 4
        assert cfg.sr_avwap is True

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(AVWAPPullbackConfig(), avwap_rr=_D("3.0"))
        assert cfg.avwap_rr == _D("3.0")
        assert cfg.avwap_zone_atr == _D("0.8")  # unchanged

    def test_inherits_infra_defaults(self) -> None:
        cfg = AVWAPPullbackConfig()
        assert cfg.bars_per_day == 78
        assert cfg.adx_len == 14
        assert cfg.be_trigger == _D("0.7")


class TestAVWAPPullbackStrategy:
    """Strategy construction and interface tests."""

    def test_construction_default(self) -> None:
        s = AVWAPPullbackStrategy()
        assert s.name == "AVWAP Pullback"
        assert s.warmup_period > 0

    def test_construction_with_config(self) -> None:
        cfg = AVWAPPullbackConfig(avwap_rr=_D("2.0"))
        s = AVWAPPullbackStrategy(config=cfg)
        assert s._c.avwap_rr == _D("2.0")

    def test_has_required_methods(self) -> None:
        s = AVWAPPullbackStrategy()
        assert callable(s.evaluate)
        assert callable(s.on_session_start)
        assert callable(s.on_position_opened)
        assert callable(s.on_position_closed)
