"""Tests for the SMC Structure strategy."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from stockdownloader.strategy.intraday.smc_structure_strategy import SMCStructureConfig
from stockdownloader.strategy.intraday.smc_structure_strategy import SMCStructureStrategy
from stockdownloader.strategy.intraday.base_strategy import InfraExitConfig


_D = Decimal


class TestSMCStructureConfig:
    """SMC Structure config tests."""

    def test_frozen(self) -> None:
        cfg = SMCStructureConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.smc_zone_atr = _D("99")  # type: ignore[misc]

    def test_inherits_infra_exit(self) -> None:
        assert issubclass(SMCStructureConfig, InfraExitConfig)

    def test_smc_defaults(self) -> None:
        cfg = SMCStructureConfig()
        assert cfg.use_smc is True
        assert cfg.smc_swing_lookback == 3
        assert cfg.smc_min_impulse_atr == _D("2.5")
        assert cfg.smc_zone_bars == 2
        assert cfg.smc_require_bos is True
        assert cfg.smc_zone_atr == _D("0.8")
        assert cfg.smc_body_min == _D("0.10")
        assert cfg.smc_longs is True
        assert cfg.smc_shorts is False
        assert cfg.smc_sweep_entry is True
        assert cfg.smc_sweep_tolerance == _D("0.3")
        assert cfg.smc_htf_align is True
        assert cfg.smc_vwap_agree is False
        assert cfg.smc_min_age == 10
        assert cfg.smc_max_age == 60
        assert cfg.smc_sl_mode == "atr"
        assert cfg.smc_sl_atr == _D("2.0")
        assert cfg.smc_sl_cap == _D("1.50")
        assert cfg.smc_rr == _D("1.0")
        assert cfg.smc_tp_mode == "rr"
        assert cfg.max_day == 2
        assert cfg.spacing == 10
        assert cfg.w_vol == 2
        assert cfg.w_sr == 2
        assert cfg.w_rsi == 1
        assert cfg.w_time == 1
        assert cfg.min_score == 3

    def test_replace_override(self) -> None:
        cfg = dataclasses.replace(SMCStructureConfig(), smc_rr=_D("3.0"))
        assert cfg.smc_rr == _D("3.0")
        assert cfg.smc_zone_atr == _D("0.8")  # unchanged

    def test_inherits_infra_defaults(self) -> None:
        cfg = SMCStructureConfig()
        assert cfg.bars_per_day == 78
        assert cfg.adx_len == 14
        assert cfg.be_trigger == _D("0.7")


class TestSMCStructureStrategy:
    """Strategy construction and interface tests."""

    def test_construction_default(self) -> None:
        s = SMCStructureStrategy()
        assert s.name == "SMC Structure"
        assert s.warmup_period > 0

    def test_construction_with_config(self) -> None:
        cfg = SMCStructureConfig(smc_rr=_D("2.0"))
        s = SMCStructureStrategy(config=cfg)
        assert s._c.smc_rr == _D("2.0")

    def test_has_required_methods(self) -> None:
        s = SMCStructureStrategy()
        assert callable(s.evaluate)
        assert callable(s.on_session_start)
        assert callable(s.on_position_opened)
        assert callable(s.on_position_closed)
