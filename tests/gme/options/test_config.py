"""Tests for GMEOptionsConfig."""
from __future__ import annotations

import pytest
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestGMEOptionsConfig:
    def test_defaults(self):
        cfg = GMEOptionsConfig()
        assert cfg.symbol == "GME"
        assert cfg.rate_limit_delay == 0.2
        assert cfg.risk_per_trade_pct == 1.0
        assert cfg.sl_atr_mult == 1.5
        assert cfg.rr_ratio == 1.5
        assert cfg.max_trades_per_day == 4
        assert cfg.circuit_breaker_losses == 3

    def test_custom_params(self):
        cfg = GMEOptionsConfig(symbol="AMC", rate_limit_delay=0.5)
        assert cfg.symbol == "AMC"
        assert cfg.rate_limit_delay == 0.5

    def test_frozen(self):
        cfg = GMEOptionsConfig()
        with pytest.raises(AttributeError):
            cfg.symbol = "AMC"

    def test_from_env_with_overrides(self):
        cfg = GMEOptionsConfig.from_env(symbol="BBBY")
        assert cfg.symbol == "BBBY"

    def test_data_dir_default(self):
        cfg = GMEOptionsConfig()
        assert "data/GME/options" in str(cfg.data_dir)
