"""Test StrategyConfigMixin serialization round-trip."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

import pytest

from stockdownloader.strategies.config_base import StrategyConfigMixin


# -- Test fixture config -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _SampleConfig(StrategyConfigMixin):
    """Minimal config for testing the mixin."""

    period: int = 14
    threshold: Decimal = Decimal("1.5")
    enabled: bool = True
    label: str = "test"


# -- Tests ---------------------------------------------------------------------


class TestToDictToJson:
    def test_to_dict_returns_plain_dict(self):
        c = _SampleConfig()
        d = c.to_dict()
        assert d == {
            "period": 14,
            "threshold": Decimal("1.5"),
            "enabled": True,
            "label": "test",
        }

    def test_to_json_decimal_as_string(self):
        c = _SampleConfig(threshold=Decimal("2.5"))
        j = c.to_json()
        parsed = json.loads(j)
        # Decimal is serialized as a string, not float
        assert parsed["threshold"] == "2.5"
        assert isinstance(parsed["threshold"], str)

    def test_to_json_includes_all_fields(self):
        c = _SampleConfig()
        parsed = json.loads(c.to_json())
        assert set(parsed.keys()) == {"period", "threshold", "enabled", "label"}


class TestFromDictFromJson:
    def test_from_dict_basic(self):
        c = _SampleConfig.from_dict({"period": 7, "threshold": "3.0"})
        assert c.period == 7
        assert c.threshold == Decimal("3.0")

    def test_from_dict_coerces_decimal_from_string(self):
        c = _SampleConfig.from_dict({"threshold": "0.123"})
        assert c.threshold == Decimal("0.123")
        assert isinstance(c.threshold, Decimal)

    def test_from_dict_coerces_decimal_from_float(self):
        c = _SampleConfig.from_dict({"threshold": 2.5})
        assert c.threshold == Decimal("2.5")

    def test_from_dict_coerces_int_from_float(self):
        c = _SampleConfig.from_dict({"period": 7.0})
        assert c.period == 7
        assert isinstance(c.period, int)

    def test_from_dict_preserves_bool(self):
        c = _SampleConfig.from_dict({"enabled": False})
        assert c.enabled is False

    def test_from_dict_unknown_field_raises(self):
        with pytest.raises(TypeError):
            _SampleConfig.from_dict({"nonexistent": 42})


class TestRoundTrips:
    def test_dict_round_trip(self):
        c = _SampleConfig(period=7, threshold=Decimal("3.0"), enabled=False, label="x")
        c2 = _SampleConfig.from_dict(c.to_dict())
        assert c2 == c

    def test_json_round_trip(self):
        c = _SampleConfig(period=21, threshold=Decimal("0.5"), enabled=False)
        c2 = _SampleConfig.from_json(c.to_json())
        assert c2 == c

    def test_json_round_trip_default_values(self):
        c = _SampleConfig()
        c2 = _SampleConfig.from_json(c.to_json())
        assert c2 == c


class TestFileIO:
    def test_save_and_load(self, tmp_path):
        c = _SampleConfig(period=5, threshold=Decimal("99.99"))
        path = tmp_path / "config.json"
        c.save(path)
        c2 = _SampleConfig.load(path)
        assert c2 == c

    def test_save_creates_valid_json(self, tmp_path):
        c = _SampleConfig()
        path = tmp_path / "config.json"
        c.save(path)
        parsed = json.loads(path.read_text())
        assert parsed["period"] == 14
