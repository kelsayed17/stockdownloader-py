"""Tests for the JSON config loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stockdownloader.util.config_loader import Config, load_config, _deep_merge


class TestConfig:
    """Tests for the Config wrapper."""

    def test_empty_config(self) -> None:
        cfg = Config()
        assert cfg.get("anything") is None
        assert cfg.get("anything", 42) == 42

    def test_flat_access(self) -> None:
        cfg = Config({"a": 1, "b": "hello"})
        assert cfg.get("a") == 1
        assert cfg.get("b") == "hello"
        assert cfg.get("c", "default") == "default"

    def test_dot_notation(self) -> None:
        cfg = Config({"training": {"n_estimators": 200, "max_depth": 4}})
        assert cfg.get("training.n_estimators") == 200
        assert cfg.get("training.max_depth") == 4
        assert cfg.get("training.missing", 99) == 99

    def test_deep_dot_notation(self) -> None:
        cfg = Config({"a": {"b": {"c": {"d": 42}}}})
        assert cfg.get("a.b.c.d") == 42
        assert cfg.get("a.b.c.e") is None

    def test_getitem(self) -> None:
        cfg = Config({"key": "value"})
        assert cfg["key"] == "value"
        with pytest.raises(KeyError):
            _ = cfg["missing"]

    def test_contains(self) -> None:
        cfg = Config({"present": 1})
        assert "present" in cfg
        assert "absent" not in cfg

    def test_section(self) -> None:
        cfg = Config({"training": {"n_estimators": 200}})
        sub = cfg.section("training")
        assert sub.get("n_estimators") == 200

    def test_section_missing(self) -> None:
        cfg = Config({})
        sub = cfg.section("missing")
        assert sub.get("anything") is None

    def test_merge(self) -> None:
        base = Config({"a": 1, "b": {"x": 10}})
        merged = base.merge({"a": 2, "b": {"y": 20}})
        assert merged.get("a") == 2
        assert merged.get("b.x") == 10
        assert merged.get("b.y") == 20
        # Original unchanged
        assert base.get("a") == 1

    def test_repr(self) -> None:
        cfg = Config({"a": 1})
        assert "Config" in repr(cfg)


class TestDeepMerge:
    def test_flat(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        over = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, over)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}


class TestLoadConfig:
    def test_from_data(self) -> None:
        cfg = load_config(data={"key": "val"})
        assert cfg.get("key") == "val"

    def test_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"training": {"n_estimators": 500}}))
        cfg = load_config(str(p))
        assert cfg.get("training.n_estimators") == 500

    def test_with_overrides(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"a": 1, "b": 2}))
        cfg = load_config(str(p), overrides={"b": 3, "c": 4})
        assert cfg.get("a") == 1
        assert cfg.get("b") == 3
        assert cfg.get("c") == 4

    def test_missing_file(self) -> None:
        cfg = load_config("/nonexistent/path.json")
        assert cfg.data == {}

    def test_none_path(self) -> None:
        cfg = load_config(None)
        assert cfg.data == {}

    def test_none_with_overrides(self) -> None:
        cfg = load_config(None, overrides={"key": "val"})
        assert cfg.get("key") == "val"
