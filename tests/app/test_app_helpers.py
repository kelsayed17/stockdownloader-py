"""Tests for shared CLI application helpers."""

from __future__ import annotations

from stockdownloader.app.app_helpers import box_title, status_label


class TestStatusLabel:
    def test_robust(self):
        assert status_label(0.8) == "ROBUST"
        assert status_label(1.0) == "ROBUST"
        assert status_label(0.95) == "ROBUST"

    def test_acceptable(self):
        assert status_label(0.5) == "ACCEPTABLE"
        assert status_label(0.7) == "ACCEPTABLE"

    def test_overfit(self):
        assert status_label(0.49) == "OVERFIT"
        assert status_label(0.0) == "OVERFIT"
        assert status_label(-0.1) == "OVERFIT"


class TestBoxTitle:
    def test_basic(self):
        result = box_title("HELLO", width=20)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "HELLO" in lines[1]
        assert len(lines[0]) == 22  # width + 2 border chars

    def test_default_width(self):
        result = box_title("TEST")
        lines = result.split("\n")
        assert len(lines[0]) == 102  # 100 + 2 border chars

    def test_unicode_borders(self):
        result = box_title("X", width=10)
        lines = result.split("\n")
        assert lines[0].startswith("\u2554")
        assert lines[0].endswith("\u2557")
        assert lines[2].startswith("\u255a")
        assert lines[2].endswith("\u255d")
