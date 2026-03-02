"""Smoke tests for scripts/spy_tournament_compare.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ and scripts/ are importable
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))


def test_import():
    """Verify that run_comparison is importable from the runner script."""
    from spy_tournament_compare import run_comparison

    assert callable(run_comparison)


def test_tournament_strategies_registered():
    """Verify all 3 tournament strategies instantiate correctly."""
    from stockdownloader.strategies.loader import ensure_registered
    from stockdownloader.strategies.registry import StrategyRegistry

    ensure_registered()

    keys = ["spy-macd-obv", "spy-sma-2021", "spy-macd-opt"]
    expected_classes = [
        "MACDOBVStrategy",
        "SMACross2021Strategy",
        "MACDOptimizedStrategy",
    ]

    for key, expected_cls in zip(keys, expected_classes):
        entry = StrategyRegistry.get(key)
        assert entry is not None, f"Strategy '{key}' not found in registry"
        assert entry.category == "intraday"

        # Instantiate with default kwargs
        strategy = entry.factory(**entry.default_kwargs)
        assert type(strategy).__name__ == expected_cls
        assert hasattr(strategy, "evaluate")
        assert hasattr(strategy, "name")
