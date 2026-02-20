"""Tests for AlertStore — JSON-file-backed advisory persistence."""

from __future__ import annotations

import json

import pytest

from stockdownloader.analysis.alert_store import AlertStore
from stockdownloader.model.signal_advisory import (
    AdvisoryAction,
    AdvisoryReasoning,
    SignalAdvisory,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _reasoning() -> AdvisoryReasoning:
    return AdvisoryReasoning(
        signal_confluence="10/15 bullish",
        regime_alignment="Aligned",
        walk_forward_validated="OK",
    )


def _advisory(
    symbol: str = "SPY",
    action: AdvisoryAction = AdvisoryAction.BUY,
    timestamp: str = "2024-06-15T16:00:00-05:00",
    confidence: float = 0.75,
) -> SignalAdvisory:
    return SignalAdvisory(
        symbol=symbol,
        timestamp=timestamp,
        action=action,
        confidence=confidence,
        regime="strong_trend_up",
        regime_confidence=0.85,
        entry_price=475.50,
        stop_loss=471.00,
        take_profit=484.00,
        risk_reward=1.5,
        position_size_pct=1.0,
        reasoning=_reasoning(),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestAlertStoreSaveRetrieve:
    def test_save_and_get_latest(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        adv = _advisory()
        assert store.save(adv) is True
        latest = store.get_latest("SPY")
        assert latest is not None
        assert latest["symbol"] == "SPY"
        assert latest["action"] == "BUY"

    def test_save_returns_false_for_duplicate(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        adv = _advisory()
        assert store.save(adv) is True
        assert store.save(adv) is False  # duplicate

    def test_different_action_same_day_not_duplicate(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        adv_buy = _advisory(action=AdvisoryAction.BUY)
        adv_sell = _advisory(action=AdvisoryAction.SELL)
        assert store.save(adv_buy) is True
        assert store.save(adv_sell) is True

    def test_different_day_same_action_not_duplicate(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        adv1 = _advisory(timestamp="2024-06-15T16:00:00-05:00")
        adv2 = _advisory(timestamp="2024-06-16T16:00:00-05:00")
        assert store.save(adv1) is True
        assert store.save(adv2) is True

    def test_newest_first_ordering(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        adv1 = _advisory(timestamp="2024-06-10T16:00:00-05:00")
        adv2 = _advisory(timestamp="2024-06-11T16:00:00-05:00")
        adv3 = _advisory(timestamp="2024-06-12T16:00:00-05:00")
        store.save(adv1)
        store.save(adv2)
        store.save(adv3)

        history = store.get_history("SPY")
        assert len(history) == 3
        # Newest inserted last → sits at index 0
        assert history[0]["timestamp"] == "2024-06-12T16:00:00-05:00"

    def test_get_history_limits(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        for i in range(10):
            store.save(_advisory(
                timestamp=f"2024-06-{10 + i:02d}T16:00:00-05:00",
                action=AdvisoryAction.BUY if i % 2 == 0 else AdvisoryAction.SELL,
            ))

        assert len(store.get_history("SPY", last_n=5)) == 5

    def test_get_latest_empty(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)
        assert store.get_latest("SPY") is None

    def test_get_history_empty(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)
        assert store.get_history("SPY") == []


class TestAlertStoreClear:
    def test_clear_symbol(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        store.save(_advisory(symbol="SPY"))
        store.save(_advisory(symbol="AAPL"))
        removed = store.clear("SPY")
        assert removed == 1
        assert store.get_latest("SPY") is None
        assert store.get_latest("AAPL") is not None

    def test_clear_all(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)

        store.save(_advisory(symbol="SPY"))
        store.save(_advisory(symbol="AAPL"))
        removed = store.clear()
        assert removed == 2
        assert store.get_latest("SPY") is None
        assert store.get_latest("AAPL") is None

    def test_clear_nonexistent_symbol(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)
        assert store.clear("NOPE") == 0


class TestAlertStorePruning:
    def test_max_history_pruning(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path, max_history=3)

        for i in range(5):
            store.save(_advisory(
                timestamp=f"2024-06-{10 + i:02d}T16:00:00-05:00",
                # Alternate actions to avoid dedup
                action=AdvisoryAction.BUY if i % 3 == 0 else (
                    AdvisoryAction.SELL if i % 3 == 1 else AdvisoryAction.HOLD
                ),
            ))

        history = store.get_history("SPY", last_n=100)
        assert len(history) <= 3


class TestAlertStorePersistence:
    def test_reload_from_disk(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]

        store1 = AlertStore(filepath=path)
        store1.save(_advisory())

        # New instance reads from same file
        store2 = AlertStore(filepath=path)
        latest = store2.get_latest("SPY")
        assert latest is not None
        assert latest["action"] == "BUY"

    def test_corrupted_file_handled(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        path.write_text("not json at all", encoding="utf-8")

        store = AlertStore(filepath=path)
        assert store.get_latest("SPY") is None
        # Should still be usable
        store.save(_advisory())
        assert store.get_latest("SPY") is not None

    def test_non_dict_file_handled(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        path.write_text("[1, 2, 3]", encoding="utf-8")

        store = AlertStore(filepath=path)
        assert store.get_latest("SPY") is None

    def test_empty_file_handled(self, tmp_path: object) -> None:
        path = tmp_path / "alerts.json"  # type: ignore[operator]
        path.write_text("", encoding="utf-8")

        store = AlertStore(filepath=path)
        assert store.get_latest("SPY") is None

    def test_creates_parent_dirs(self, tmp_path: object) -> None:
        path = tmp_path / "deep" / "nested" / "alerts.json"  # type: ignore[operator]
        store = AlertStore(filepath=path)
        store.save(_advisory())
        assert path.exists()
