"""Tests for the signal-monitor CLI entry point."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.analysis.alert_store import AlertStore
from stockdownloader.analysis.signal_advisor import AdvisorConfig, SignalAdvisor
from stockdownloader.app.monitor_app import (
    _analyze_symbol,
    _build_parser,
    _print_advisory,
    _run_once,
    main,
)
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.signal import (
    AdvisoryAction,
    AdvisoryReasoning,
    OptionsAdvisory,
    SignalAdvisory as SignalAdvisoryModel,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_bar(date: str, close: float) -> PriceData:
    c = Decimal(str(close))
    return PriceData(
        date=date, open=c, high=c + 1, low=c - 1, close=c, adj_close=c, volume=1_000_000,
    )


def _mock_advisory(
    symbol: str = "SPY",
    action: AdvisoryAction = AdvisoryAction.BUY,
    confidence: float = 0.75,
) -> SignalAdvisoryModel:
    return SignalAdvisoryModel(
        symbol=symbol,
        timestamp="2024-06-15T16:00:00-05:00",
        action=action,
        confidence=confidence,
        regime="strong_trend_up",
        regime_confidence=0.85,
        entry_price=475.50,
        stop_loss=471.00,
        take_profit=484.00,
        risk_reward=1.5,
        position_size_pct=1.0,
        reasoning=AdvisoryReasoning(
            signal_confluence="10/15 bullish",
            regime_alignment="Aligned",
            walk_forward_validated="OK",
            key_bullish=["RSI bounce"],
            key_bearish=["BB squeeze"],
        ),
        call_advisory=OptionsAdvisory(
            action="BUY", strike=476, dte=30, delta=0.50, est_premium=5.45,
        ),
    )


# ------------------------------------------------------------------
# _analyze_symbol
# ------------------------------------------------------------------


class TestAnalyzeSymbol:
    @patch("stockdownloader.app.monitor_app.fetch_daily_data")
    def test_no_data_returns_none(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        mock_fetch.return_value = []
        advisor = MagicMock(spec=SignalAdvisor)
        store = AlertStore(filepath=tmp_path / "test.json")

        output_lines: list[str] = []
        result = _analyze_symbol(
            "BAD", advisor, store, print_fn=output_lines.append,
        )
        assert result is None
        assert any("No data" in line for line in output_lines)

    @patch("stockdownloader.app.monitor_app.fetch_daily_data")
    def test_new_signal_returned(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        mock_fetch.return_value = [_make_bar(f"2024-01-{i+1:02d}", 100 + i) for i in range(250)]
        adv = _mock_advisory()
        advisor = MagicMock(spec=SignalAdvisor)
        advisor.evaluate.return_value = adv
        store = AlertStore(filepath=tmp_path / "test.json")

        output_lines: list[str] = []
        result = _analyze_symbol(
            "SPY", advisor, store, print_fn=output_lines.append,
        )
        assert result is not None
        assert result.action == AdvisoryAction.BUY

    @patch("stockdownloader.app.monitor_app.fetch_daily_data")
    def test_duplicate_returns_none(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        mock_fetch.return_value = [_make_bar(f"2024-01-{i+1:02d}", 100 + i) for i in range(250)]
        adv = _mock_advisory()
        advisor = MagicMock(spec=SignalAdvisor)
        advisor.evaluate.return_value = adv
        store = AlertStore(filepath=tmp_path / "test.json")

        # First call saves
        _analyze_symbol("SPY", advisor, store, print_fn=lambda x: None)
        # Second call is duplicate
        result = _analyze_symbol("SPY", advisor, store, print_fn=lambda x: None)
        assert result is None

    @patch("stockdownloader.app.monitor_app.fetch_daily_data")
    def test_below_min_confidence(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        mock_fetch.return_value = [_make_bar(f"2024-01-{i+1:02d}", 100 + i) for i in range(250)]
        adv = _mock_advisory(confidence=0.3)
        advisor = MagicMock(spec=SignalAdvisor)
        advisor.evaluate.return_value = adv
        store = AlertStore(filepath=tmp_path / "test.json")

        result = _analyze_symbol(
            "SPY", advisor, store,
            min_confidence=0.5,
            print_fn=lambda x: None,
        )
        assert result is None


# ------------------------------------------------------------------
# _print_advisory
# ------------------------------------------------------------------


class TestPrintAdvisory:
    def test_does_not_raise(self) -> None:
        adv = _mock_advisory()
        lines: list[str] = []
        _print_advisory(adv, print_fn=lines.append)
        assert len(lines) > 5
        assert any("SPY" in line for line in lines)
        assert any("BUY" in line for line in lines)
        assert any("DISCLAIMER" in line for line in lines)

    def test_hold_advisory(self) -> None:
        adv = _mock_advisory(action=AdvisoryAction.HOLD, confidence=0.0)
        lines: list[str] = []
        _print_advisory(adv, print_fn=lines.append)
        assert any("HOLD" in line for line in lines)

    def test_with_options(self) -> None:
        adv = _mock_advisory()
        lines: list[str] = []
        _print_advisory(adv, print_fn=lines.append)
        assert any("Call:" in line for line in lines)


# ------------------------------------------------------------------
# _build_parser / argparse
# ------------------------------------------------------------------


class TestParser:
    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.symbols == ["SPY"]
        assert args.monitor is False
        assert args.interval == 300
        assert args.min_confidence == 0.0

    def test_custom_symbols(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["AAPL", "MSFT"])
        assert args.symbols == ["AAPL", "MSFT"]

    def test_monitor_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--monitor", "--interval", "60"])
        assert args.monitor is True
        assert args.interval == 60

    def test_json_output(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--json-output", "out.json"])
        assert args.json_output == "out.json"

    def test_min_confidence(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--min-confidence", "0.7"])
        assert args.min_confidence == pytest.approx(0.7)


# ------------------------------------------------------------------
# main() integration
# ------------------------------------------------------------------


class TestMain:
    @patch("stockdownloader.app.monitor_app.fetch_daily_data")
    def test_one_shot_no_data(self, mock_fetch: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        mock_fetch.return_value = []
        main(["SPY", "--alert-store", str(tmp_path / "store.json")])
        captured = capsys.readouterr()
        # Either "No data" or "No new signals" should appear
        assert "No" in captured.out
