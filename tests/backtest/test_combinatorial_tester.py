"""Tests for CombinatorialTester, CombinatorialConfig, and related helpers."""
from __future__ import annotations

import io
import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
from typing import Any

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.combinatorial_tester import (
    CombinatorialConfig,
    CombinatorialTester,
    ComboResult,
)
from stockdownloader.util.file_helper import TeeWriter
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.strategy.signals.signal_registry import (
    SignalGeneratorRegistry,
)
from stockdownloader.strategy.signals.stacked_signal_engine import (
    AggregationMode,
)
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.timeframe_aggregator import Timeframe


# ======================================================================
# Helpers
# ======================================================================


def _make_intraday_bar(
    close: str,
    date: str = "2025-01-02 09:30:00-05:00",
    volume: int = 10000,
) -> IntradayPriceData:
    c = Decimal(close)
    return IntradayPriceData(
        date=date,
        open=c - Decimal("0.50"),
        high=c + Decimal("1"),
        low=c - Decimal("1"),
        close=c,
        adj_close=c,
        volume=volume,
    )


def _make_session(
    n_bars: int = 78,
    base_price: float = 100.0,
    date: str = "2025-01-02",
) -> list[IntradayPriceData]:
    """Generate a single trading day of 5-minute bars."""
    bars = []
    for i in range(n_bars):
        hour = 9 + (i * 5 + 30) // 60
        minute = (i * 5 + 30) % 60
        timestamp = f"{date} {hour:02d}:{minute:02d}:00-05:00"
        c = Decimal(str(round(base_price + i * 0.05, 2)))
        bars.append(IntradayPriceData(
            date=timestamp,
            open=c - Decimal("0.05"),
            high=c + Decimal("0.50"),
            low=c - Decimal("0.50"),
            close=c,
            adj_close=c,
            volume=10000 + i * 100,
        ))
    return bars


def _make_multi_day_data(n_days: int = 3, bars_per_day: int = 78) -> list[IntradayPriceData]:
    """Generate multi-day intraday data."""
    all_bars = []
    for day in range(n_days):
        date_str = f"2025-01-{day + 2:02d}"
        base = 100.0 + day * 5
        all_bars.extend(_make_session(
            n_bars=bars_per_day,
            base_price=base,
            date=date_str,
        ))
    return all_bars


class _SimpleTestGenerator(AtomicSignalGenerator):
    """Minimal generator for testing combinatorial logic."""

    def __init__(
        self,
        name: str = "simple",
        category: str = "momentum",
        score: float = 0.0,
    ):
        self._name = name
        self._category = category
        self._score = score

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def warmup_period(self) -> int:
        return 2

    def evaluate(self, data, index, hub) -> SignalResult:
        if index < 2:
            return SignalResult.neutral()
        direction = (
            SignalDirection.BULLISH if self._score > 0.1
            else SignalDirection.BEARISH if self._score < -0.1
            else SignalDirection.NEUTRAL
        )
        return SignalResult(
            score=self._score,
            direction=direction,
            fired=abs(self._score) > 0.1,
        )

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {}


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(SignalGeneratorRegistry._entries)
    SignalGeneratorRegistry.clear()
    yield
    SignalGeneratorRegistry._entries = saved


def _register_test_generators():
    """Register a small set of test generators across categories."""
    SignalGeneratorRegistry.register(
        name="test_rsi",
        display_name="Test RSI",
        category="momentum",
        factory=lambda: _SimpleTestGenerator("test_rsi", "momentum", 0.5),
    )
    SignalGeneratorRegistry.register(
        name="test_macd",
        display_name="Test MACD",
        category="momentum",
        factory=lambda: _SimpleTestGenerator("test_macd", "momentum", 0.3),
    )
    SignalGeneratorRegistry.register(
        name="test_sma",
        display_name="Test SMA",
        category="trend",
        factory=lambda: _SimpleTestGenerator("test_sma", "trend", 0.4),
    )
    SignalGeneratorRegistry.register(
        name="test_adx",
        display_name="Test ADX",
        category="trend",
        factory=lambda: _SimpleTestGenerator("test_adx", "trend", 0.2),
    )
    SignalGeneratorRegistry.register(
        name="test_bb",
        display_name="Test BB",
        category="volatility",
        factory=lambda: _SimpleTestGenerator("test_bb", "volatility", 0.6),
    )
    SignalGeneratorRegistry.register(
        name="test_obv",
        display_name="Test OBV",
        category="volume",
        factory=lambda: _SimpleTestGenerator("test_obv", "volume", 0.1),
    )


# ======================================================================
# CombinatorialConfig tests
# ======================================================================


class TestCombinatorialConfig:
    def test_defaults(self):
        cfg = CombinatorialConfig()
        assert cfg.min_combo_size == 2
        assert cfg.max_combo_size == 4
        assert cfg.min_category_diversity == 2
        assert cfg.timeframes == [Timeframe.M5]
        assert cfg.weights == [1.0]
        assert AggregationMode.WEIGHTED_AVERAGE in cfg.modes
        assert AggregationMode.UNANIMOUS in cfg.modes
        assert cfg.require_fire is True
        assert cfg.allow_shorts is True
        assert cfg.initial_capital == Decimal("100000")
        assert cfg.risk_per_trade == Decimal("0.01")
        assert cfg.generator_names is None

    def test_custom_values(self):
        cfg = CombinatorialConfig(
            min_combo_size=3,
            max_combo_size=5,
            min_category_diversity=3,
            timeframes=[Timeframe.M5, Timeframe.H1],
            weights=[0.5, 1.0, 2.0],
            modes=[AggregationMode.MAJORITY_VOTE],
            buy_thresholds=[0.4],
            sell_thresholds=[0.4],
            require_fire=False,
            allow_shorts=False,
            initial_capital=Decimal("50000"),
            risk_per_trade=Decimal("0.02"),
            generator_names=["rsi", "macd"],
        )
        assert cfg.min_combo_size == 3
        assert cfg.max_combo_size == 5
        assert cfg.min_category_diversity == 3
        assert cfg.generator_names == ["rsi", "macd"]


# ======================================================================
# ComboResult tests
# ======================================================================


class TestComboResult:
    def test_construction(self):
        bt_result = BacktestResult("test", Decimal("100000"))
        combo = ComboResult(
            generator_names=("rsi", "macd"),
            timeframe=Timeframe.M5,
            weight=1.0,
            mode=AggregationMode.WEIGHTED_AVERAGE,
            buy_threshold=0.3,
            sell_threshold=0.3,
            result=bt_result,
            score=42.5,
        )
        assert combo.generator_names == ("rsi", "macd")
        assert combo.timeframe == Timeframe.M5
        assert combo.weight == 1.0
        assert combo.mode == AggregationMode.WEIGHTED_AVERAGE
        assert combo.buy_threshold == 0.3
        assert combo.sell_threshold == 0.3
        assert combo.score == 42.5

    def test_frozen(self):
        bt_result = BacktestResult("test", Decimal("100000"))
        combo = ComboResult(
            generator_names=("rsi",),
            timeframe=Timeframe.M5,
            weight=1.0,
            mode=AggregationMode.WEIGHTED_AVERAGE,
            buy_threshold=0.3,
            sell_threshold=0.3,
            result=bt_result,
            score=0.0,
        )
        with pytest.raises(AttributeError):
            combo.score = 1.0  # type: ignore[misc]


# ======================================================================
# TeeWriter tests
# ======================================================================


class TestTeeWriter:
    def test_tee_writes_to_both(self):
        file_buf = io.StringIO()
        tee = TeeWriter(file_buf)

        # Capture stdout
        import sys
        old_stdout = sys.stdout
        stdout_buf = io.StringIO()
        sys.stdout = stdout_buf
        try:
            tee.write("hello world")
        finally:
            sys.stdout = old_stdout

        assert file_buf.getvalue() == "hello world"
        assert stdout_buf.getvalue() == "hello world"


# ======================================================================
# _generate_combos tests
# ======================================================================


class TestGenerateCombos:
    def test_basic_combo_generation(self):
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        gen_names = SignalGeneratorRegistry.list_names()

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
        )
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)

        # All combos should have exactly 2 generators
        assert all(len(c) == 2 for c in combos)

        # All combos should satisfy category diversity >= 2
        for combo in combos:
            categories = {entries[n].category for n in combo}
            assert len(categories) >= 2

    def test_category_diversity_filter(self):
        """Combos with same category should be excluded when diversity >= 2."""
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        gen_names = SignalGeneratorRegistry.list_names()

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
        )
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)

        # test_rsi + test_macd are both "momentum" -> should be excluded
        same_category = ("test_macd", "test_rsi")
        assert same_category not in combos

    def test_diversity_one_allows_same_category(self):
        """With min_category_diversity=1, same-category combos are allowed."""
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        gen_names = SignalGeneratorRegistry.list_names()

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=1,
        )
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)

        # Now same-category combos should be included
        same_category = ("test_macd", "test_rsi")
        assert same_category in combos

    def test_combo_sizes(self):
        """Combos of sizes min through max should be generated."""
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        gen_names = SignalGeneratorRegistry.list_names()

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=4,
            min_category_diversity=1,
        )
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)

        sizes = {len(c) for c in combos}
        assert 2 in sizes
        assert 3 in sizes
        assert 4 in sizes

    def test_combos_are_sorted_tuples(self):
        """Each combo is a sorted tuple of generator names."""
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        gen_names = SignalGeneratorRegistry.list_names()

        cfg = CombinatorialConfig(min_combo_size=2, max_combo_size=3, min_category_diversity=1)
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)

        for combo in combos:
            assert isinstance(combo, tuple)
            assert list(combo) == sorted(combo)

    def test_empty_gen_names(self):
        """No generator names produces no combos."""
        entries = {}
        cfg = CombinatorialConfig(min_combo_size=2, max_combo_size=3, min_category_diversity=1)
        combos = CombinatorialTester._generate_combos([], entries, cfg)
        assert combos == []

    def test_high_diversity_excludes_all(self):
        """If diversity requirement exceeds available categories, no combos."""
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        # Only use 2 momentum generators
        gen_names = ["test_rsi", "test_macd"]

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
        )
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)
        # Both are "momentum", can't satisfy diversity >= 2
        assert combos == []

    def test_specific_generator_names(self):
        """Only specified generator names are used."""
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        gen_names = ["test_rsi", "test_sma"]

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=1,
        )
        combos = CombinatorialTester._generate_combos(gen_names, entries, cfg)
        assert len(combos) == 1
        assert combos[0] == ("test_rsi", "test_sma")


# ======================================================================
# format_results tests
# ======================================================================


class TestFormatResults:
    def test_format_empty_results(self):
        output = CombinatorialTester.format_results([], top_n=10)
        assert "TOP" in output
        assert "0" in output

    def test_format_with_results(self):
        bt_result = BacktestResult("test", Decimal("100000"))
        bt_result.final_capital = Decimal("110000")
        bt_result.equity_curve = [
            Decimal("100000"), Decimal("105000"), Decimal("110000"),
        ]

        combo = ComboResult(
            generator_names=("rsi", "macd"),
            timeframe=Timeframe.M5,
            weight=1.0,
            mode=AggregationMode.WEIGHTED_AVERAGE,
            buy_threshold=0.3,
            sell_threshold=0.3,
            result=bt_result,
            score=42.5,
        )
        output = CombinatorialTester.format_results([combo], top_n=10)
        assert "rsi+macd" in output
        assert "42.5" in output or "42.50" in output

    def test_format_top_n_limits(self):
        results = []
        for i in range(20):
            bt_result = BacktestResult(f"test_{i}", Decimal("100000"))
            bt_result.final_capital = Decimal("100000")
            bt_result.equity_curve = [Decimal("100000")]
            results.append(ComboResult(
                generator_names=(f"gen_{i}",),
                timeframe=Timeframe.M5,
                weight=1.0,
                mode=AggregationMode.WEIGHTED_AVERAGE,
                buy_threshold=0.3,
                sell_threshold=0.3,
                result=bt_result,
                score=float(i),
            ))
        output = CombinatorialTester.format_results(results, top_n=5)
        # Should contain TOP 5
        assert "TOP 5" in output


# ======================================================================
# frequency_analysis tests
# ======================================================================


class TestFrequencyAnalysis:
    def test_frequency_analysis(self):
        results = []
        bt = BacktestResult("test", Decimal("100000"))
        bt.final_capital = Decimal("100000")
        bt.equity_curve = [Decimal("100000")]

        for _ in range(3):
            results.append(ComboResult(
                generator_names=("rsi", "macd"),
                timeframe=Timeframe.M5,
                weight=1.0,
                mode=AggregationMode.WEIGHTED_AVERAGE,
                buy_threshold=0.3,
                sell_threshold=0.3,
                result=bt,
                score=1.0,
            ))
        results.append(ComboResult(
            generator_names=("sma", "macd"),
            timeframe=Timeframe.M5,
            weight=1.0,
            mode=AggregationMode.WEIGHTED_AVERAGE,
            buy_threshold=0.3,
            sell_threshold=0.3,
            result=bt,
            score=0.5,
        ))

        output = CombinatorialTester.frequency_analysis(results, top_n=10)
        assert "GENERATOR FREQUENCY" in output
        assert "macd" in output
        assert "rsi" in output
        assert "sma" in output


# ======================================================================
# category_pair_analysis tests
# ======================================================================


class TestCategoryPairAnalysis:
    def test_category_pair_analysis(self):
        _register_test_generators()
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}

        bt = BacktestResult("test", Decimal("100000"))
        bt.final_capital = Decimal("110000")
        bt.equity_curve = [Decimal("100000"), Decimal("110000")]

        results = [
            ComboResult(
                generator_names=("test_rsi", "test_sma"),
                timeframe=Timeframe.M5,
                weight=1.0,
                mode=AggregationMode.WEIGHTED_AVERAGE,
                buy_threshold=0.3,
                sell_threshold=0.3,
                result=bt,
                score=10.0,
            ),
            ComboResult(
                generator_names=("test_bb", "test_obv"),
                timeframe=Timeframe.M5,
                weight=1.0,
                mode=AggregationMode.WEIGHTED_AVERAGE,
                buy_threshold=0.3,
                sell_threshold=0.3,
                result=bt,
                score=8.0,
            ),
        ]

        output = CombinatorialTester.category_pair_analysis(results, entries)
        assert "BEST COMBO PER CATEGORY PAIR" in output
        assert "momentum" in output
        assert "trend" in output


# ======================================================================
# CombinatorialTester.run integration test (mocked backtest)
# ======================================================================


class TestCombinatorialTesterRun:
    def test_run_with_mock_engine(self):
        """Test run() end-to-end with mocked backtest engine."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE],
            buy_thresholds=[0.3],
            sell_thresholds=[0.3],
            generator_names=["test_rsi", "test_sma"],
            initial_capital=Decimal("100000"),
        )

        # Create a mock BacktestResult
        mock_result = BacktestResult("mock", Decimal("100000"))
        mock_result.final_capital = Decimal("105000")
        mock_result.equity_curve = [
            Decimal("100000"), Decimal("102000"), Decimal("105000"),
        ]

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.return_value = mock_result

            tester = CombinatorialTester(cfg, data)
            results = tester.run()

        assert len(results) >= 1
        # Results should be sorted by score descending
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_run_uses_generator_names_from_config(self):
        """When generator_names is set, only those generators are used."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE],
            buy_thresholds=[0.3],
            sell_thresholds=[0.3],
            generator_names=["test_rsi", "test_sma"],
        )

        mock_result = BacktestResult("mock", Decimal("100000"))
        mock_result.final_capital = Decimal("100000")
        mock_result.equity_curve = [Decimal("100000")]

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.return_value = mock_result

            tester = CombinatorialTester(cfg, data)
            results = tester.run()

        # With 2 generators from different categories and diversity=2,
        # there should be exactly 1 combo: (test_rsi, test_sma)
        assert len(results) == 1
        assert results[0].generator_names == ("test_rsi", "test_sma")

    def test_run_handles_backtest_failure(self):
        """Backtest exceptions are caught and the combo is skipped."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE],
            buy_thresholds=[0.3],
            sell_thresholds=[0.3],
            generator_names=["test_rsi", "test_sma"],
        )

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.side_effect = RuntimeError("Boom")

            tester = CombinatorialTester(cfg, data)
            results = tester.run()

        # Should return empty list since all backtests failed
        assert results == []

    def test_run_sorts_by_score_descending(self):
        """Results are sorted by score in descending order."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE],
            buy_thresholds=[0.2, 0.3],
            sell_thresholds=[0.3],
            generator_names=["test_rsi", "test_sma"],
        )

        # Return different results for different calls
        call_count = [0]

        def mock_run(strategy, data):
            call_count[0] += 1
            result = BacktestResult("mock", Decimal("100000"))
            result.final_capital = Decimal(str(100000 + call_count[0] * 1000))
            result.equity_curve = [Decimal("100000"), result.final_capital]
            return result

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.side_effect = mock_run

            tester = CombinatorialTester(cfg, data)
            results = tester.run()

        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_run_with_log_file(self):
        """Log file receives progress output."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE],
            buy_thresholds=[0.3],
            sell_thresholds=[0.3],
            generator_names=["test_rsi", "test_sma"],
        )

        mock_result = BacktestResult("mock", Decimal("100000"))
        mock_result.final_capital = Decimal("100000")
        mock_result.equity_curve = [Decimal("100000")]

        log_buf = io.StringIO()

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.return_value = mock_result

            tester = CombinatorialTester(cfg, data, log_file=log_buf)
            tester.run()

        log_output = log_buf.getvalue()
        assert "Generated" in log_output
        assert "Completed" in log_output

    def test_run_multiple_modes(self):
        """Multiple aggregation modes multiply the configuration count."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE, AggregationMode.UNANIMOUS],
            buy_thresholds=[0.3],
            sell_thresholds=[0.3],
            generator_names=["test_rsi", "test_sma"],
        )

        mock_result = BacktestResult("mock", Decimal("100000"))
        mock_result.final_capital = Decimal("100000")
        mock_result.equity_curve = [Decimal("100000")]

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.return_value = mock_result

            tester = CombinatorialTester(cfg, data)
            results = tester.run()

        # 1 combo * 2 modes * 1 weight * 1 timeframe * 1 buy * 1 sell = 2
        assert len(results) == 2
        modes = {r.mode for r in results}
        assert modes == {AggregationMode.WEIGHTED_AVERAGE, AggregationMode.UNANIMOUS}

    def test_run_multiple_thresholds(self):
        """Multiple thresholds multiply configuration count."""
        _register_test_generators()

        data = _make_multi_day_data(n_days=2, bars_per_day=78)

        cfg = CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=2,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE],
            buy_thresholds=[0.2, 0.3],
            sell_thresholds=[0.3, 0.5],
            generator_names=["test_rsi", "test_sma"],
        )

        mock_result = BacktestResult("mock", Decimal("100000"))
        mock_result.final_capital = Decimal("100000")
        mock_result.equity_curve = [Decimal("100000")]

        with patch(
            "stockdownloader.backtest.combinatorial_tester.IntradayBacktestEngine"
        ) as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.run.return_value = mock_result

            tester = CombinatorialTester(cfg, data)
            results = tester.run()

        # 1 combo * 1 mode * 1 weight * 1 tf * 2 buy * 2 sell = 4
        assert len(results) == 4
