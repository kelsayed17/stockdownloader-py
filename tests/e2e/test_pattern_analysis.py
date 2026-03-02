"""End-to-end test for the pattern analysis pipeline that mirrors TrendAnalysisApp.
Uses real SPY price data to generate price movement patterns for multiple
simulated tickers, then runs the full PatternAnalyzer pipeline.

Data flow: PriceData -> pattern generation (like YahooHistoricalClient)
  -> defaultdict(set) -> analyze() -> list[PatternResult]
  -> print_results()

This exercises: PriceData, HistoricalData, PatternResult, PatternAnalyzer,
BigDecimalMath, and the pattern storage (defaultdict(set)).
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
from math import copysign

import pytest

from stockdownloader.analysis.pattern_analyzer import analyze, print_results
from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.alert import PatternResult
from stockdownloader.core.models.market_data import HistoricalData
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.math import divide, scale2

PATTERN_DAYS = 7


def _generate_patterns(spy_data: list[PriceData], ticker: str, start_idx: int) -> HistoricalData:
    """Generates patterns for a ticker starting at the given index.
    Mirrors YahooHistoricalClient.parsePatterns logic.
    """
    data = HistoricalData(ticker)
    up_down_list: list[int] = []

    for i in range(start_idx + 1, min(start_idx + PATTERN_DAYS + 1, len(spy_data))):
        close_price = spy_data[i].close
        prev_close = spy_data[i - 1].close

        close_change = (
            (close_price - prev_close)
            / prev_close
            * Decimal("100")
        ).quantize(Decimal("1E+0"), rounding=ROUND_CEILING)

        # signum
        if close_change > 0:
            up_down_list.append(1)
        elif close_change < 0:
            up_down_list.append(-1)
        else:
            up_down_list.append(0)

        data.patterns[str(up_down_list)].add(ticker)

    return data


@pytest.fixture(scope="module")
def spy_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0
    return data


@pytest.fixture(scope="module")
def all_patterns(spy_data):
    patterns: defaultdict[str, set[str]] = defaultdict(set)

    # Simulate what TrendAnalysisApp does: for each ticker, generate 7-day patterns
    # We use different segments of real SPY data as if they were different tickers
    simulated_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "V", "UNH",
    ]

    for t, ticker in enumerate(simulated_tickers):
        start_idx = t * 20  # Offset each ticker's data window
        if start_idx + PATTERN_DAYS + 1 >= len(spy_data):
            break

        hd = _generate_patterns(spy_data, ticker, start_idx)
        for key, values in hd.patterns.items():
            patterns[key] |= values

    # Also generate overlapping patterns to create pattern frequency
    start_idx = 0
    while start_idx + PATTERN_DAYS + 1 < len(spy_data):
        synthetic_ticker = f"SYN{start_idx}"
        hd = _generate_patterns(spy_data, synthetic_ticker, start_idx)
        for key, values in hd.patterns.items():
            patterns[key] |= values
        start_idx += 5

    total_entries = sum(len(v) for v in patterns.values())
    assert total_entries > 0, "Should have generated patterns"
    return patterns


@pytest.fixture(scope="module")
def analysis_results(all_patterns):
    results = analyze(all_patterns)
    return results


# ========== Pattern Generation ==========


def test_patterns_generated_from_real_data(all_patterns):
    assert len(all_patterns) > 0, "Should have multiple distinct pattern keys"
    total_entries = sum(len(v) for v in all_patterns.values())
    assert total_entries > 10, "Should have many pattern-ticker entries"


def test_pattern_keys_have_expected_format(all_patterns):
    for key in all_patterns:
        assert key.startswith("["), "Pattern key should start with ["
        assert key.endswith("]"), "Pattern key should end with ]"
        # Pattern values should be -1, 0, or 1
        inner = key[1:-1]
        for val in inner.split(", "):
            v = int(val.strip())
            assert -1 <= v <= 1, f"Pattern value should be -1, 0, or 1, got {v}"


def test_patterns_associated_with_correct_tickers(all_patterns):
    # Verify that synthetic tickers are associated with patterns
    found_synthetic = False
    for key in all_patterns:
        tickers = all_patterns[key]
        for ticker in tickers:
            if ticker.startswith("SYN"):
                found_synthetic = True
                break
        if found_synthetic:
            break
    assert found_synthetic, "Should find synthetic tickers in patterns"


# ========== PatternAnalyzer Results ==========


def test_analyzer_produces_results(analysis_results):
    assert analysis_results is not None
    assert len(analysis_results) > 0, "Analysis should produce results"


def test_results_are_sorted_by_frequency_descending(analysis_results):
    prev_freq = None
    for result in analysis_results:
        if prev_freq is not None:
            assert result.pattern_freq <= prev_freq, \
                "Results should be sorted by frequency descending"
        prev_freq = result.pattern_freq


def test_pattern_result_fields_populated(analysis_results):
    for result in analysis_results:
        assert result.pattern is not None, "Pattern should not be None"
        assert result.similar is not None, "Similar pattern should not be None"
        assert result.offset is not None, "Offset pattern should not be None"
        assert result.pattern_freq is not None, "Pattern frequency should not be None"
        assert result.similar_freq is not None, "Similar frequency should not be None"
        assert result.offset_freq is not None, "Offset frequency should not be None"
        assert result.accuracy is not None, "Accuracy should not be None"
        assert result.pattern_symbols is not None, "Pattern symbols should not be None"


def test_similar_pattern_is_inverse(analysis_results):
    for result in analysis_results:
        pattern = result.pattern
        similar = result.similar

        # Parse the first element and check it's inverted
        pattern_inner = pattern[1:-1]
        similar_inner = similar[1:-1]

        pattern_vals = pattern_inner.split(", ")
        similar_vals = similar_inner.split(", ")

        assert len(pattern_vals) == len(similar_vals), \
            "Pattern and similar should have same length"

        # First element should be inverted
        first = int(pattern_vals[0].strip())
        similar_first = int(similar_vals[0].strip())
        if first == 1:
            assert similar_first == -1
        elif first == -1:
            assert similar_first == 1
        else:
            assert similar_first == 0

        # Remaining elements should be the same
        for i in range(1, len(pattern_vals)):
            assert pattern_vals[i].strip() == similar_vals[i].strip(), \
                "Non-first elements should be identical between pattern and similar"


def test_offset_pattern_removes_first_element(analysis_results):
    for result in analysis_results:
        pattern = result.pattern
        offset = result.offset

        pattern_inner = pattern[1:-1]
        pattern_vals = pattern_inner.split(", ")

        # The offset is derived by the PatternAnalyzer via list slicing [1:]
        # For a pattern like [1, -1, 0], the offset should be [-1, 0]
        if len(pattern_vals) > 1:
            offset_inner = offset[1:-1]
            offset_vals = offset_inner.split(", ")

            assert len(offset_vals) == len(pattern_vals) - 1, \
                f"Offset pattern should have one fewer element for pattern {pattern}"

            # Offset values should match the SIMILAR pattern's [1..n]
            similar_inner = result.similar[1:-1]
            similar_vals = similar_inner.split(", ")

            for i in range(len(offset_vals)):
                assert similar_vals[i + 1].strip() == offset_vals[i].strip(), \
                    "Offset should match similar pattern shifted by one"


def test_accuracy_percentage_bounds(analysis_results):
    for result in analysis_results:
        accuracy = result.accuracy
        assert accuracy >= Decimal("0"), "Accuracy should be >= 0%"
        assert accuracy <= Decimal("100"), "Accuracy should be <= 100%"


def test_accuracy_formula_verification(analysis_results):
    for result in analysis_results:
        pf = result.pattern_freq
        sf = result.similar_freq
        total = pf + sf

        if total > Decimal("0"):
            expected_accuracy = scale2(divide(pf * Decimal("100"), total))
            assert expected_accuracy == result.accuracy, \
                "Accuracy should be (patternFreq * 100) / (patternFreq + similarFreq)"


def test_pattern_symbols_are_non_empty(analysis_results):
    for result in analysis_results:
        assert len(result.pattern_symbols) > 0, \
            "Pattern should have at least one associated symbol"
        # Frequency should match number of symbols
        assert len(result.pattern_symbols) == int(result.pattern_freq), \
            "Pattern frequency should match number of symbols"


# ========== Report Generation ==========


def test_print_results_generates_output(analysis_results):
    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        print_results(analysis_results)
    finally:
        sys.stdout = old_stdout

    output = capture.getvalue()
    assert "Top 10 patterns" in output, "Should contain header"
    assert "Top 10 patterns offset by one day" in output, \
        "Should contain offset header"
    assert "Frequency:" in output, "Should contain frequency column"
    assert "Percentage:" in output, "Should contain percentage column"
    assert "Pattern:" in output, "Should contain pattern column"
    assert "Stocks:" in output, "Should contain stocks column"


# ========== HistoricalData Integration ==========


def test_historical_data_pattern_storage(spy_data):
    hd = HistoricalData("TEST")
    hd.highest_price_this_qtr = Decimal("200.00")
    hd.lowest_price_this_qtr = Decimal("150.00")
    hd.highest_price_last_qtr = Decimal("195.00")
    hd.lowest_price_last_qtr = Decimal("145.00")

    # Generate patterns
    generated = _generate_patterns(spy_data, "TEST", 0)

    assert len(generated.patterns) > 0, "Generated patterns should not be empty"
    assert generated.ticker == "TEST"


def test_multiple_tickers_generate_distinct_patterns(spy_data):
    hd1 = _generate_patterns(spy_data, "TICK1", 0)
    hd2 = _generate_patterns(spy_data, "TICK2", 50)

    # Different starting points may generate different patterns
    combined: defaultdict[str, set[str]] = defaultdict(set)
    for key, values in hd1.patterns.items():
        combined[key] |= values
    for key, values in hd2.patterns.items():
        combined[key] |= values

    # Combined should have entries from both tickers
    found_tick1 = False
    found_tick2 = False
    for key in combined:
        if "TICK1" in combined[key]:
            found_tick1 = True
        if "TICK2" in combined[key]:
            found_tick2 = True
    assert found_tick1, "Should find TICK1 in combined patterns"
    assert found_tick2, "Should find TICK2 in combined patterns"


# ========== PatternResult Record Behavior ==========


def test_pattern_result_comparable_ordering():
    # PatternResult's __lt__ sorts by frequency descending
    result1 = PatternResult(
        "[1, 0]", "[-1, 0]", "[0]",
        Decimal("10"), Decimal("1"), Decimal("1"),
        Decimal("90.91"),
        frozenset({"A"}), frozenset({"B"}), frozenset({"C"}),
    )
    result2 = PatternResult(
        "[0, 1]", "[0, -1]", "[1]",
        Decimal("1"), Decimal("1"), Decimal("1"),
        Decimal("50.00"),
        frozenset({"D"}), frozenset({"E"}), frozenset({"F"}),
    )

    assert result1 < result2, \
        "Higher frequency should sort first (descending)"


# ========== Full Pipeline with All Pattern Lengths ==========


def test_patterns_grow_incrementally_per_bar(spy_data):
    # The YahooHistoricalClient adds to the upDownList incrementally:
    # After 1 bar: [1]
    # After 2 bars: [1, -1]
    # After 3 bars: [1, -1, 0]
    # etc.
    # This means patterns of length 1 through PATTERN_DAYS should exist
    hd = _generate_patterns(spy_data, "GROW", 10)

    found_short = False
    found_long = False
    for key in hd.patterns:
        inner = key[1:-1]
        elements = len(inner.split(", "))
        if elements == 1:
            found_short = True
        if elements >= 5:
            found_long = True
    assert found_short, "Should have short (1-element) patterns"
    assert found_long, "Should have longer patterns"
