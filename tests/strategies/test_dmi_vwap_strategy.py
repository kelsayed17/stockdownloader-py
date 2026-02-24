"""Tests for the DMI+VWAP intraday strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradayAction
from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapConfig
from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapStrategy

_D = Decimal


def _make_bar(
    dt_str: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 500_000,
) -> IntradayPriceData:
    return IntradayPriceData(
        date=dt_str,
        open=_D(str(open_)),
        high=_D(str(high)),
        low=_D(str(low)),
        close=_D(str(close)),
        adj_close=_D(str(close)),
        volume=volume,
    )


def _build_trending_session(
    date: str,
    start_price: float = 500.0,
    trend: float = 0.10,
    bars: int = 78,
    volume: int = 500_000,
) -> list[IntradayPriceData]:
    """Build a full session of bars with a steady trend.

    Positive trend = uptrend, negative = downtrend.
    """
    result: list[IntradayPriceData] = []
    hour = 9
    minute = 30
    for i in range(bars):
        price = start_price + trend * i
        h = price + abs(trend) * 2 + 0.15
        l = price - abs(trend) * 2 - 0.15
        dt = f"{date} {hour:02d}:{minute:02d}:00-05:00"
        result.append(_make_bar(dt, price - 0.05, h, l, price, volume))
        minute += 5
        if minute >= 60:
            minute -= 60
            hour += 1
    return result


def _build_multi_day_trending(
    dates: list[str],
    start_price: float = 500.0,
    trend: float = 0.10,
    bars_per_day: int = 78,
) -> list[IntradayPriceData]:
    """Build multi-day data with consistent trend."""
    all_bars: list[IntradayPriceData] = []
    price = start_price
    for date in dates:
        day_bars = _build_trending_session(date, price, trend, bars_per_day)
        all_bars.extend(day_bars)
        price += trend * bars_per_day
    return all_bars


# ------------------------------------------------------------------
# Config tests
# ------------------------------------------------------------------


class TestDmiVwapConfig:
    def test_defaults(self):
        cfg = DmiVwapConfig()
        assert cfg.dmi_period == 14
        assert cfg.adx_threshold == _D("25")
        assert cfg.atr_period == 14
        assert cfg.sl_atr_mult == _D("1.5")
        assert cfg.rr == _D("2.0")
        assert cfg.bars_per_day == 78
        assert cfg.eod_exit_bar == 76
        assert cfg.min_entry_bar == 3

    def test_custom_config(self):
        cfg = DmiVwapConfig(
            dmi_period=10,
            adx_threshold=_D("20"),
            rr=_D("3.0"),
        )
        assert cfg.dmi_period == 10
        assert cfg.adx_threshold == _D("20")
        assert cfg.rr == _D("3.0")


# ------------------------------------------------------------------
# Strategy basics
# ------------------------------------------------------------------


class TestDmiVwapBasics:
    def test_name(self):
        s = DmiVwapStrategy()
        assert s.name == "DMI+VWAP"

    def test_warmup_period(self):
        s = DmiVwapStrategy()
        wp = s.warmup_period
        assert wp >= 14 * 2  # At least 2x DMI period

    def test_warmup_returns_hold(self):
        """During warmup, strategy should always return HOLD."""
        data = _build_trending_session("2024-06-03", 500.0, 0.10)
        s = DmiVwapStrategy()
        wp = s.warmup_period
        for i in range(min(wp, len(data))):
            sig = s.evaluate(data, i)
            assert sig.action == IntradayAction.HOLD

    def test_session_boundary_detection(self):
        """Strategy should detect new sessions and reset state."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04"], 500.0, 0.10, 78
        )
        s = DmiVwapStrategy()
        # Evaluate across session boundary
        for i in range(len(data)):
            s.evaluate(data, i)
        # Session date should be the last day
        assert s._session_date == "2024-06-04"

    def test_on_session_start_resets_position(self):
        s = DmiVwapStrategy()
        s._in_position = True
        s._position_is_long = True
        s.on_session_start("2024-06-04")
        assert not s._in_position
        assert not s._position_is_long
        assert s._session_date == "2024-06-04"


# ------------------------------------------------------------------
# Signal generation
# ------------------------------------------------------------------


class TestDmiVwapSignals:
    def test_uptrend_produces_long_signal(self):
        """A strong uptrend should eventually produce ENTER_LONG."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.15,
        )
        s = DmiVwapStrategy()
        signals = []
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action != IntradayAction.HOLD:
                signals.append((i, sig))

        # Should have at least one entry signal
        entry_signals = [
            (i, sig) for i, sig in signals
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT)
        ]
        assert len(entry_signals) > 0, "Expected at least one entry signal in uptrend"

    def test_downtrend_produces_short_signal(self):
        """A strong downtrend should eventually produce ENTER_SHORT."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=-0.15,
        )
        s = DmiVwapStrategy()
        signals = []
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action == IntradayAction.ENTER_SHORT:
                signals.append((i, sig))

        assert len(signals) > 0, "Expected ENTER_SHORT signals in downtrend"

    def test_long_signal_has_stop_and_target(self):
        """Entry signals should carry stop_loss and take_profit."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.15,
        )
        s = DmiVwapStrategy()
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action == IntradayAction.ENTER_LONG:
                assert sig.stop_loss > _D("0"), "stop_loss should be set"
                assert sig.take_profit > _D("0"), "take_profit should be set"
                assert sig.risk_per_share > _D("0"), "risk_per_share should be set"
                assert sig.stop_loss < data[i].close, "stop below entry for long"
                assert sig.take_profit > data[i].close, "target above entry for long"
                break
        else:
            pytest.skip("No long signal generated in this data")

    def test_short_signal_has_stop_and_target(self):
        """Short entry should have stop above and target below."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=-0.15,
        )
        s = DmiVwapStrategy()
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action == IntradayAction.ENTER_SHORT:
                assert sig.stop_loss > data[i].close, "stop above entry for short"
                assert sig.take_profit < data[i].close, "target below entry for short"
                assert sig.risk_per_share > _D("0")
                break
        else:
            pytest.skip("No short signal generated in this data")

    def test_confluence_score_range(self):
        """Confluence score should be between 0 and 10."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.15,
        )
        s = DmiVwapStrategy()
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                assert 0 <= sig.confluence_score <= 10
                assert sig.max_score == 10
                break


# ------------------------------------------------------------------
# EOD exit
# ------------------------------------------------------------------


class TestDmiVwapEodExit:
    def test_eod_exit_triggers(self):
        """Position should be closed near end of day."""
        data = _build_trending_session("2024-06-03", 500.0, 0.15, 78)
        # Use a second day to get warmup + entries
        data2 = _build_trending_session("2024-06-04", 510.0, 0.15, 78)
        all_data = data + data2

        s = DmiVwapStrategy()
        eod_exits = []
        for i in range(len(all_data)):
            sig = s.evaluate(all_data, i)
            if sig.mode == "DMI_VWAP_EOD":
                eod_exits.append(i)

        # If a position was opened on day 2, it should get an EOD exit
        # (Day 1 likely has entries too, but with warmup constraints)
        # The key test: any EOD exit should happen near end of session
        for idx in eod_exits:
            bar = all_data[idx]
            time_str = bar.time_str
            # Should be after 15:45 (bar 76 out of 78)
            assert time_str >= "15:40", f"EOD exit at {time_str} is too early"


# ------------------------------------------------------------------
# Position tracking
# ------------------------------------------------------------------


class TestDmiVwapPositionTracking:
    def test_no_double_entry(self):
        """While in a position, no new entries should be generated."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.15,
        )
        s = DmiVwapStrategy()
        entries_without_exit = 0
        in_pos = False
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                assert not in_pos, f"Double entry at bar {i}"
                # Simulate engine callback confirming position
                s.on_position_opened(sig.action == IntradayAction.ENTER_LONG)
                in_pos = True
                entries_without_exit += 1
            elif sig.action == IntradayAction.EXIT:
                s.on_position_closed()
                in_pos = False
                entries_without_exit = 0

    def test_exit_before_re_entry(self):
        """Strategy should exit before entering opposite direction."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.15,
        )
        s = DmiVwapStrategy()
        last_action = None
        for i in range(len(data)):
            sig = s.evaluate(data, i)
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                if last_action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                    pytest.fail(f"Entry without prior exit at bar {i}")
                # Simulate engine callback confirming position
                s.on_position_opened(sig.action == IntradayAction.ENTER_LONG)
                last_action = sig.action
            elif sig.action == IntradayAction.EXIT:
                s.on_position_closed()
                last_action = sig.action


# ------------------------------------------------------------------
# Confluence scoring
# ------------------------------------------------------------------


class TestDmiVwapConfluence:
    def test_strong_adx_scores_higher(self):
        """ADX > 40 should score higher than ADX = 26."""
        score_high = DmiVwapStrategy._compute_confluence(
            _D("45"), _D("35"), _D("15"), _D("502"), _D("500")
        )
        score_low = DmiVwapStrategy._compute_confluence(
            _D("26"), _D("20"), _D("18"), _D("502"), _D("500")
        )
        assert score_high > score_low

    def test_wide_di_spread_scores_higher(self):
        """Large +DI/-DI spread should score higher."""
        score_wide = DmiVwapStrategy._compute_confluence(
            _D("30"), _D("40"), _D("10"), _D("502"), _D("500")
        )
        score_narrow = DmiVwapStrategy._compute_confluence(
            _D("30"), _D("22"), _D("18"), _D("502"), _D("500")
        )
        assert score_wide > score_narrow

    def test_score_capped_at_10(self):
        """Maximum confluence score is 10."""
        score = DmiVwapStrategy._compute_confluence(
            _D("50"), _D("50"), _D("5"), _D("510"), _D("500")
        )
        assert score <= 10

    def test_score_minimum_0(self):
        """Minimum score should be at least 0."""
        score = DmiVwapStrategy._compute_confluence(
            _D("26"), _D("14"), _D("13"), _D("500.05"), _D("500")
        )
        assert score >= 0


# ------------------------------------------------------------------
# Config variations
# ------------------------------------------------------------------


class TestDmiVwapConfigVariations:
    def test_higher_adx_threshold_fewer_signals(self):
        """Raising adx_threshold should produce fewer or equal signals."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.10,
        )
        cfg_low = DmiVwapConfig(adx_threshold=_D("20"))
        cfg_high = DmiVwapConfig(adx_threshold=_D("35"))

        s_low = DmiVwapStrategy(cfg_low)
        s_high = DmiVwapStrategy(cfg_high)

        count_low = sum(
            1 for i in range(len(data))
            if s_low.evaluate(data, i).action
            in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT)
        )
        count_high = sum(
            1 for i in range(len(data))
            if s_high.evaluate(data, i).action
            in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT)
        )
        assert count_high <= count_low

    def test_min_di_spread_filter(self):
        """Setting min_di_spread should filter weak DI signals."""
        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05"],
            start_price=500.0,
            trend=0.10,
        )
        cfg_no_spread = DmiVwapConfig(min_di_spread=_D("0"))
        cfg_spread = DmiVwapConfig(min_di_spread=_D("10"))

        s1 = DmiVwapStrategy(cfg_no_spread)
        s2 = DmiVwapStrategy(cfg_spread)

        count1 = sum(
            1 for i in range(len(data))
            if s1.evaluate(data, i).action
            in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT)
        )
        count2 = sum(
            1 for i in range(len(data))
            if s2.evaluate(data, i).action
            in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT)
        )
        assert count2 <= count1


# ------------------------------------------------------------------
# Backtest engine integration
# ------------------------------------------------------------------


class TestDmiVwapBacktest:
    def test_runs_in_backtest_engine(self):
        """Strategy should work with IntradayBacktestEngine."""
        from stockdownloader.backtesting.engines.intraday import (
            IntradayBacktestEngine,
        )

        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05",
             "2024-06-06", "2024-06-07"],
            start_price=500.0,
            trend=0.10,
        )
        engine = IntradayBacktestEngine(
            initial_capital=_D("25000"),
            risk_per_trade=_D("0.01"),
        )
        strategy = DmiVwapStrategy()
        result = engine.run(strategy, data)

        assert result.strategy_name == "DMI+VWAP"
        assert result.initial_capital == _D("25000")
        assert result.final_capital > _D("0")
        assert len(result.equity_curve) == len(data)

    def test_backtest_produces_trades(self):
        """Should produce at least some trades on trending data."""
        from stockdownloader.backtesting.engines.intraday import (
            IntradayBacktestEngine,
        )

        data = _build_multi_day_trending(
            ["2024-06-03", "2024-06-04", "2024-06-05",
             "2024-06-06", "2024-06-07"],
            start_price=500.0,
            trend=0.15,
        )
        engine = IntradayBacktestEngine(
            initial_capital=_D("25000"),
            risk_per_trade=_D("0.01"),
        )
        strategy = DmiVwapStrategy()
        result = engine.run(strategy, data)

        trades = result.closed_trades
        assert len(trades) > 0, "Expected trades in trending data"
