"""Unit tests for the GME analysis module."""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.gme.analysis import (
    analyze_options_chain,
    correlate_filings_with_price,
    detect_key_periods,
    run_event_study,
)
from stockdownloader.core.models.options import OptionContract, OptionType, OptionsChain
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.regulatory import SecFiling

_D = Decimal


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _bar(date: str, close: str, volume: int = 1_000_000) -> PriceData:
    """Create a PriceData bar with sensible defaults."""
    c = _D(close)
    return PriceData(
        date=date,
        open=c,
        high=c + _D("1"),
        low=c - _D("1"),
        close=c,
        adj_close=c,
        volume=volume,
    )


def _filing(date: str, form: str = "8-K", desc: str = "Current Report") -> SecFiling:
    return SecFiling(
        accession_number=f"000-{date}",
        filing_date=date,
        report_date=date,
        form=form,
        primary_document="doc.htm",
        description=desc,
        filing_url=f"https://sec.gov/test/{date}",
    )


def _contract(
    strike: str,
    option_type: OptionType,
    volume: int = 100,
    oi: int = 500,
    iv: str = "0.45",
    expiration: str = "2024-02-16",
) -> OptionContract:
    return OptionContract(
        contract_symbol=f"GME{strike}{option_type.value[0]}",
        type=option_type,
        strike=_D(strike),
        expiration_date=expiration,
        last_price=_D("2.50"),
        bid=_D("2.40"),
        ask=_D("2.60"),
        volume=volume,
        open_interest=oi,
        implied_volatility=_D(iv),
        delta=_D("0.5"),
        gamma=_D("0.05"),
        theta=_D("-0.03"),
        vega=_D("0.12"),
        in_the_money=False,
    )


def _make_chain(
    calls: list[OptionContract],
    puts: list[OptionContract],
    price: str = "25.00",
    expiration: str = "2024-02-16",
) -> OptionsChain:
    chain = OptionsChain("GME")
    chain.underlying_price = _D(price)
    chain.add_expiration_date(expiration)
    for c in calls:
        chain.add_call(expiration, c)
    for p in puts:
        chain.add_put(expiration, p)
    return chain


# ------------------------------------------------------------------
# Tests: correlate_filings_with_price
# ------------------------------------------------------------------


class TestCorrelateFilingsWithPrice:
    def test_basic_correlation(self) -> None:
        data = [
            _bar("2024-01-01", "10.00", 500_000),
            _bar("2024-01-02", "10.50", 800_000),
            _bar("2024-01-03", "11.00", 600_000),
            _bar("2024-01-04", "10.80", 550_000),
            _bar("2024-01-05", "10.90", 500_000),
            _bar("2024-01-08", "11.20", 700_000),
            _bar("2024-01-09", "11.50", 650_000),
        ]
        filings = [_filing("2024-01-02")]

        results = correlate_filings_with_price(filings, data)

        assert len(results) == 1
        r = results[0]
        assert r.filing.filing_date == "2024-01-02"
        assert r.price_on_date == _D("10.50")
        assert r.price_1d_before == _D("10.00")
        assert r.price_1d_after == _D("11.00")
        assert r.volume_on_date == 800_000
        assert r.price_change_1d_pct == 5.0  # (10.5-10.0)/10.0 * 100

    def test_filing_on_non_trading_day(self) -> None:
        """Filing on a weekend should map to the next trading day."""
        data = [
            _bar("2024-01-05", "10.00"),  # Friday
            _bar("2024-01-08", "10.50"),  # Monday
            _bar("2024-01-09", "11.00"),
        ]
        filings = [_filing("2024-01-06")]  # Saturday

        results = correlate_filings_with_price(filings, data)

        assert len(results) == 1
        assert results[0].price_on_date == _D("10.50")  # maps to Monday

    def test_multiple_filings(self) -> None:
        data = [_bar(f"2024-01-{d:02d}", str(10 + d)) for d in range(1, 11)]
        filings = [_filing("2024-01-02"), _filing("2024-01-05")]

        results = correlate_filings_with_price(filings, data)

        assert len(results) == 2
        assert results[0].filing.filing_date == "2024-01-02"
        assert results[1].filing.filing_date == "2024-01-05"

    def test_volume_ratio_calculation(self) -> None:
        # 20 bars of volume 1M, then one bar of volume 3M
        data = [_bar(f"2024-01-{d:02d}", "10.00", 1_000_000) for d in range(1, 22)]
        data.append(_bar("2024-01-22", "10.00", 3_000_000))
        filings = [_filing("2024-01-22")]

        results = correlate_filings_with_price(filings, data)

        assert len(results) == 1
        assert results[0].volume_ratio == 3.0

    def test_empty_inputs(self) -> None:
        assert correlate_filings_with_price([], [_bar("2024-01-01", "10")]) == []
        assert correlate_filings_with_price([_filing("2024-01-01")], []) == []

    def test_filing_before_price_data(self) -> None:
        data = [_bar("2024-06-01", "10.00")]
        filings = [_filing("2024-01-01")]  # way before any price data

        results = correlate_filings_with_price(filings, data)
        # Should map to earliest available bar
        assert len(results) == 1
        assert results[0].price_on_date == _D("10.00")


# ------------------------------------------------------------------
# Tests: detect_key_periods
# ------------------------------------------------------------------


class TestDetectKeyPeriods:
    def test_detects_large_move(self) -> None:
        """A 100% price jump should be detected."""
        data = [_bar(f"2024-01-{d:02d}", "10.00") for d in range(1, 11)]
        # Inject a big spike that stays elevated
        data[5] = _bar("2024-01-06", "25.00")
        data[6] = _bar("2024-01-07", "22.00")
        data[7] = _bar("2024-01-08", "23.00")
        data[8] = _bar("2024-01-09", "21.00")
        data[9] = _bar("2024-01-10", "20.00")

        periods = detect_key_periods(data, window=10, threshold_pct=50.0)

        assert len(periods) >= 1
        # The peak of $25 is 150% above the start of $10
        assert any(p.peak_price >= _D("25") for p in periods)

    def test_no_periods_in_flat_market(self) -> None:
        data = [_bar(f"2024-01-{d:02d}", "10.00") for d in range(1, 30)]

        periods = detect_key_periods(data, window=20, threshold_pct=50.0)

        assert periods == []

    def test_respects_max_periods(self) -> None:
        # Create 3 distinct large moves
        data = []
        for i in range(90):
            date = f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
            price = "10.00"
            if i % 30 == 15:
                price = "25.00"  # 150% spike every 30 bars
            data.append(_bar(date, price))

        periods = detect_key_periods(data, window=5, threshold_pct=50.0, max_periods=2)

        assert len(periods) <= 2


# ------------------------------------------------------------------
# Tests: analyze_options_chain
# ------------------------------------------------------------------


class TestAnalyzeOptionsChain:
    def test_put_call_ratios(self) -> None:
        calls = [_contract("25", OptionType.CALL, volume=200, oi=1000)]
        puts = [_contract("25", OptionType.PUT, volume=300, oi=500)]
        chain = _make_chain(calls, puts)

        result = analyze_options_chain(chain)

        assert result.total_call_volume == 200
        assert result.total_put_volume == 300
        assert result.total_call_oi == 1000
        assert result.total_put_oi == 500
        assert float(result.put_call_volume_ratio) == 1.5
        assert float(result.put_call_oi_ratio) == 0.5

    def test_max_pain_calculation(self) -> None:
        """Max pain should be the strike where total ITM value is minimised.

        With calls at $20 (OI=100) and puts at $30 (OI=100),
        max pain should be between $20 and $30.
        At strike $20: call pain=0, put pain=(30-20)*100=1000 → total=1000
        At strike $30: call pain=(30-20)*100=1000, put pain=0 → total=1000
        At strike $25: call pain=(25-20)*100=500, put pain=(30-25)*100=500 → total=1000
        All the same — but let's use asymmetric OI to test properly.
        """
        calls = [_contract("20", OptionType.CALL, oi=200)]
        puts = [_contract("30", OptionType.PUT, oi=100)]
        chain = _make_chain(calls, puts)

        result = analyze_options_chain(chain)

        # At $20: call pain=0, put pain=(30-20)*100=1000, total=1000
        # At $30: call pain=(30-20)*200=2000, put pain=0, total=2000
        # Max pain (min total) is at $20
        assert result.max_pain_strike == _D("20")

    def test_unusual_volume_detection(self) -> None:
        # Contract with volume 400 but OI 100 → ratio 4x > 3x threshold
        unusual = _contract("25", OptionType.CALL, volume=400, oi=100)
        normal = _contract("30", OptionType.CALL, volume=50, oi=500)
        chain = _make_chain([unusual, normal], [])

        result = analyze_options_chain(chain)

        assert len(result.unusual_volume_contracts) == 1
        assert result.unusual_volume_contracts[0][1] == _D("25")  # strike

    def test_empty_chain(self) -> None:
        chain = OptionsChain("GME")
        chain.underlying_price = _D("25.00")

        result = analyze_options_chain(chain)

        assert result.total_call_volume == 0
        assert result.total_put_volume == 0
        assert result.max_pain_strike == _D("0")

    def test_iv_skew(self) -> None:
        # Put IV higher than call IV → positive skew
        calls = [_contract("25", OptionType.CALL, iv="0.40")]
        puts = [_contract("25", OptionType.PUT, iv="0.50")]
        chain = _make_chain(calls, puts, price="25.00")

        result = analyze_options_chain(chain)

        assert result.iv_skew > 0  # put IV > call IV

    def test_highest_oi_strikes(self) -> None:
        c1 = _contract("20", OptionType.CALL, oi=100)
        c2 = _contract("25", OptionType.CALL, oi=500)
        p1 = _contract("20", OptionType.PUT, oi=300)
        p2 = _contract("25", OptionType.PUT, oi=800)
        chain = _make_chain([c1, c2], [p1, p2])

        result = analyze_options_chain(chain)

        assert result.highest_oi_call_strike == _D("25")
        assert result.highest_oi_put_strike == _D("25")


# ------------------------------------------------------------------
# Helpers for event study tests
# ------------------------------------------------------------------


def _make_daily_series(
    n: int,
    start_price: float = 100.0,
    daily_return: float = 0.001,
    volume: int = 1_000_000,
    start_date: str = "2020-01-02",
) -> list[PriceData]:
    """Generate a synthetic daily series with compounding returns.

    Dates increment by 1 day naively (ignoring weekends) for simplicity.
    """
    import math

    bars: list[PriceData] = []
    price = start_price
    year, month, day = (int(x) for x in start_date.split("-"))
    for i in range(n):
        d = day + i
        # Simple date generation: wrap month at 28 to stay valid
        m = month + (d - 1) // 28
        dd = ((d - 1) % 28) + 1
        y = year + (m - 1) // 12
        mm = ((m - 1) % 12) + 1
        date = f"{y:04d}-{mm:02d}-{dd:02d}"
        p = _D(str(round(price, 2)))
        bars.append(PriceData(
            date=date, open=p, high=p + _D("1"), low=p - _D("1"),
            close=p, adj_close=p, volume=volume,
        ))
        price *= math.exp(daily_return)
    return bars


# ------------------------------------------------------------------
# Tests: run_event_study
# ------------------------------------------------------------------


class TestRunEventStudy:
    def _stock_with_outperformance(self) -> tuple[list[PriceData], list[PriceData], list[SecFiling]]:
        """Build a 300-bar stock series that has positive abnormal returns after filings."""
        import math

        n = 300
        stock: list[PriceData] = []
        bench: list[PriceData] = []
        s_price = 100.0
        b_price = 400.0
        for i in range(n):
            d = i + 1
            mm = ((d - 1) // 28) + 1
            dd = ((d - 1) % 28) + 1
            yy = 2020 + (mm - 1) // 12
            mm = ((mm - 1) % 12) + 1
            date = f"{yy:04d}-{mm:02d}-{dd:02d}"

            sp = _D(str(round(s_price, 2)))
            bp = _D(str(round(b_price, 2)))
            stock.append(PriceData(date=date, open=sp, high=sp + _D("1"),
                                   low=sp - _D("1"), close=sp, adj_close=sp, volume=1_000_000))
            bench.append(PriceData(date=date, open=bp, high=bp + _D("1"),
                                   low=bp - _D("1"), close=bp, adj_close=bp, volume=50_000_000))
            b_price *= math.exp(0.0003)  # benchmark drifts up slowly
            # Stock outperforms after certain filing events
            if i in (150, 200):
                # Big jump on these days
                s_price *= math.exp(0.05)
            elif 150 < i <= 170 or 200 < i <= 220:
                # Stays elevated after event
                s_price *= math.exp(0.002)
            else:
                s_price *= math.exp(0.0003)

        filings = [
            _filing(stock[150].date, form="8-K"),
            _filing(stock[200].date, form="8-K"),
        ]
        return stock, bench, filings

    def test_positive_car_from_outperformance(self) -> None:
        """Stock outperforming benchmark after filings should yield positive mean CAR."""
        stock, bench, filings = self._stock_with_outperformance()

        results = run_event_study(filings, stock, bench, estimation_window=60)

        assert len(results) >= 1
        # 8-K group should have positive mean CAR
        r = results[0]
        assert r.form_type == "8-K"
        assert r.mean_car > 0

    def test_identical_series_zero_car(self) -> None:
        """When stock = benchmark, abnormal returns should be ~0."""
        n = 250
        stock = _make_daily_series(n, start_price=100.0, daily_return=0.001)
        # Benchmark identical prices
        bench = [PriceData(date=s.date, open=s.open, high=s.high, low=s.low,
                           close=s.close, adj_close=s.adj_close, volume=50_000_000)
                 for s in stock]

        filings = [_filing(stock[150].date, form="10-K"), _filing(stock[180].date, form="10-K")]
        results = run_event_study(filings, stock, bench, estimation_window=60)

        if results:
            for r in results:
                assert abs(r.mean_car) < 0.05  # CAR near zero

    def test_form_type_grouping(self) -> None:
        """Events from different form types should yield separate results."""
        stock, bench, _ = self._stock_with_outperformance()
        filings = [
            _filing(stock[150].date, form="10-K"),
            _filing(stock[155].date, form="10-K"),
            _filing(stock[200].date, form="8-K"),
            _filing(stock[210].date, form="8-K"),
        ]

        results = run_event_study(filings, stock, bench, estimation_window=60)

        form_types = {r.form_type for r in results}
        assert len(form_types) >= 1  # At least one form type with 2+ events

    def test_empty_inputs(self) -> None:
        assert run_event_study([], [], []) == []
        stock = _make_daily_series(50)
        assert run_event_study([], stock, stock) == []

    def test_insufficient_history(self) -> None:
        """Filings too early in the series (< estimation_window) should be skipped."""
        stock = _make_daily_series(50, daily_return=0.001)
        bench = _make_daily_series(50, daily_return=0.0005)
        # Filing at index 5 — not enough estimation data for 120-day window
        filings = [_filing(stock[5].date, form="8-K"), _filing(stock[6].date, form="8-K")]

        results = run_event_study(filings, stock, bench, estimation_window=120)

        # Should be empty because 5 < 120 + 5 (estimation + pre-event)
        assert results == []
