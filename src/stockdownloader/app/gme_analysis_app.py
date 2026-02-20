"""Comprehensive GME (GameStop) analysis application.

Multi-stage pipeline:
  Stage 1 — Download 6-year daily price data from Yahoo Finance
  Stage 2 — Download recent intraday 5-min data (~60 days)
  Stage 3 — Download SEC EDGAR filings
  Stage 4 — Download current options chain snapshot
  Stage 5 — Run analysis (patterns, filing correlation, options)
  Stage 6 — Generate comprehensive terminal report

Usage::

    python -m stockdownloader.app.gme_analysis_app
    python -m stockdownloader.app.gme_analysis_app --skip-intraday --skip-options
    python -m stockdownloader.app.gme_analysis_app --log output/gme.log
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.app.app_helpers import add_log_arg
from stockdownloader.analysis.gme import (
    EventStudyResult,
    FilingImpact,
    KeyPeriod,
    OptionsAnalysis,
    PriceStatistics,
    ReturnDistribution,
    StructuralBreak,
    VolatilityRegime,
    VolumeProfile,
    analyze_options_chain,
    analyze_return_distribution,
    compute_price_statistics,
    compute_volume_profile,
    correlate_filings_with_price,
    detect_key_periods,
    detect_structural_breaks,
    detect_volatility_regimes,
    run_event_study,
)
from stockdownloader.analysis.signal_generator import generate_alert
from stockdownloader.data.sec_edgar_client import SecEdgarClient
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.data.yahoo_options_client import YahooOptionsClient
from stockdownloader.model.alert_result import AlertResult
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.options import OptionsChain
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.regulatory_records import SecFiling

logger = logging.getLogger(__name__)

_SYMBOL = "GME"
_GME_FORM_TYPES = ["10-K", "10-Q", "8-K", "SC 13D", "SC 13D/A", "DEF 14A", "DEFA14A", "4"]
_SIX_YEARS_DAYS = 365 * 6 + 1  # ~6 years in calendar days

# Unicode box-drawing characters
_TL = "\u2554"  # ╔
_TR = "\u2557"  # ╗
_BL = "\u255a"  # ╚
_BR = "\u255d"  # ╝
_H = "\u2550"   # ═
_V = "\u2551"   # ║


# ======================================================================
# Output helpers
# ======================================================================


class _TeeWriter:
    """Callable tee — delegates to :class:`~stockdownloader.util.file_helper.TeeWriter`.

    Provides ``out("text")`` callable interface used throughout this module.
    When no log path is given, writes to stdout only.
    """

    def __init__(self, log_path: str | None = None) -> None:
        from stockdownloader.util.io_helpers import TeeWriter

        self._file = open(log_path, "w") if log_path else None  # noqa: SIM115
        self._tee: TeeWriter | None = TeeWriter(self._file) if self._file else None

    def __call__(self, text: str = "") -> None:
        if self._tee is not None:
            self._tee.print(text)
        else:
            print(text)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _header(out: _TeeWriter, title: str) -> None:
    """Print a stage header."""
    out()
    out(_TL + _H * 76 + _TR)
    out(_V + f"  {title}".ljust(76) + _V)
    out(_BL + _H * 76 + _BR)
    out()


def _section(out: _TeeWriter, title: str) -> None:
    """Print a report section header."""
    out()
    out(f"  {'─' * 72}")
    out(f"  {title}")
    out(f"  {'─' * 72}")


def _fmt_price(val: Decimal) -> str:
    return f"${val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"


def _fmt_pct(val: float) -> str:
    return f"{val:+.2f}%"


def _fmt_volume(vol: int) -> str:
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.2f}B"
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.2f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return str(vol)


# ======================================================================
# Stage 1: Daily Price Data
# ======================================================================


def _stage_1_daily(out: _TeeWriter) -> tuple[list[PriceData], list[PriceData]]:
    _header(out, "STAGE 1: DOWNLOADING 6-YEAR DAILY PRICE DATA")

    client = YahooDataClient()

    # Use epoch-based fetch for precise daily data over 6 years.
    # The range-based API ("max") can return monthly bars for long ranges.
    now = datetime.now()
    start = now - timedelta(days=_SIX_YEARS_DAYS)
    start_epoch = int(start.timestamp())
    end_epoch = int(now.timestamp())

    out(f"  Fetching {_SYMBOL} daily data ({start.strftime('%Y-%m-%d')} to "
        f"{now.strftime('%Y-%m-%d')})...")
    data = client.fetch_price_data_by_epoch(_SYMBOL, start_epoch, end_epoch)

    if not data:
        out("  ERROR: Could not fetch daily data for GME.")
        return [], []

    out(f"  Loaded {len(data)} trading days")
    if data:
        out(f"  Date range: {data[0].date} to {data[-1].date}")
        out(f"  Current price: {_fmt_price(data[-1].close)}")

    # Fetch SPY benchmark data for event study
    out()
    out(f"  Fetching SPY benchmark data (same date range)...")
    spy_data = client.fetch_price_data_by_epoch("SPY", start_epoch, end_epoch)
    out(f"  Loaded {len(spy_data)} SPY trading days")

    return data, spy_data


# ======================================================================
# Stage 2: Intraday Data
# ======================================================================


def _stage_2_intraday(out: _TeeWriter) -> list[IntradayPriceData]:
    _header(out, "STAGE 2: DOWNLOADING RECENT INTRADAY DATA (5-MIN)")

    client = YahooDataClient()
    out(f"  Fetching {_SYMBOL} 5-min bars (~60 calendar days)...")
    data = client.fetch_intraday_history(_SYMBOL, total_days=60, interval="5m")

    out(f"  Loaded {len(data)} intraday bars")
    if data:
        out(f"  Date range: {data[0].date} to {data[-1].date}")
    out()
    out("  NOTE: Yahoo limits intraday data to ~60 days.")
    out("  For 6-year pattern analysis, daily candles are used (Stage 1).")
    return data


# ======================================================================
# Stage 3: SEC Filings
# ======================================================================


def _stage_3_sec_filings(out: _TeeWriter, user_agent: str) -> list[SecFiling]:
    _header(out, "STAGE 3: DOWNLOADING SEC EDGAR FILINGS")

    client = SecEdgarClient(user_agent=user_agent)
    out(f"  Fetching GameStop SEC filings (CIK 0001326380)...")
    out(f"  Form types: {', '.join(_GME_FORM_TYPES)}")
    filings = client.fetch_gme_filings(form_types=_GME_FORM_TYPES)

    out(f"  Downloaded {len(filings)} filings")

    # Count by form type
    form_counts: dict[str, int] = defaultdict(int)
    for f in filings:
        form_counts[f.form] += 1
    for form, count in sorted(form_counts.items(), key=lambda x: -x[1]):
        out(f"    {form:>12s}: {count}")

    return filings


# ======================================================================
# Stage 4: Options Chain
# ======================================================================


def _stage_4_options(out: _TeeWriter) -> OptionsChain | None:
    _header(out, "STAGE 4: DOWNLOADING CURRENT OPTIONS CHAIN")

    try:
        client = YahooOptionsClient()
        out(f"  Fetching {_SYMBOL} options chain...")
        chain = client.download(_SYMBOL)
        out(f"  Expirations: {len(chain.expiration_dates)}")
        out(f"  Total calls: {len(chain.all_calls)}")
        out(f"  Total puts:  {len(chain.all_puts)}")
        out(f"  Underlying:  {_fmt_price(chain.underlying_price)}")
        return chain
    except Exception as exc:
        out(f"  WARNING: Could not fetch options chain: {exc}")
        return None


# ======================================================================
# Stage 5: Analysis
# ======================================================================


def _stage_5_analysis(
    out: _TeeWriter,
    daily_data: list[PriceData],
    benchmark_data: list[PriceData],
    filings: list[SecFiling],
    chain: OptionsChain | None,
) -> tuple[
    PriceStatistics | None,
    list[FilingImpact],
    list[KeyPeriod],
    OptionsAnalysis | None,
    AlertResult | None,
    ReturnDistribution | None,
    VolatilityRegime | None,
    list[StructuralBreak],
    list[EventStudyResult],
    VolumeProfile | None,
]:
    _header(out, "STAGE 5: RUNNING ANALYSIS")

    stats: PriceStatistics | None = None
    filing_impacts: list[FilingImpact] = []
    key_periods: list[KeyPeriod] = []
    opts_analysis: OptionsAnalysis | None = None
    alert: AlertResult | None = None
    ret_dist: ReturnDistribution | None = None
    vol_regime: VolatilityRegime | None = None
    structural_breaks: list[StructuralBreak] = []
    event_study_results: list[EventStudyResult] = []
    volume_profile: VolumeProfile | None = None

    if daily_data:
        out("  Computing price statistics...")
        stats = compute_price_statistics(daily_data)

        out("  Detecting key periods (major price moves)...")
        key_periods = detect_key_periods(daily_data)
        out(f"    Found {len(key_periods)} key periods")

        out("  Analysing return distribution (skew, kurtosis, VaR, autocorrelation)...")
        ret_dist = analyze_return_distribution(daily_data)
        out(f"    Normality (JB p={ret_dist.jarque_bera_p:.4f}): "
            f"{'normal' if ret_dist.is_normal else 'non-normal'}")

        out("  Detecting volatility regimes...")
        vol_regime = detect_volatility_regimes(daily_data)
        regime_summary = ", ".join(f"{k}={v}" for k, v in sorted(vol_regime.regime_counts.items()))
        out(f"    Regimes: {regime_summary}")

        out("  Detecting structural breaks...")
        structural_breaks = detect_structural_breaks(daily_data)
        out(f"    Found {len(structural_breaks)} structural break clusters")

        out("  Computing volume profile...")
        volume_profile = compute_volume_profile(daily_data)
        out(f"    POC: ${volume_profile.poc_price:.2f}, "
            f"Value Area: ${volume_profile.value_area_low:.2f}–${volume_profile.value_area_high:.2f}")
        out(f"    Volume anomaly days: {len(volume_profile.anomaly_days)}")

        if filings:
            out("  Correlating filings with price action (legacy)...")
            filing_impacts = correlate_filings_with_price(filings, daily_data)
            out(f"    Correlated {len(filing_impacts)} filings with price data")

            if benchmark_data:
                out("  Running event study (Campbell-Lo-MacKinlay)...")
                event_study_results = run_event_study(filings, daily_data, benchmark_data)
                out(f"    Analysed {len(event_study_results)} form-type groups")

        out("  Generating technical signals...")
        try:
            alert = generate_alert(_SYMBOL, daily_data)
        except Exception as exc:
            out(f"    WARNING: Signal generation failed: {exc}")

    if chain is not None:
        out("  Analysing options chain...")
        opts_analysis = analyze_options_chain(chain)

    out("  Analysis complete.")
    return (stats, filing_impacts, key_periods, opts_analysis, alert,
            ret_dist, vol_regime, structural_breaks, event_study_results, volume_profile)


# ======================================================================
# Stage 6: Report
# ======================================================================


def _stage_6_report(
    out: _TeeWriter,
    daily_data: list[PriceData],
    intraday_data: list[IntradayPriceData],
    stats: PriceStatistics | None,
    filing_impacts: list[FilingImpact],
    key_periods: list[KeyPeriod],
    opts_analysis: OptionsAnalysis | None,
    alert: AlertResult | None,
    ret_dist: ReturnDistribution | None,
    vol_regime: VolatilityRegime | None,
    structural_breaks: list[StructuralBreak],
    event_study_results: list[EventStudyResult],
    volume_profile: VolumeProfile | None,
) -> None:
    _header(out, "COMPREHENSIVE GME ANALYSIS REPORT")

    section_num = 0

    # ------------------------------------------------------------------
    # Section: Price Statistics
    # ------------------------------------------------------------------
    if stats:
        section_num += 1
        _section(out, f"{section_num}. PRICE STATISTICS (6-YEAR OVERVIEW)")
        out()
        out(f"  Period:             {stats.period_start} to {stats.period_end}")
        out(f"  Trading Days:       {stats.trading_days:,}")
        out(f"  Start Price:        {_fmt_price(stats.start_price)}")
        out(f"  End Price:          {_fmt_price(stats.end_price)}")
        out(f"  Total Return:       {_fmt_pct(stats.total_return_pct)}")
        out()
        out(f"  All-Time High:      {_fmt_price(stats.all_time_high)}  ({stats.all_time_high_date})")
        out(f"  All-Time Low:       {_fmt_price(stats.all_time_low)}  ({stats.all_time_low_date})")
        out()
        out(f"  Avg Daily Volume:   {_fmt_volume(stats.avg_daily_volume)}")
        out(f"  Max Daily Volume:   {_fmt_volume(stats.max_daily_volume)}  ({stats.max_volume_date})")
        out(f"  Annual Volatility:  {stats.volatility_annual * 100:.1f}%")

    # ------------------------------------------------------------------
    # Section: Return Distribution
    # ------------------------------------------------------------------
    if ret_dist:
        section_num += 1
        _section(out, f"{section_num}. RETURN DISTRIBUTION ANALYSIS")
        out()
        out(f"  Daily Log Returns (n={stats.trading_days - 1 if stats else '?'}):")
        out(f"    Mean:                 {ret_dist.mean * 100:.4f}% per day")
        out(f"    Std Dev:              {ret_dist.std * 100:.4f}% per day")
        out(f"    Skewness:             {ret_dist.skewness:+.4f}"
            f"  {'(left tail)' if ret_dist.skewness < -0.5 else '(right tail)' if ret_dist.skewness > 0.5 else '(symmetric)'}")
        out(f"    Excess Kurtosis:      {ret_dist.kurtosis:+.4f}"
            f"  {'(fat tails)' if ret_dist.kurtosis > 1 else '(thin tails)' if ret_dist.kurtosis < -1 else '(near normal)'}")
        out()
        out(f"  Normality Test (Jarque-Bera):")
        out(f"    Statistic:            {ret_dist.jarque_bera_stat:.2f}")
        out(f"    p-value:              {ret_dist.jarque_bera_p:.6f}")
        out(f"    Verdict:              {'NORMAL (fail to reject H0)' if ret_dist.is_normal else 'NON-NORMAL (reject H0 at 5%)'}")
        out()
        out(f"  Student-t Fit:")
        out(f"    Degrees of Freedom:   {ret_dist.t_fit_df:.2f}"
            f"  {'(very heavy tails)' if ret_dist.t_fit_df < 4 else '(heavy tails)' if ret_dist.t_fit_df < 10 else '(moderate tails)' if ret_dist.t_fit_df < 30 else '(near Gaussian)'}")
        out()
        out(f"  Risk Measures (empirical):")
        out(f"    VaR  95%:             {ret_dist.var_95 * 100:+.4f}% daily")
        out(f"    VaR  99%:             {ret_dist.var_99 * 100:+.4f}% daily")
        out(f"    CVaR 95% (Exp. Short):{ret_dist.cvar_95 * 100:+.4f}% daily")
        out(f"    CVaR 99% (Exp. Short):{ret_dist.cvar_99 * 100:+.4f}% daily")
        out()
        out(f"  Autocorrelation (returns, lag 1..5): "
            f"{', '.join(f'{a:+.4f}' for a in ret_dist.autocorr_returns)}")
        out(f"  Autocorrelation (|ret|, lag 1..5):   "
            f"{', '.join(f'{a:+.4f}' for a in ret_dist.autocorr_abs_returns)}")
        if ret_dist.autocorr_abs_returns and ret_dist.autocorr_abs_returns[0] > 0.1:
            out(f"    → Significant volatility clustering detected (lag-1 |ret| acf > 0.10)")

    # ------------------------------------------------------------------
    # Section: Volatility Regimes
    # ------------------------------------------------------------------
    if vol_regime and vol_regime.regime_counts:
        section_num += 1
        _section(out, f"{section_num}. VOLATILITY REGIME ANALYSIS")
        out()
        low_u, high_l, ext_l = vol_regime.regime_thresholds
        out(f"  Thresholds: Low < {low_u * 100:.1f}% | Normal ≤ {high_l * 100:.1f}% "
            f"| High ≤ {ext_l * 100:.1f}% | Extreme")
        out()
        out(f"  Current Regime:         {vol_regime.current_regime}")
        out(f"  Current Realised Vol:   {vol_regime.current_vol * 100:.1f}% (annualised)")
        out()

        out(f"  Regime Distribution:")
        total = sum(vol_regime.regime_counts.values())
        for regime in ("Low", "Normal", "High", "Extreme"):
            count = vol_regime.regime_counts.get(regime, 0)
            pct = count / total * 100 if total > 0 else 0
            durs = vol_regime.regime_durations.get(regime, [])
            avg_dur = sum(durs) / len(durs) if durs else 0
            out(f"    {regime:<10s} {count:>5d} days ({pct:>5.1f}%)  "
                f"avg duration: {avg_dur:.1f} days")
        out()

        if vol_regime.regime_stats:
            out(f"  Per-Regime Return Statistics:")
            out(f"  {'Regime':<10s} {'Mean Ret':>10s} {'Std Dev':>10s} {'Skew':>8s} {'Kurt':>8s}")
            out(f"  {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 8}")
            for regime in ("Low", "Normal", "High", "Extreme"):
                if regime in vol_regime.regime_stats:
                    m, s, sk, ku = vol_regime.regime_stats[regime]
                    out(f"  {regime:<10s} {m * 100:>+9.4f}% {s * 100:>9.4f}% {sk:>+7.3f} {ku:>+7.3f}")
            out()

        if vol_regime.vol_breakouts:
            out(f"  Normal → High/Extreme Breakouts ({len(vol_regime.vol_breakouts)}):")
            for date, vol in vol_regime.vol_breakouts[:15]:
                out(f"    {date}  vol = {vol * 100:.1f}%")

    # ------------------------------------------------------------------
    # Section: Structural Breaks
    # ------------------------------------------------------------------
    if structural_breaks:
        section_num += 1
        _section(out, f"{section_num}. STRUCTURAL BREAKS (SPIKE CLUSTERS)")
        out()
        out(f"  Top {len(structural_breaks)} break clusters (by peak absolute return):")
        out()
        out(f"  {'#':>3s}  {'Start':<12s} {'End':<12s} {'Peak':<12s} "
            f"{'|Ret|':>7s} {'Spikes':>7s} {'Pre Vol':>8s} {'Post Vol':>8s} {'Cum Ret':>9s}")
        out(f"  {'─' * 3}  {'─' * 12} {'─' * 12} {'─' * 12} "
            f"{'─' * 7} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 9}")

        for i, b in enumerate(structural_breaks, 1):
            out(
                f"  {i:>3d}  {b.start_date:<12s} {b.end_date:<12s} {b.peak_date:<12s} "
                f"{b.peak_abs_return * 100:>6.2f}% {b.num_spike_days:>7d} "
                f"{b.pre_regime_vol * 100:>7.1f}% {b.post_regime_vol * 100:>7.1f}% "
                f"{b.cumulative_return * 100:>+8.2f}%"
            )

    # ------------------------------------------------------------------
    # Section: Event Study
    # ------------------------------------------------------------------
    if event_study_results:
        section_num += 1
        _section(out, f"{section_num}. EVENT STUDY: SEC FILINGS (CAMPBELL-LO-MACKINLAY)")
        out()
        out("  Methodology: OLS market model (120-day estimation window)")
        out("  Benchmark: SPY | Event window: [0, +20] trading days")
        out()
        out(f"  {'Form':<12s} {'N':>4s} {'Mean CAR':>10s} {'Std CAR':>10s} "
            f"{'t-stat':>8s} {'p-value':>8s} {'Med CAR':>10s} {'CAR 1d':>10s} {'CAR 5d':>10s}")
        out(f"  {'─' * 12} {'─' * 4} {'─' * 10} {'─' * 10} "
            f"{'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10}")

        for r in event_study_results:
            sig = "**" if r.p_value < 0.05 else "*" if r.p_value < 0.10 else ""
            out(
                f"  {r.form_type:<12s} {r.event_count:>4d} "
                f"{r.mean_car * 100:>+9.3f}% {r.std_car * 100:>9.3f}% "
                f"{r.t_stat:>+7.3f} {r.p_value:>8.4f}{sig:2s}"
                f"{r.median_car * 100:>+9.3f}% "
                f"{r.mean_car_1d * 100:>+9.3f}% {r.mean_car_5d * 100:>+9.3f}%"
            )
        out()
        out("  ** p < 0.05 (significant)   * p < 0.10 (marginally significant)")

    # ------------------------------------------------------------------
    # Section: Volume Profile
    # ------------------------------------------------------------------
    if volume_profile and volume_profile.price_levels:
        section_num += 1
        _section(out, f"{section_num}. VOLUME PROFILE")
        out()
        out(f"  Point of Control (POC):   ${volume_profile.poc_price:.2f}")
        out(f"  Value Area (70% vol):     ${volume_profile.value_area_low:.2f} – "
            f"${volume_profile.value_area_high:.2f}")
        out()
        obv_dir = {1: "Rising", -1: "Falling", 0: "Flat"}
        out(f"  OBV Trend (20-day):       {obv_dir.get(volume_profile.obv_trend_direction, 'Unknown')}")
        out(f"  OBV/Price Divergence:     {'YES — potential reversal signal' if volume_profile.obv_price_divergence else 'No'}")

        if volume_profile.anomaly_days:
            out()
            out(f"  Volume Anomaly Days ({len(volume_profile.anomaly_days)} detected, z > 3.0):")
            for date, vol, z in volume_profile.anomaly_days[:15]:
                out(f"    {date}  vol={_fmt_volume(vol)}  z-score={z:.1f}")

    # ------------------------------------------------------------------
    # Section: Technical Signals
    # ------------------------------------------------------------------
    if alert:
        section_num += 1
        _section(out, f"{section_num}. CURRENT TECHNICAL SIGNALS")
        out()
        out(f"  Signal Direction:   {alert.direction.value}")
        out(f"  Confluence:         {alert.signal_strength}")
        out()
        if alert.bullish_indicators:
            out(f"  Bullish ({len(alert.bullish_indicators)}):")
            for ind in alert.bullish_indicators:
                out(f"    + {ind}")
        if alert.bearish_indicators:
            out(f"  Bearish ({len(alert.bearish_indicators)}):")
            for ind in alert.bearish_indicators:
                out(f"    - {ind}")
        out()
        out(f"  Call Rec:  {alert.call_recommendation}")
        out(f"  Put Rec:   {alert.put_recommendation}")

    # ------------------------------------------------------------------
    # Section: Options Snapshot
    # ------------------------------------------------------------------
    if opts_analysis:
        section_num += 1
        _section(out, f"{section_num}. OPTIONS CHAIN SNAPSHOT")
        out()
        out(f"  Underlying Price:      {_fmt_price(opts_analysis.underlying_price)}")
        out(f"  Nearest Expiry:        {opts_analysis.nearest_expiry}")
        out()
        out(f"  Call Volume / Put Volume:   {_fmt_volume(opts_analysis.total_call_volume)} / "
            f"{_fmt_volume(opts_analysis.total_put_volume)}")
        out(f"  Call OI / Put OI:          {_fmt_volume(opts_analysis.total_call_oi)} / "
            f"{_fmt_volume(opts_analysis.total_put_oi)}")
        out()
        out(f"  Put/Call Volume Ratio:     {float(opts_analysis.put_call_volume_ratio):.4f}")
        out(f"  Put/Call OI Ratio:         {float(opts_analysis.put_call_oi_ratio):.4f}")
        out()
        out(f"  Max Pain Strike:           {_fmt_price(opts_analysis.max_pain_strike)}")
        out(f"  Highest OI Call Strike:    {_fmt_price(opts_analysis.highest_oi_call_strike)}")
        out(f"  Highest OI Put Strike:     {_fmt_price(opts_analysis.highest_oi_put_strike)}")
        out()
        out(f"  IV Skew (Put - Call):      {opts_analysis.iv_skew * 100:.2f}%")

        if opts_analysis.unusual_volume_contracts:
            out()
            out(f"  Unusual Volume ({len(opts_analysis.unusual_volume_contracts)} contracts):")
            for symbol, strike, vol, oi in opts_analysis.unusual_volume_contracts[:10]:
                ratio = vol / oi if oi > 0 else 0
                out(f"    {symbol:<25s}  Strike: {_fmt_price(strike):>8s}  "
                    f"Vol: {vol:>6,d}  OI: {oi:>6,d}  ({ratio:.1f}x)")

    # ------------------------------------------------------------------
    # Section: Intraday Snapshot
    # ------------------------------------------------------------------
    if intraday_data:
        section_num += 1
        _section(out, f"{section_num}. INTRADAY SNAPSHOT (5-MIN BARS)")
        out()
        out(f"  Total Bars:   {len(intraday_data):,}")
        out(f"  Date Range:   {intraday_data[0].date} to {intraday_data[-1].date}")

        # Compute basic intraday stats
        intraday_high = max(intraday_data, key=lambda b: b.high)
        intraday_low = min(intraday_data, key=lambda b: b.low)
        total_vol = sum(b.volume for b in intraday_data)
        avg_bar_vol = total_vol // len(intraday_data) if intraday_data else 0

        out(f"  High:         {_fmt_price(intraday_high.high)} ({intraday_high.date})")
        out(f"  Low:          {_fmt_price(intraday_low.low)} ({intraday_low.date})")
        out(f"  Total Volume: {_fmt_volume(total_vol)}")
        out(f"  Avg Bar Vol:  {_fmt_volume(avg_bar_vol)}")

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    out()
    out(f"  {'─' * 72}")
    out("  DISCLAIMER: This analysis is for educational and informational purposes")
    out("  only. It does not constitute financial advice. Past performance does not")
    out("  guarantee future results. Always do your own research before trading.")
    out(f"  {'─' * 72}")
    out()


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Entry point for the GME comprehensive analysis application."""
    parser = argparse.ArgumentParser(
        description="Comprehensive GME (GameStop) Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages:\n"
            "  1. Download 6-year daily price data\n"
            "  2. Download ~60-day intraday 5-min data\n"
            "  3. Download SEC EDGAR filings\n"
            "  4. Download current options chain\n"
            "  5. Run analysis (patterns, filing correlation, options)\n"
            "  6. Generate terminal report\n"
        ),
    )
    parser.add_argument(
        "--skip-intraday",
        action="store_true",
        help="Skip Stage 2 (intraday data download, faster run)",
    )
    parser.add_argument(
        "--skip-options",
        action="store_true",
        help="Skip Stage 4 (options chain download)",
    )
    add_log_arg(
        parser,
        help="Write output to a log file in addition to stdout",
        metavar="FILE",
    )
    parser.add_argument(
        "--user-agent",
        default="StockDownloader admin@example.com",
        help="User-Agent for SEC EDGAR requests (default: %(default)s)",
    )
    args = parser.parse_args()

    out = _TeeWriter(args.log_file)

    start_time = time.time()

    out(_TL + _H * 76 + _TR)
    out(_V + "  COMPREHENSIVE GME (GAMESTOP) ANALYSIS".ljust(76) + _V)
    out(_BL + _H * 76 + _BR)
    out()
    out(f"  Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Stage 1: Daily data (GME + SPY benchmark)
    daily_data, benchmark_data = _stage_1_daily(out)
    if not daily_data:
        out("  FATAL: No daily data available. Cannot proceed.")
        out.close()
        return

    # Stage 2: Intraday data
    intraday_data: list[IntradayPriceData] = []
    if not args.skip_intraday:
        intraday_data = _stage_2_intraday(out)
    else:
        out()
        out("  Stage 2 skipped (--skip-intraday)")

    # Stage 3: SEC filings
    filings = _stage_3_sec_filings(out, args.user_agent)

    # Stage 4: Options chain
    chain: OptionsChain | None = None
    if not args.skip_options:
        chain = _stage_4_options(out)
    else:
        out()
        out("  Stage 4 skipped (--skip-options)")

    # Stage 5: Analysis
    (stats, filing_impacts, key_periods, opts_analysis, alert,
     ret_dist, vol_regime, structural_breaks, event_study_results, volume_profile,
    ) = _stage_5_analysis(
        out, daily_data, benchmark_data, filings, chain,
    )

    # Stage 6: Report
    _stage_6_report(
        out, daily_data, intraday_data,
        stats, filing_impacts, key_periods,
        opts_analysis, alert,
        ret_dist, vol_regime, structural_breaks,
        event_study_results, volume_profile,
    )

    elapsed = time.time() - start_time
    out(f"  Total time: {elapsed:.1f}s")
    out()

    out.close()


if __name__ == "__main__":
    main()
