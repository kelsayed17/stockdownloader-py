"""Options gamma exposure, max pain, and flow analysis.

Aggregates data from :class:`OptionsChain` (fetched by Yahoo Options Client)
to compute:

- **Net Gamma Exposure (GEX)** per strike and total — reveals market maker
  hedging pressure.  Positive GEX = MMs sell into rallies (dampening).
  Negative GEX = MMs buy into rallies (amplifying / gamma squeeze).
- **Max Pain** — the strike price where total option premium decay is
  maximized, i.e. where market makers profit most.
- **Put/Call Ratios** — by volume, open interest, and dollar-weighted.
- **Unusual Activity** — strikes with volume significantly exceeding OI.
- **Gamma Walls** — strikes with outsized gamma concentration.

Usage::

    from stockdownloader.data.market.yahoo_options_client import YahooOptionsClient
    from stockdownloader.analysis.options.gamma_analyzer import OptionsGammaAnalyzer

    client = YahooOptionsClient()
    chain = client.download("GME")

    analyzer = OptionsGammaAnalyzer()
    report = analyzer.analyze(chain)

    print(f"Max Pain: ${report.max_pain}")
    print(f"Net GEX: {report.net_gex:,.0f} shares")
    print(f"Put/Call: {report.pcr_volume:.2f}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from stockdownloader.analysis.options.gex import (
    StrikeGamma,
    _MULT,
    combine_signals,
    compute_expiry_gex,
    compute_gex_by_strike,
    compute_max_pain,
    compute_max_pain_from_volume,
    find_gex_flip,
    interpret_gex,
    interpret_max_pain,
    interpret_pcr,
    interpret_pcr_volume_only,
)
from stockdownloader.core.models.options import OptionsChain

logger = logging.getLogger(__name__)


@dataclass
class UnusualActivity:
    """A strike with unusual options volume."""

    strike: float
    expiration: str
    option_type: str      # "CALL" or "PUT"
    volume: int
    open_interest: int
    vol_oi_ratio: float
    implied_volatility: float
    notional: float       # Dollar notional of the volume


@dataclass
class OptionsFlowReport:
    """Complete options analysis report."""

    symbol: str
    underlying_price: float
    analysis_time: str

    # Max pain
    max_pain: float
    max_pain_distance_pct: float  # % distance from current price to max pain

    # Gamma exposure
    net_gex: float                # Total net GEX in shares
    gex_by_strike: list[StrikeGamma]
    gamma_wall_call: float        # Strike with highest call gamma concentration
    gamma_wall_put: float         # Strike with highest put gamma concentration
    gex_flip_point: float         # Price where GEX flips from positive to negative

    # Put/Call ratios
    pcr_volume: float             # Put/Call ratio by volume
    pcr_oi: float                 # Put/Call ratio by open interest
    pcr_dollar: float             # Put/Call ratio by dollar volume

    # Volume / OI stats
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int

    # Unusual activity
    unusual_activity: list[UnusualActivity]

    # Expiration breakdown
    nearest_expiry: str
    nearest_expiry_gex: float
    expirations_analyzed: int

    # Directional signals
    gex_signal: str               # "dampening", "amplifying", or "neutral"
    max_pain_signal: str          # "bullish", "bearish", or "neutral"
    pcr_signal: str               # "bullish", "bearish", or "neutral"
    overall_signal: str           # Combined assessment

    # Data quality flags
    has_oi: bool = True           # True if open interest data is available
    has_greeks: bool = True       # True if greeks (delta/gamma) are available
    max_pain_source: str = "oi"   # "oi" (normal) or "volume" (fallback)
    data_quality: str = "full"    # "full", "partial", or "degraded"

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        d = asdict(self)
        return d


class OptionsGammaAnalyzer:
    """Analyzes options chains for gamma exposure, max pain, and flow signals.

    Parameters
    ----------
    cache_dir:
        Directory for persisting analysis results.
    unusual_vol_oi_threshold:
        Minimum volume/OI ratio to flag as unusual activity.
    """

    def __init__(
        self,
        data_dir: str = "data",
        unusual_vol_oi_threshold: float = 3.0,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._unusual_threshold = unusual_vol_oi_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, chain: OptionsChain) -> OptionsFlowReport:
        """Run full options analysis on an :class:`OptionsChain`.

        Automatically detects data quality issues (missing OI, missing
        Greeks) and falls back to volume-based calculations when needed.

        Returns an :class:`OptionsFlowReport` with all metrics computed.
        """
        symbol = chain.underlying_symbol
        price = float(chain.underlying_price)

        if price <= 0:
            raise ValueError(f"Invalid underlying price: {price}")

        # ----------------------------------------------------------
        # Data quality detection
        # ----------------------------------------------------------
        total_call_vol = chain.total_call_volume
        total_put_vol = chain.total_put_volume
        total_call_oi = chain.total_call_open_interest
        total_put_oi = chain.total_put_open_interest

        has_oi = (total_call_oi + total_put_oi) > 0
        has_greeks = any(
            float(c.gamma) != 0 for c in chain.all_calls
        ) or any(
            float(p.gamma) != 0 for p in chain.all_puts
        )

        if not has_oi:
            logger.info(
                "Options data for %s has zero OI — using volume-based "
                "fallbacks for max pain and PCR",
                symbol,
            )
        if not has_greeks:
            logger.info(
                "Options data for %s has zero Greeks — GEX analysis "
                "will be marked as insufficient",
                symbol,
            )

        # Determine data quality level
        if has_oi and has_greeks:
            data_quality = "full"
        elif has_oi or has_greeks:
            data_quality = "partial"
        else:
            data_quality = "degraded"

        # ----------------------------------------------------------
        # Aggregate all contracts across expirations
        # ----------------------------------------------------------
        gex_map = compute_gex_by_strike(chain, price)

        # Max pain: use OI-based if available, volume-based fallback otherwise
        if has_oi:
            max_pain = compute_max_pain(chain, price)
            mp_source = "oi"
        else:
            max_pain = compute_max_pain_from_volume(chain, price)
            mp_source = "volume"

        # Put/Call ratios
        pcr_volume = total_put_vol / max(total_call_vol, 1)
        pcr_oi = total_put_oi / max(total_call_oi, 1) if has_oi else 0.0

        # Dollar-weighted PCR
        call_dollar = sum(
            float(c.last_price) * c.volume * _MULT
            for c in chain.all_calls
        )
        put_dollar = sum(
            float(p.last_price) * p.volume * _MULT
            for p in chain.all_puts
        )
        pcr_dollar = put_dollar / max(call_dollar, 1.0)

        # Net GEX (only meaningful if we have Greeks)
        if has_greeks:
            net_gex = sum(sg.net_gex for sg in gex_map.values())
        else:
            net_gex = 0.0

        # Gamma walls (strikes with highest OI-weighted gamma)
        gamma_wall_call = 0.0
        gamma_wall_put = 0.0
        max_call_gex = 0.0
        max_put_gex = 0.0
        if has_greeks:
            for sg in gex_map.values():
                call_gex = sg.call_gamma * sg.call_oi * _MULT * price
                put_gex = sg.put_gamma * sg.put_oi * _MULT * price
                if call_gex > max_call_gex:
                    max_call_gex = call_gex
                    gamma_wall_call = sg.strike
                if put_gex > max_put_gex:
                    max_put_gex = put_gex
                    gamma_wall_put = sg.strike

        # GEX flip point (strike where cumulative GEX changes sign)
        gex_flip = find_gex_flip(gex_map, price) if has_greeks else 0.0

        # Unusual activity (requires OI > 0)
        unusual = self._find_unusual_activity(chain) if has_oi else []

        # Nearest expiry analysis
        nearest_exp = ""
        nearest_gex = 0.0
        if chain.expiration_dates:
            nearest_exp = chain.expiration_dates[0]
            if has_greeks:
                nearest_gex = compute_expiry_gex(chain, nearest_exp, price)

        # ----------------------------------------------------------
        # Directional signals
        # ----------------------------------------------------------
        if has_greeks:
            gex_signal = interpret_gex(net_gex, price)
        else:
            gex_signal = "insufficient_data"

        if has_oi:
            mp_signal = interpret_max_pain(max_pain, price)
        else:
            # Volume-based max pain is less reliable; still compute but
            # note it in the report
            mp_signal = interpret_max_pain(max_pain, price)

        # For PCR, prefer volume ratio when OI is missing
        if has_oi:
            pcr_signal = interpret_pcr(pcr_volume, pcr_oi)
        else:
            pcr_signal = interpret_pcr_volume_only(pcr_volume)

        overall = combine_signals(gex_signal, mp_signal, pcr_signal)

        mp_dist = ((max_pain - price) / price) * 100 if price > 0 else 0

        report = OptionsFlowReport(
            symbol=symbol,
            underlying_price=round(price, 2),
            analysis_time=datetime.utcnow().isoformat(),
            max_pain=round(max_pain, 2),
            max_pain_distance_pct=round(mp_dist, 2),
            net_gex=round(net_gex, 0),
            gex_by_strike=sorted(gex_map.values(), key=lambda s: s.strike),
            gamma_wall_call=round(gamma_wall_call, 2),
            gamma_wall_put=round(gamma_wall_put, 2),
            gex_flip_point=round(gex_flip, 2),
            pcr_volume=round(pcr_volume, 4),
            pcr_oi=round(pcr_oi, 4),
            pcr_dollar=round(pcr_dollar, 4),
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            unusual_activity=unusual[:20],  # Top 20
            nearest_expiry=nearest_exp,
            nearest_expiry_gex=round(nearest_gex, 0),
            expirations_analyzed=len(chain.expiration_dates),
            gex_signal=gex_signal,
            max_pain_signal=mp_signal,
            pcr_signal=pcr_signal,
            overall_signal=overall,
            has_oi=has_oi,
            has_greeks=has_greeks,
            max_pain_source=mp_source,
            data_quality=data_quality,
        )

        return report

    def analyze_and_cache(self, chain: OptionsChain) -> OptionsFlowReport:
        """Analyze chain and persist results to cache."""
        report = self.analyze(chain)
        self._save_cache(report)
        return report

    # ------------------------------------------------------------------
    # Unusual Activity Detection
    # ------------------------------------------------------------------

    def _find_unusual_activity(
        self, chain: OptionsChain
    ) -> list[UnusualActivity]:
        """Find strikes with volume significantly exceeding open interest."""
        unusual: list[UnusualActivity] = []

        for exp in chain.expiration_dates:
            for call in chain.get_calls(exp):
                if call.open_interest > 0 and call.volume > 0:
                    ratio = call.volume / call.open_interest
                    if ratio >= self._unusual_threshold:
                        unusual.append(UnusualActivity(
                            strike=float(call.strike),
                            expiration=exp,
                            option_type="CALL",
                            volume=call.volume,
                            open_interest=call.open_interest,
                            vol_oi_ratio=round(ratio, 2),
                            implied_volatility=round(
                                float(call.implied_volatility), 4
                            ),
                            notional=round(
                                float(call.last_price) * call.volume * _MULT, 2
                            ),
                        ))

            for put in chain.get_puts(exp):
                if put.open_interest > 0 and put.volume > 0:
                    ratio = put.volume / put.open_interest
                    if ratio >= self._unusual_threshold:
                        unusual.append(UnusualActivity(
                            strike=float(put.strike),
                            expiration=exp,
                            option_type="PUT",
                            volume=put.volume,
                            open_interest=put.open_interest,
                            vol_oi_ratio=round(ratio, 2),
                            implied_volatility=round(
                                float(put.implied_volatility), 4
                            ),
                            notional=round(
                                float(put.last_price) * put.volume * _MULT, 2
                            ),
                        ))

        # Sort by notional value descending
        unusual.sort(key=lambda u: u.notional, reverse=True)
        return unusual

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _symbol_dir(self, symbol: str) -> Path:
        """Return per-symbol data directory, creating it if needed."""
        d = self._data_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_cache(self, report: OptionsFlowReport) -> None:
        """Persist analysis report to JSON cache."""
        cache_file = self._symbol_dir(report.symbol) / "options_gamma.json"
        try:
            # Convert StrikeGamma and UnusualActivity to dicts
            data = report.to_dict()
            cache_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(
                "Saved options gamma analysis for %s to %s",
                report.symbol, cache_file,
            )
        except OSError as exc:
            logger.warning(
                "Failed to save gamma cache for %s: %s",
                report.symbol, exc,
            )

    def load_cache(self, symbol: str) -> dict | None:
        """Load cached analysis report."""
        cache_file = self._symbol_dir(symbol) / "options_gamma.json"
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
