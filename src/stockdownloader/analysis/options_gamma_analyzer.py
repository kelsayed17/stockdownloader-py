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

    from stockdownloader.data.yahoo_options_client import YahooOptionsClient
    from stockdownloader.analysis.options_gamma_analyzer import OptionsGammaAnalyzer

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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from stockdownloader.model.options import OptionsChain, OptionType

logger = logging.getLogger(__name__)

# Contract multiplier (1 option = 100 shares)
_MULT = 100


@dataclass
class StrikeGamma:
    """Gamma exposure at a single strike price."""

    strike: float
    call_gamma: float     # Raw gamma of calls at this strike
    put_gamma: float      # Raw gamma of puts at this strike
    call_oi: int          # Call open interest
    put_oi: int           # Put open interest
    call_volume: int      # Call volume
    put_volume: int       # Put volume
    net_gex: float        # Net gamma exposure in shares
    call_iv: float        # Call implied volatility
    put_iv: float         # Put implied volatility


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
        gex_map = self._compute_gex_by_strike(chain, price)

        # Max pain: use OI-based if available, volume-based fallback otherwise
        if has_oi:
            max_pain = self._compute_max_pain(chain, price)
            mp_source = "oi"
        else:
            max_pain = self._compute_max_pain_from_volume(chain, price)
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
        gex_flip = self._find_gex_flip(gex_map, price) if has_greeks else 0.0

        # Unusual activity (requires OI > 0)
        unusual = self._find_unusual_activity(chain) if has_oi else []

        # Nearest expiry analysis
        nearest_exp = ""
        nearest_gex = 0.0
        if chain.expiration_dates:
            nearest_exp = chain.expiration_dates[0]
            if has_greeks:
                nearest_gex = self._compute_expiry_gex(chain, nearest_exp, price)

        # ----------------------------------------------------------
        # Directional signals
        # ----------------------------------------------------------
        if has_greeks:
            gex_signal = self._interpret_gex(net_gex, price)
        else:
            gex_signal = "insufficient_data"

        if has_oi:
            mp_signal = self._interpret_max_pain(max_pain, price)
        else:
            # Volume-based max pain is less reliable; still compute but
            # note it in the report
            mp_signal = self._interpret_max_pain(max_pain, price)

        # For PCR, prefer volume ratio when OI is missing
        if has_oi:
            pcr_signal = self._interpret_pcr(pcr_volume, pcr_oi)
        else:
            pcr_signal = self._interpret_pcr_volume_only(pcr_volume)

        overall = self._combine_signals(gex_signal, mp_signal, pcr_signal)

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
    # Gamma Exposure (GEX)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gex_by_strike(
        chain: OptionsChain, price: float
    ) -> dict[float, StrikeGamma]:
        """Compute net gamma exposure at each strike across all expirations.

        GEX formula per strike:
            Call GEX = call_gamma * call_OI * 100 * price
            Put GEX  = put_gamma * put_OI * 100 * price * (-1)
            Net GEX  = Call GEX + Put GEX

        Market makers are short calls and long puts (from retail buying),
        so:
        - Positive net GEX → MMs sell into rallies, buy dips (dampening)
        - Negative net GEX → MMs amplify moves (gamma squeeze potential)
        """
        gex_map: dict[float, dict] = {}

        # Aggregate calls
        for call in chain.all_calls:
            strike = float(call.strike)
            gamma = float(call.gamma)
            if strike not in gex_map:
                gex_map[strike] = {
                    "call_gamma": 0.0, "put_gamma": 0.0,
                    "call_oi": 0, "put_oi": 0,
                    "call_volume": 0, "put_volume": 0,
                    "call_iv": 0.0, "put_iv": 0.0,
                    "call_count": 0, "put_count": 0,
                }
            gex_map[strike]["call_gamma"] += gamma * call.open_interest
            gex_map[strike]["call_oi"] += call.open_interest
            gex_map[strike]["call_volume"] += call.volume
            gex_map[strike]["call_iv"] += float(call.implied_volatility)
            gex_map[strike]["call_count"] += 1

        # Aggregate puts
        for put in chain.all_puts:
            strike = float(put.strike)
            gamma = float(put.gamma)
            if strike not in gex_map:
                gex_map[strike] = {
                    "call_gamma": 0.0, "put_gamma": 0.0,
                    "call_oi": 0, "put_oi": 0,
                    "call_volume": 0, "put_volume": 0,
                    "call_iv": 0.0, "put_iv": 0.0,
                    "call_count": 0, "put_count": 0,
                }
            gex_map[strike]["put_gamma"] += gamma * put.open_interest
            gex_map[strike]["put_oi"] += put.open_interest
            gex_map[strike]["put_volume"] += put.volume
            gex_map[strike]["put_iv"] += float(put.implied_volatility)
            gex_map[strike]["put_count"] += 1

        # Convert to StrikeGamma objects
        result: dict[float, StrikeGamma] = {}
        for strike, data in gex_map.items():
            # Net GEX: calls contribute positive, puts negative
            call_gex = data["call_gamma"] * _MULT * price
            put_gex = -data["put_gamma"] * _MULT * price
            net = call_gex + put_gex

            avg_call_iv = (
                data["call_iv"] / data["call_count"]
                if data["call_count"] > 0 else 0.0
            )
            avg_put_iv = (
                data["put_iv"] / data["put_count"]
                if data["put_count"] > 0 else 0.0
            )

            # For StrikeGamma, store the OI-weighted average gamma
            call_gamma_avg = (
                data["call_gamma"] / data["call_oi"]
                if data["call_oi"] > 0 else 0.0
            )
            put_gamma_avg = (
                data["put_gamma"] / data["put_oi"]
                if data["put_oi"] > 0 else 0.0
            )

            result[strike] = StrikeGamma(
                strike=strike,
                call_gamma=round(call_gamma_avg, 6),
                put_gamma=round(put_gamma_avg, 6),
                call_oi=data["call_oi"],
                put_oi=data["put_oi"],
                call_volume=data["call_volume"],
                put_volume=data["put_volume"],
                net_gex=round(net, 0),
                call_iv=round(avg_call_iv, 4),
                put_iv=round(avg_put_iv, 4),
            )

        return result

    @staticmethod
    def _compute_expiry_gex(
        chain: OptionsChain, expiry: str, price: float
    ) -> float:
        """Compute net GEX for a single expiration date."""
        gex = 0.0
        for call in chain.get_calls(expiry):
            gex += float(call.gamma) * call.open_interest * _MULT * price
        for put in chain.get_puts(expiry):
            gex -= float(put.gamma) * put.open_interest * _MULT * price
        return gex

    @staticmethod
    def _find_gex_flip(
        gex_map: dict[float, StrikeGamma], price: float
    ) -> float:
        """Find the strike price where GEX flips sign nearest to current price.

        Returns 0.0 if no flip point is found.
        """
        sorted_strikes = sorted(gex_map.keys())
        if len(sorted_strikes) < 2:
            return 0.0

        flip_point = 0.0
        min_dist = float("inf")

        for i in range(len(sorted_strikes) - 1):
            s1 = sorted_strikes[i]
            s2 = sorted_strikes[i + 1]
            g1 = gex_map[s1].net_gex
            g2 = gex_map[s2].net_gex

            if g1 * g2 < 0:  # Sign change
                # Linear interpolation
                ratio = abs(g1) / (abs(g1) + abs(g2))
                flip = s1 + (s2 - s1) * ratio
                dist = abs(flip - price)
                if dist < min_dist:
                    min_dist = dist
                    flip_point = flip

        return flip_point

    # ------------------------------------------------------------------
    # Max Pain
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_pain(chain: OptionsChain, price: float) -> float:
        """Compute the max pain strike — the price at which the total
        value of all outstanding options is minimized.

        At max pain, the maximum number of options expire worthless,
        benefiting option sellers (typically market makers).
        """
        # Collect all unique strikes
        strikes: set[float] = set()
        for c in chain.all_calls:
            strikes.add(float(c.strike))
        for p in chain.all_puts:
            strikes.add(float(p.strike))

        if not strikes:
            return price

        # For each potential settlement price, compute total intrinsic value
        # of all options.  Max pain = strike with minimum total value.
        min_pain = float("inf")
        max_pain_strike = price

        sorted_strikes = sorted(strikes)

        for test_price in sorted_strikes:
            total_pain = 0.0

            # Pain for call holders (loss if price < strike)
            for call in chain.all_calls:
                strike = float(call.strike)
                if test_price > strike:
                    total_pain += (test_price - strike) * call.open_interest * _MULT
                # If test_price <= strike, calls expire worthless (no pain for MMs)

            # Pain for put holders (loss if price > strike)
            for put in chain.all_puts:
                strike = float(put.strike)
                if test_price < strike:
                    total_pain += (strike - test_price) * put.open_interest * _MULT

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price

        return max_pain_strike

    @staticmethod
    def _compute_max_pain_from_volume(
        chain: OptionsChain, price: float
    ) -> float:
        """Compute max pain using volume instead of open interest.

        This is a fallback for when Yahoo returns OI=0 for all contracts.
        Volume-based max pain is less precise but still useful: it reflects
        current trading activity rather than accumulated positions.
        """
        strikes: set[float] = set()
        for c in chain.all_calls:
            strikes.add(float(c.strike))
        for p in chain.all_puts:
            strikes.add(float(p.strike))

        if not strikes:
            return price

        min_pain = float("inf")
        max_pain_strike = price

        sorted_strikes = sorted(strikes)

        for test_price in sorted_strikes:
            total_pain = 0.0

            for call in chain.all_calls:
                strike = float(call.strike)
                if test_price > strike:
                    total_pain += (test_price - strike) * call.volume * _MULT

            for put in chain.all_puts:
                strike = float(put.strike)
                if test_price < strike:
                    total_pain += (strike - test_price) * put.volume * _MULT

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price

        return max_pain_strike

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
    # Signal Interpretation
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_gex(net_gex: float, price: float) -> str:
        """Interpret net GEX direction.

        Positive GEX → dampening (MMs hedge by selling into rallies).
        Negative GEX → amplifying (gamma squeeze potential).
        """
        # Normalize by price to make threshold meaningful
        gex_normalized = net_gex / (price * 1_000_000) if price > 0 else 0
        if gex_normalized > 0.5:
            return "dampening"
        elif gex_normalized < -0.5:
            return "amplifying"
        return "neutral"

    @staticmethod
    def _interpret_max_pain(max_pain: float, price: float) -> str:
        """Interpret max pain relative to current price."""
        if price <= 0:
            return "neutral"
        dist_pct = (max_pain - price) / price * 100
        if dist_pct > 3:
            return "bullish"  # Max pain above price → gravitational pull up
        elif dist_pct < -3:
            return "bearish"  # Max pain below price → gravitational pull down
        return "neutral"

    @staticmethod
    def _interpret_pcr(pcr_vol: float, pcr_oi: float) -> str:
        """Interpret put/call ratio.

        High PCR (> 1.2) = excessive put buying → contrarian bullish
        Low PCR (< 0.5) = excessive call buying → contrarian bearish
        """
        avg_pcr = (pcr_vol + pcr_oi) / 2
        if avg_pcr > 1.2:
            return "bullish"   # Contrarian: too many puts → squeeze fuel
        elif avg_pcr < 0.5:
            return "bearish"   # Contrarian: too many calls → potential fade
        return "neutral"

    @staticmethod
    def _interpret_pcr_volume_only(pcr_vol: float) -> str:
        """Interpret put/call ratio using volume only (OI unavailable).

        Same thresholds as the standard PCR interpreter but uses only
        volume data, which is noisier than OI.
        """
        if pcr_vol > 1.2:
            return "bullish"   # Contrarian: heavy put volume
        elif pcr_vol < 0.5:
            return "bearish"   # Heavy call volume
        return "neutral"

    @staticmethod
    def _combine_signals(gex: str, mp: str, pcr: str) -> str:
        """Combine individual signals into an overall assessment."""
        scores = {"bullish": 1, "amplifying": 1, "neutral": 0,
                  "bearish": -1, "dampening": -1,
                  "insufficient_data": 0}
        # Count how many signals actually have data
        valid_signals = [s for s in (gex, mp, pcr)
                         if s != "insufficient_data"]
        total = scores.get(gex, 0) + scores.get(mp, 0) + scores.get(pcr, 0)

        if not valid_signals:
            return "INSUFFICIENT_DATA"

        if total >= 2:
            return "BULLISH"
        elif total <= -2:
            return "BEARISH"
        elif total > 0:
            return "SLIGHTLY_BULLISH"
        elif total < 0:
            return "SLIGHTLY_BEARISH"
        return "NEUTRAL"

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
