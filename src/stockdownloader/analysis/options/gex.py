"""GEX, max-pain, and signal interpretation helpers.

Extracted from :mod:`options_gamma_analyzer` to keep the analyzer class
focused on orchestration while housing the pure-computation logic here.

All functions are stateless (no instance data) and operate on
:class:`~stockdownloader.core.models.options.OptionsChain` objects or simple
numeric inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from stockdownloader.core.models.options import OptionsChain

# Contract multiplier (1 option = 100 shares)
_MULT = 100


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# GEX computation
# ------------------------------------------------------------------

def compute_gex_by_strike(
    chain: OptionsChain, price: float
) -> dict[float, StrikeGamma]:
    """Compute net gamma exposure at each strike across all expirations.

    GEX formula per strike:
        Call GEX = call_gamma * call_OI * 100 * price
        Put GEX  = put_gamma * put_OI * 100 * price * (-1)
        Net GEX  = Call GEX + Put GEX

    Market makers are short calls and long puts (from retail buying),
    so:
    - Positive net GEX -> MMs sell into rallies, buy dips (dampening)
    - Negative net GEX -> MMs amplify moves (gamma squeeze potential)
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


def compute_expiry_gex(
    chain: OptionsChain, expiry: str, price: float
) -> float:
    """Compute net GEX for a single expiration date."""
    gex = 0.0
    for call in chain.get_calls(expiry):
        gex += float(call.gamma) * call.open_interest * _MULT * price
    for put in chain.get_puts(expiry):
        gex -= float(put.gamma) * put.open_interest * _MULT * price
    return gex


def find_gex_flip(
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
# Max pain
# ------------------------------------------------------------------

def compute_max_pain(chain: OptionsChain, price: float) -> float:
    """Compute the max pain strike -- the price at which the total
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


def compute_max_pain_from_volume(
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
# Signal interpretation
# ------------------------------------------------------------------

def interpret_gex(net_gex: float, price: float) -> str:
    """Interpret net GEX direction.

    Positive GEX -> dampening (MMs hedge by selling into rallies).
    Negative GEX -> amplifying (gamma squeeze potential).
    """
    # Normalize by price to make threshold meaningful
    gex_normalized = net_gex / (price * 1_000_000) if price > 0 else 0
    if gex_normalized > 0.5:
        return "dampening"
    elif gex_normalized < -0.5:
        return "amplifying"
    return "neutral"


def interpret_max_pain(max_pain: float, price: float) -> str:
    """Interpret max pain relative to current price."""
    if price <= 0:
        return "neutral"
    dist_pct = (max_pain - price) / price * 100
    if dist_pct > 3:
        return "bullish"  # Max pain above price -> gravitational pull up
    elif dist_pct < -3:
        return "bearish"  # Max pain below price -> gravitational pull down
    return "neutral"


def interpret_pcr(pcr_vol: float, pcr_oi: float) -> str:
    """Interpret put/call ratio.

    High PCR (> 1.2) = excessive put buying -> contrarian bullish
    Low PCR (< 0.5) = excessive call buying -> contrarian bearish
    """
    avg_pcr = (pcr_vol + pcr_oi) / 2
    if avg_pcr > 1.2:
        return "bullish"   # Contrarian: too many puts -> squeeze fuel
    elif avg_pcr < 0.5:
        return "bearish"   # Contrarian: too many calls -> potential fade
    return "neutral"


def interpret_pcr_volume_only(pcr_vol: float) -> str:
    """Interpret put/call ratio using volume only (OI unavailable).

    Same thresholds as the standard PCR interpreter but uses only
    volume data, which is noisier than OI.
    """
    if pcr_vol > 1.2:
        return "bullish"   # Contrarian: heavy put volume
    elif pcr_vol < 0.5:
        return "bearish"   # Heavy call volume
    return "neutral"


def combine_signals(gex: str, mp: str, pcr: str) -> str:
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
