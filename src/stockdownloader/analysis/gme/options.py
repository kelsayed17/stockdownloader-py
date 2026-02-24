"""Options chain analysis: max pain, P/C ratios, unusual activity, IV skew."""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.analysis.gme.models import OptionsAnalysis
from stockdownloader.core.models.options import OptionContract, OptionsChain

_ZERO = Decimal("0")

__all__ = [
    "analyze_options_chain",
]


def analyze_options_chain(
    chain: OptionsChain,
    underlying_price: Decimal | None = None,
) -> OptionsAnalysis:
    """Analyse an options chain snapshot: P/C ratios, max pain, unusual activity."""
    price = underlying_price or chain.underlying_price or _ZERO

    all_calls = chain.all_calls
    all_puts = chain.all_puts

    total_call_vol = chain.total_call_volume
    total_put_vol = chain.total_put_volume
    total_call_oi = chain.total_call_open_interest
    total_put_oi = chain.total_put_open_interest

    # Put/call ratios
    pc_vol_ratio = chain.put_call_ratio
    pc_oi_ratio = (
        Decimal(total_put_oi) / Decimal(total_call_oi)
        if total_call_oi > 0 else _ZERO
    ).quantize(Decimal("0.0001"))

    # Nearest expiry
    expirations = chain.expiration_dates
    nearest_expiry = expirations[0] if expirations else ""

    # Max pain for nearest expiry
    max_pain = _compute_max_pain(chain, nearest_expiry) if nearest_expiry else _ZERO

    # Highest OI strikes
    highest_oi_call = _highest_oi_strike(all_calls)
    highest_oi_put = _highest_oi_strike(all_puts)

    # Unusual volume: contracts where volume > 3x open interest
    unusual = _find_unusual_volume(all_calls + all_puts)

    # IV skew: ATM put IV minus ATM call IV (nearest expiry)
    iv_skew = _compute_iv_skew(chain, nearest_expiry, price)

    return OptionsAnalysis(
        underlying_price=price,
        total_call_volume=total_call_vol,
        total_put_volume=total_put_vol,
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        put_call_volume_ratio=pc_vol_ratio,
        put_call_oi_ratio=pc_oi_ratio,
        max_pain_strike=max_pain,
        nearest_expiry=nearest_expiry,
        highest_oi_call_strike=highest_oi_call,
        highest_oi_put_strike=highest_oi_put,
        unusual_volume_contracts=unusual,
        iv_skew=round(iv_skew, 4),
    )


def _compute_max_pain(chain: OptionsChain, expiration: str) -> Decimal:
    """Compute the max-pain strike for a given expiration.

    Max pain is the strike where the total value of expiring options
    (calls + puts) is minimised for holders.
    """
    calls = chain.get_calls(expiration)
    puts = chain.get_puts(expiration)
    if not calls and not puts:
        return _ZERO

    # Collect all unique strikes
    strikes: set[Decimal] = set()
    for c in calls:
        strikes.add(c.strike)
    for p in puts:
        strikes.add(p.strike)

    best_strike = _ZERO
    min_pain = float("inf")

    for strike in sorted(strikes):
        total_pain = 0.0
        # Call holders lose money when stock is below their strike
        # Put holders lose money when stock is above their strike
        for c in calls:
            if strike < c.strike:
                # Calls expire worthless -> holders lose premium (irrelevant to pin)
                pass
            else:
                # Calls are ITM -> holders gain (strike - call_strike) * OI
                total_pain += float(strike - c.strike) * c.open_interest
        for p in puts:
            if strike > p.strike:
                # Puts expire worthless -> holders lose premium
                pass
            else:
                # Puts are ITM -> holders gain (put_strike - strike) * OI
                total_pain += float(p.strike - strike) * p.open_interest

        if total_pain < min_pain:
            min_pain = total_pain
            best_strike = strike

    return best_strike


def _highest_oi_strike(contracts: list[OptionContract]) -> Decimal:
    """Return the strike with the highest open interest."""
    if not contracts:
        return _ZERO
    best = max(contracts, key=lambda c: c.open_interest)
    return best.strike


def _find_unusual_volume(
    contracts: list[OptionContract],
    ratio_threshold: float = 3.0,
) -> list[tuple[str, Decimal, int, int]]:
    """Find contracts where volume exceeds *ratio_threshold* x open interest."""
    unusual: list[tuple[str, Decimal, int, int]] = []
    for c in contracts:
        if c.open_interest > 0 and c.volume > ratio_threshold * c.open_interest:
            unusual.append((c.contract_symbol, c.strike, c.volume, c.open_interest))
    # Sort by volume descending
    unusual.sort(key=lambda x: x[2], reverse=True)
    return unusual[:20]  # cap at 20 most unusual


def _compute_iv_skew(
    chain: OptionsChain,
    expiration: str,
    underlying_price: Decimal,
) -> float:
    """Compute IV skew: average ATM put IV minus average ATM call IV.

    ATM is defined as contracts within 5% of the underlying price.
    """
    if underlying_price <= 0:
        return 0.0

    atm_range = float(underlying_price) * 0.05
    price_f = float(underlying_price)

    call_ivs: list[float] = []
    for c in chain.get_calls(expiration):
        if abs(float(c.strike) - price_f) <= atm_range:
            call_ivs.append(float(c.implied_volatility))

    put_ivs: list[float] = []
    for p in chain.get_puts(expiration):
        if abs(float(p.strike) - price_f) <= atm_range:
            put_ivs.append(float(p.implied_volatility))

    avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0.0
    avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else 0.0

    return avg_put_iv - avg_call_iv
