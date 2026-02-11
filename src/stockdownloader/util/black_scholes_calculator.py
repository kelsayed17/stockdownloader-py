"""Black-Scholes option pricing model for synthetic option price estimation.

Used by the options backtesting engine to estimate option prices from
underlying price data when historical options data is not available.
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

from scipy.stats import norm

from stockdownloader.model.option_type import OptionType


def price(
    option_type: OptionType,
    spot: Decimal,
    strike: Decimal,
    time_to_expiry: Decimal,
    risk_free_rate: Decimal,
    volatility: Decimal,
) -> Decimal:
    """Calculate the theoretical option price using the Black-Scholes formula.

    Args:
        option_type: ``OptionType.CALL`` or ``OptionType.PUT``.
        spot: Current underlying price.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years (e.g. 30/365 for 30 days).
        risk_free_rate: Annual risk-free interest rate (e.g. 0.05 for 5%).
        volatility: Annualized implied volatility (e.g. 0.20 for 20%).

    Returns:
        Theoretical option price as a ``Decimal`` rounded to 4 decimal places.
    """
    if time_to_expiry <= Decimal('0'):
        return intrinsic_value(option_type, spot, strike)
    if volatility <= Decimal('0'):
        return intrinsic_value(option_type, spot, strike)

    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry)
    r = float(risk_free_rate)
    sigma = float(volatility)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == OptionType.CALL:
        result = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        result = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return Decimal(str(max(result, 0.0))).quantize(
        Decimal('0.0001'), rounding=ROUND_HALF_UP
    )


def delta(
    option_type: OptionType,
    spot: Decimal,
    strike: Decimal,
    time_to_expiry: Decimal,
    risk_free_rate: Decimal,
    volatility: Decimal,
) -> Decimal:
    """Calculate delta: rate of change of option price w.r.t. underlying price."""
    if time_to_expiry <= Decimal('0') or volatility <= Decimal('0'):
        if option_type == OptionType.CALL:
            return Decimal('1') if spot > strike else Decimal('0')
        else:
            return Decimal('-1') if spot < strike else Decimal('0')

    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry)
    r = float(risk_free_rate)
    sigma = float(volatility)

    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * math.sqrt(T))

    if option_type == OptionType.CALL:
        d = norm.cdf(d1)
    else:
        d = norm.cdf(d1) - 1.0

    return Decimal(str(d)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)


def theta(
    option_type: OptionType,
    spot: Decimal,
    strike: Decimal,
    time_to_expiry: Decimal,
    risk_free_rate: Decimal,
    volatility: Decimal,
) -> Decimal:
    """Calculate theta: time decay per day."""
    if time_to_expiry <= Decimal('0') or volatility <= Decimal('0'):
        return Decimal('0')

    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry)
    r = float(risk_free_rate)
    sigma = float(volatility)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    nd1 = math.exp(-d1 * d1 / 2.0) / math.sqrt(2 * math.pi)

    if option_type == OptionType.CALL:
        daily_theta = (
            -S * nd1 * sigma / (2 * sqrt_t) - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365.0
    else:
        daily_theta = (
            -S * nd1 * sigma / (2 * sqrt_t) + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365.0

    return Decimal(str(daily_theta)).quantize(
        Decimal('0.000001'), rounding=ROUND_HALF_UP
    )


def estimate_volatility(
    close_prices: list[Decimal], lookback: int
) -> Decimal:
    """Estimate implied volatility from historical close prices.

    Uses the standard deviation of log returns, annualized assuming 252
    trading days per year.

    Args:
        close_prices: Sequence of closing prices as ``Decimal``.
        lookback: Number of recent prices to consider.

    Returns:
        Annualized volatility as a ``Decimal`` (minimum 0.01).
    """
    if close_prices is None or len(close_prices) < 2:
        return Decimal('0.200000')  # default 20% IV

    start = max(0, len(close_prices) - lookback)
    count = len(close_prices) - start - 1
    if count < 1:
        return Decimal('0.200000')

    log_returns: list[float] = []
    for i in range(count):
        prev = float(close_prices[start + i])
        curr = float(close_prices[start + i + 1])
        if prev > 0:
            log_returns.append(math.log(curr / prev))
        else:
            log_returns.append(0.0)

    mean = sum(log_returns) / len(log_returns)

    sum_sq_diff = sum((lr - mean) ** 2 for lr in log_returns)
    daily_vol = math.sqrt(sum_sq_diff / len(log_returns))

    # Annualize
    annual_vol = daily_vol * math.sqrt(252)
    return Decimal(str(max(annual_vol, 0.01))).quantize(
        Decimal('0.000001'), rounding=ROUND_HALF_UP
    )


def intrinsic_value(
    option_type: OptionType, spot: Decimal, strike: Decimal
) -> Decimal:
    """Calculate the intrinsic value of an option.

    For calls: max(spot - strike, 0).  For puts: max(strike - spot, 0).
    """
    if option_type == OptionType.CALL:
        diff = spot - strike
    else:
        diff = strike - spot
    return max(diff, Decimal('0')).quantize(
        Decimal('0.0001'), rounding=ROUND_HALF_UP
    )
