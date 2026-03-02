"""OptionsStateEngine -- IV Surface, GEX, OI, and Volume metrics.

Computes ~40 daily metrics from raw options chain data, including:
- Implied volatility surface (ATM IV by tenor, skew, term structure)
- Gamma exposure (net GEX, flip price, call/put walls)
- Open interest flow (P/C ratios, concentration, weighted strike)
- Volume and premium (call/put volumes, premium imbalance)

Uses float-based Black-Scholes for speed (no Decimal overhead).
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Contract multiplier (1 option contract = 100 shares)
_MULT = 100


# ------------------------------------------------------------------
# Black-Scholes helpers (float-based, module-level for speed)
# ------------------------------------------------------------------


def _bs_price(is_call: bool, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes option price.

    Parameters
    ----------
    is_call:
        True for call, False for put.
    S:
        Spot price of the underlying.
    K:
        Strike price.
    T:
        Time to expiry in years.
    r:
        Risk-free rate (annualized).
    sigma:
        Volatility (annualized).

    Returns
    -------
    Theoretical option price.
    """
    if T <= 0 or sigma <= 0:
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(is_call: bool, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt_t)

    if is_call:
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt_t)

    # N'(d1) = standard normal PDF
    nd1_prime = math.exp(-d1 * d1 / 2.0) / math.sqrt(2.0 * math.pi)
    return nd1_prime / (S * sigma * sqrt_t)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (sensitivity to volatility)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt_t)
    nd1_prime = math.exp(-d1 * d1 / 2.0) / math.sqrt(2.0 * math.pi)
    return S * nd1_prime * sqrt_t


def _implied_vol(
    is_call: bool,
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """Implied volatility via Newton-Raphson.

    Parameters
    ----------
    is_call:
        True for call, False for put.
    S, K, T, r:
        Spot, strike, time-to-expiry (years), risk-free rate.
    market_price:
        Observed market price of the option.
    max_iter:
        Maximum Newton-Raphson iterations.
    tol:
        Convergence tolerance.

    Returns
    -------
    Implied volatility clamped to [0.01, 10.0].
    """
    if market_price <= 0 or T <= 0:
        return 0.5

    # Initial guess
    sigma = 0.5

    for _ in range(max_iter):
        price = _bs_price(is_call, S, K, T, r, sigma)
        vega = _bs_vega(S, K, T, r, sigma)

        if vega < 1e-12:
            break

        diff = price - market_price
        if abs(diff) < tol:
            break

        sigma = sigma - diff / vega
        # Clamp to valid range
        sigma = max(0.01, min(sigma, 10.0))

    return max(0.01, min(sigma, 10.0))


# ------------------------------------------------------------------
# OptionsStateEngine
# ------------------------------------------------------------------


class OptionsStateEngine:
    """Computes daily options state metrics from raw chain data.

    Parameters
    ----------
    spot_prices:
        Mapping of date strings (``YYYY-MM-DD``) to spot prices.
        Used by :meth:`build` to look up the underlying price for
        each trading day.
    """

    RISK_FREE_RATE: float = 0.05

    def __init__(self, spot_prices: dict[str, float] | None = None) -> None:
        self.spot_prices = spot_prices or {}
        self._iv_history: list[float] = []

    # ------------------------------------------------------------------
    # IV Surface
    # ------------------------------------------------------------------

    def _compute_iv_surface(
        self, df: pd.DataFrame, trade_date: date, spot: float
    ) -> dict[str, float]:
        """Compute implied volatility surface metrics.

        Returns 8 metrics: atm_iv_30d, atm_iv_60d, atm_iv_90d,
        iv_skew_25d_30, iv_term_slope, iv_percentile, iv_change_1d,
        iv_change_5d.
        """
        r = self.RISK_FREE_RATE
        metrics: dict[str, float] = {}

        # Compute ATM IV for each tenor bucket
        for tenor_days, label in [(30, "30d"), (60, "60d"), (90, "90d")]:
            atm_iv = self._compute_atm_iv(df, trade_date, spot, tenor_days, r)
            metrics[f"atm_iv_{label}"] = atm_iv

        # IV skew: 25-delta skew for 30-day tenor
        metrics["iv_skew_25d_30"] = self._compute_skew(df, trade_date, spot, 30)

        # IV term structure slope (90d - 30d)
        iv_30 = metrics.get("atm_iv_30d", 0.0)
        iv_90 = metrics.get("atm_iv_90d", 0.0)
        metrics["iv_term_slope"] = iv_90 - iv_30

        # IV percentile (rank vs history)
        current_iv = iv_30 if iv_30 > 0 else 0.5
        self._iv_history.append(current_iv)
        metrics["iv_percentile"] = self._iv_percentile(
            current_iv, self._iv_history
        )

        # iv_change_1d and iv_change_5d -- filled by build() from prior days
        metrics["iv_change_1d"] = 0.0
        metrics["iv_change_5d"] = 0.0

        return metrics

    def _compute_atm_iv(
        self,
        df: pd.DataFrame,
        trade_date: date,
        spot: float,
        tenor_days: int,
        r: float,
    ) -> float:
        """Compute ATM implied vol for options near a given tenor."""
        target_exp = trade_date + timedelta(days=tenor_days)

        # Parse expiration dates
        exp_dates = pd.to_datetime(df["expiration"])
        days_to_exp = (exp_dates - pd.Timestamp(trade_date)).dt.days

        # Find options closest to target tenor (within +/- 15 days)
        tenor_mask = (days_to_exp >= tenor_days - 15) & (
            days_to_exp <= tenor_days + 15
        )
        sub = df[tenor_mask].copy()

        if sub.empty:
            return 0.5  # Default IV

        # Find strikes closest to ATM
        sub = sub.copy()
        sub.loc[:, "_moneyness"] = abs(sub["strike"] - spot) / spot

        # ATM: within 10% of spot
        atm_mask = sub["_moneyness"] <= 0.10
        atm = sub[atm_mask]

        if atm.empty:
            # Use closest strike
            atm = sub.nsmallest(4, "_moneyness")

        ivs = []
        for _, row in atm.iterrows():
            is_call = row["option_type"] == "call"
            K = row["strike"]
            T_years = max(
                (pd.Timestamp(row["expiration"]) - pd.Timestamp(trade_date)).days / 365.0,
                1 / 365.0,
            )
            market_price = row["close"]
            if market_price > 0 and K > 0:
                iv = _implied_vol(is_call, spot, K, T_years, r, market_price)
                if 0.01 < iv < 10.0:
                    ivs.append(iv)

        return float(np.mean(ivs)) if ivs else 0.5

    def _compute_skew(
        self, df: pd.DataFrame, trade_date: date, spot: float, tenor_days: int
    ) -> float:
        """Compute 25-delta skew for a given tenor.

        Skew = IV(25-delta put) - IV(25-delta call).
        Positive skew means puts are more expensive (typical equity skew).
        """
        r = self.RISK_FREE_RATE
        exp_dates = pd.to_datetime(df["expiration"])
        days_to_exp = (exp_dates - pd.Timestamp(trade_date)).dt.days

        tenor_mask = (days_to_exp >= tenor_days - 15) & (
            days_to_exp <= tenor_days + 15
        )
        sub = df[tenor_mask].copy()

        if sub.empty:
            return 0.0

        # Compute IV and delta for each option
        put_ivs: list[tuple[float, float]] = []  # (abs_delta, iv)
        call_ivs: list[tuple[float, float]] = []

        for _, row in sub.iterrows():
            is_call = row["option_type"] == "call"
            K = row["strike"]
            T_years = max(
                (pd.Timestamp(row["expiration"]) - pd.Timestamp(trade_date)).days / 365.0,
                1 / 365.0,
            )
            market_price = row["close"]

            if market_price <= 0 or K <= 0:
                continue

            iv = _implied_vol(is_call, spot, K, T_years, r, market_price)
            delta = _bs_delta(is_call, spot, K, T_years, r, iv)

            if is_call:
                call_ivs.append((delta, iv))
            else:
                put_ivs.append((abs(delta), iv))

        # Find options closest to 25-delta
        target_delta = 0.25

        put_25d_iv = _find_closest_delta_iv(put_ivs, target_delta)
        call_25d_iv = _find_closest_delta_iv(call_ivs, target_delta)

        if put_25d_iv is not None and call_25d_iv is not None:
            return put_25d_iv - call_25d_iv
        return 0.0

    def _iv_percentile(self, current: float, history: list[float]) -> float:
        """Compute IV percentile: fraction of history values below current."""
        if not history:
            return 0.5
        below = sum(1 for v in history if v < current)
        return below / len(history)

    # ------------------------------------------------------------------
    # GEX
    # ------------------------------------------------------------------

    def _compute_gex(self, df: pd.DataFrame, spot: float) -> dict[str, float]:
        """Compute Gamma Exposure metrics.

        Returns 5 metrics: net_gex, gex_flip_price, call_wall,
        put_wall, gex_concentration.

        GEX convention: Dealers are short calls (sold to retail) and
        long puts. So:
        - Call GEX = gamma * OI * 100 * spot (positive)
        - Put GEX = -gamma * OI * 100 * spot (negative from dealer perspective)
        """
        r = self.RISK_FREE_RATE

        # Compute gamma and GEX per strike
        strike_data: dict[float, dict[str, float]] = {}

        for _, row in df.iterrows():
            K = row["strike"]
            is_call = row["option_type"] == "call"
            oi = row.get("open_interest", 0)
            if oi <= 0 or K <= 0:
                continue

            T_years = max(
                (pd.Timestamp(row["expiration"]) - pd.Timestamp(row["date"])).days / 365.0,
                1 / 365.0,
            )

            # Estimate IV from market price for gamma calculation
            market_price = row["close"]
            if market_price <= 0:
                continue

            iv = _implied_vol(is_call, spot, K, T_years, r, market_price)
            gamma = _bs_gamma(spot, K, T_years, r, iv)

            if K not in strike_data:
                strike_data[K] = {
                    "call_gex": 0.0,
                    "put_gex": 0.0,
                    "call_oi_gamma": 0.0,
                    "put_oi_gamma": 0.0,
                }

            if is_call:
                gex = gamma * oi * _MULT * spot
                strike_data[K]["call_gex"] += gex
                strike_data[K]["call_oi_gamma"] += gamma * oi
            else:
                gex = -gamma * oi * _MULT * spot
                strike_data[K]["put_gex"] += gex
                strike_data[K]["put_oi_gamma"] += gamma * oi

        if not strike_data:
            return {
                "net_gex": 0.0,
                "gex_flip_price": 0.0,
                "call_wall": spot,
                "put_wall": spot,
                "gex_concentration": 0.0,
            }

        # Net GEX = sum of call_gex + put_gex at all strikes
        net_gex = sum(
            d["call_gex"] + d["put_gex"] for d in strike_data.values()
        )

        # Call wall: strike with highest call OI * gamma
        call_wall_strike = max(
            strike_data.keys(),
            key=lambda k: strike_data[k]["call_oi_gamma"],
        )

        # Put wall: strike with highest put OI * gamma
        put_wall_strike = max(
            strike_data.keys(),
            key=lambda k: strike_data[k]["put_oi_gamma"],
        )

        # Ensure call_wall >= put_wall (swap if needed)
        if call_wall_strike < put_wall_strike:
            call_wall_strike, put_wall_strike = put_wall_strike, call_wall_strike

        # GEX flip: strike where cumulative GEX crosses zero
        sorted_strikes = sorted(strike_data.keys())
        gex_flip = 0.0
        cumulative = 0.0
        prev_cum = 0.0

        for i, k in enumerate(sorted_strikes):
            prev_cum = cumulative
            cumulative += strike_data[k]["call_gex"] + strike_data[k]["put_gex"]

            if i > 0 and prev_cum * cumulative < 0:
                # Linear interpolation
                prev_k = sorted_strikes[i - 1]
                ratio = abs(prev_cum) / (abs(prev_cum) + abs(cumulative))
                gex_flip = prev_k + (k - prev_k) * ratio
                break

        # GEX concentration: fraction of total |GEX| in top 5 strikes
        abs_gex_per_strike = {
            k: abs(d["call_gex"] + d["put_gex"])
            for k, d in strike_data.items()
        }
        total_abs_gex = sum(abs_gex_per_strike.values())
        top5 = sorted(abs_gex_per_strike.values(), reverse=True)[:5]
        gex_concentration = sum(top5) / total_abs_gex if total_abs_gex > 0 else 0.0

        return {
            "net_gex": float(net_gex),
            "gex_flip_price": float(gex_flip),
            "call_wall": float(call_wall_strike),
            "put_wall": float(put_wall_strike),
            "gex_concentration": float(gex_concentration),
        }

    # ------------------------------------------------------------------
    # OI Flow
    # ------------------------------------------------------------------

    def _compute_oi_flow(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute open interest flow metrics.

        Returns 8 metrics: total_call_oi, total_put_oi, pc_oi_ratio,
        pc_oi_ratio_change, oi_weighted_strike, oi_concentration_top5,
        near_term_oi_pct, oi_skew_delta.
        """
        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        total_call_oi = float(calls["open_interest"].sum())
        total_put_oi = float(puts["open_interest"].sum())

        pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

        # OI-weighted strike
        total_oi = total_call_oi + total_put_oi
        if total_oi > 0:
            oi_weighted_strike = float(
                (df["strike"] * df["open_interest"]).sum() / total_oi
            )
        else:
            oi_weighted_strike = 0.0

        # OI concentration: top 5 strikes as fraction of total
        strike_oi = df.groupby("strike")["open_interest"].sum()
        if len(strike_oi) > 0 and total_oi > 0:
            top5_oi = strike_oi.nlargest(5).sum()
            oi_concentration_top5 = float(top5_oi / total_oi)
        else:
            oi_concentration_top5 = 0.0

        # Near-term OI: fraction with expiration <= 30 days
        if "expiration" in df.columns and "date" in df.columns:
            exp_dates = pd.to_datetime(df["expiration"])
            trade_dates = pd.to_datetime(df["date"])
            days_to_exp = (exp_dates - trade_dates).dt.days
            near_term_mask = days_to_exp <= 30
            near_term_oi = float(df.loc[near_term_mask, "open_interest"].sum())
            near_term_oi_pct = near_term_oi / total_oi if total_oi > 0 else 0.0
        else:
            near_term_oi_pct = 0.0

        # OI skew delta: call OI above spot vs put OI below spot
        # (this is a rough directional indicator)
        oi_skew_delta = 0.0

        # pc_oi_ratio_change: filled by build() from prior day
        return {
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "pc_oi_ratio": pc_oi_ratio,
            "pc_oi_ratio_change": 0.0,
            "oi_weighted_strike": oi_weighted_strike,
            "oi_concentration_top5": oi_concentration_top5,
            "near_term_oi_pct": near_term_oi_pct,
            "oi_skew_delta": oi_skew_delta,
        }

    # ------------------------------------------------------------------
    # Volume & Premium
    # ------------------------------------------------------------------

    def _compute_volume_premium(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute volume and premium metrics.

        Returns 6 metrics: call_volume, put_volume, pc_volume_ratio,
        call_premium, put_premium, premium_imbalance.
        """
        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        call_volume = float(calls["volume"].sum())
        put_volume = float(puts["volume"].sum())

        pc_volume_ratio = put_volume / call_volume if call_volume > 0 else 0.0

        # Premium = volume * close * 100 (dollar value of options traded)
        call_premium = float((calls["volume"] * calls["close"] * _MULT).sum())
        put_premium = float((puts["volume"] * puts["close"] * _MULT).sum())

        total_premium = call_premium + put_premium
        if total_premium > 0:
            premium_imbalance = (call_premium - put_premium) / total_premium
        else:
            premium_imbalance = 0.0

        return {
            "call_volume": call_volume,
            "put_volume": put_volume,
            "pc_volume_ratio": pc_volume_ratio,
            "call_premium": call_premium,
            "put_premium": put_premium,
            "premium_imbalance": premium_imbalance,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compute_day(
        self, df: pd.DataFrame, trade_date: date, spot: float
    ) -> dict[str, Any]:
        """Compute all metrics for a single trading day.

        Parameters
        ----------
        df:
            Options chain data for this day.
        trade_date:
            The trading date.
        spot:
            Underlying spot price.

        Returns
        -------
        Dictionary of ~27 metrics plus the trade_date.
        """
        metrics: dict[str, Any] = {"trade_date": trade_date}

        # IV Surface (8 metrics)
        metrics.update(self._compute_iv_surface(df, trade_date, spot))

        # GEX (5 metrics)
        metrics.update(self._compute_gex(df, spot))

        # OI Flow (8 metrics)
        metrics.update(self._compute_oi_flow(df))

        # Volume & Premium (6 metrics)
        metrics.update(self._compute_volume_premium(df))

        return metrics

    def build(
        self,
        bars_dir: Path | None = None,
        bars_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the full state DataFrame from Parquet dir or DataFrame.

        Parameters
        ----------
        bars_dir:
            Path to a directory of monthly Parquet files.
        bars_df:
            Alternatively, a pre-loaded DataFrame of options bars.

        Returns
        -------
        DataFrame with one row per trading day, containing all metrics.
        """
        # Load data
        if bars_df is not None:
            df = bars_df
        elif bars_dir is not None:
            parquet_files = sorted(bars_dir.glob("*.parquet"))
            if not parquet_files:
                return pd.DataFrame()
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Group by trade date
        dates = sorted(df["date"].unique())
        rows: list[dict[str, Any]] = []

        # Reset IV history for fresh build
        self._iv_history = []

        for d_str in dates:
            d_str = str(d_str)
            trade_date = date.fromisoformat(d_str[:10])
            day_df = df[df["date"] == d_str]

            # Look up spot price
            spot = self.spot_prices.get(d_str[:10])
            if spot is None:
                logger.warning("No spot price for %s, skipping", d_str)
                continue

            day_metrics = self.compute_day(day_df, trade_date, spot)
            rows.append(day_metrics)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)

        # Fill delta fields from prior days
        if len(result) > 1:
            result["iv_change_1d"] = result["atm_iv_30d"].diff(1).fillna(0.0)
            if len(result) >= 6:
                result["iv_change_5d"] = result["atm_iv_30d"].diff(5).fillna(0.0)
            result["pc_oi_ratio_change"] = result["pc_oi_ratio"].diff(1).fillna(0.0)

        return result


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _find_closest_delta_iv(
    delta_iv_pairs: list[tuple[float, float]], target_delta: float
) -> float | None:
    """Find the IV of the option closest to the target delta."""
    if not delta_iv_pairs:
        return None

    closest = min(delta_iv_pairs, key=lambda x: abs(x[0] - target_delta))
    return closest[1]
