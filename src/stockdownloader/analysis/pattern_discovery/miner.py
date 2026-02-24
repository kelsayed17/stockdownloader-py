"""Pattern mining engine.

Scans historical price data for recurring multi-bar patterns (2-5 bars)
and measures follow-through returns at multiple horizons.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

from scipy.stats import t as t_dist

from stockdownloader.analysis.pattern_discovery.models import (
    PatternKey,
    PatternOutcome,
    PatternStats,
)

if TYPE_CHECKING:
    from stockdownloader.analysis.pattern_encoder import BarEncoder, BarFeatures
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.regime.detector import MarketRegime


class PatternMiner:
    """Scans price data for recurring N-bar patterns and measures follow-through.

    Parameters
    ----------
    encoder:
        Bar encoder for converting bars to categorical features.
    pattern_lengths:
        N-gram sizes to scan (default: 2, 3, 4, 5).
    horizons:
        Bars ahead to measure follow-through (default: 1, 3, 5, 10, 20).
    """

    DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
    DEFAULT_PATTERN_LENGTHS = (2, 3)

    def __init__(
        self,
        encoder: BarEncoder,
        *,
        pattern_lengths: tuple[int, ...] = DEFAULT_PATTERN_LENGTHS,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    ) -> None:
        self._encoder = encoder
        self._pattern_lengths = pattern_lengths
        self._horizons = horizons

    def mine(
        self,
        data: list[IntradayPriceData],
        start: int,
        end: int,
        *,
        regime_map: dict[str, MarketRegime] | None = None,
    ) -> dict[PatternKey, PatternStats]:
        """Mine patterns from ``data[start:end]``.

        Algorithm:
        1. Pre-encode all bars into BarFeatures (single forward pass).
        2. For each pattern length N: slide N-bar window, build key,
           measure follow-through at each horizon.
        3. Session boundary enforcement: stop measuring when
           ``trading_date`` changes.
        4. Group outcomes by key, compute statistics.

        Parameters
        ----------
        data:
            Full price data list.
        start:
            First bar index to mine from (after warmup).
        end:
            End index (exclusive).
        regime_map:
            Optional pre-computed ``{date_str: MarketRegime}`` mapping.

        Returns
        -------
        Dict mapping PatternKey to PatternStats.
        """
        end = min(end, len(data))
        if start >= end:
            return {}

        # Step 1: Pre-encode all bars
        features: list[BarFeatures | None] = [None] * len(data)
        bar_of_day_map: list[int] = [0] * len(data)

        current_date = ""
        day_bar = 0

        for i in range(start, end):
            bar = data[i]
            td = bar.date[:10] if hasattr(bar, "date") else ""
            if td != current_date:
                current_date = td
                day_bar = 0
            day_bar += 1
            bar_of_day_map[i] = day_bar
            features[i] = self._encoder.encode(data, i)

        # Step 2: Slide windows and collect outcomes
        raw: dict[PatternKey, list[PatternOutcome]] = defaultdict(list)
        max_horizon = max(self._horizons) if self._horizons else 20

        for n in self._pattern_lengths:
            for i in range(start, end - n + 1):
                # Build pattern key
                key_parts: list[BarFeatures] = []
                valid = True
                for j in range(n):
                    f = features[i + j]
                    if f is None:
                        valid = False
                        break
                    key_parts.append(f)

                if not valid:
                    continue

                pattern_key = tuple(key_parts)
                pattern_end = i + n - 1
                entry_date = data[pattern_end].date[:10]

                # Measure follow-through
                outcome = self._measure_outcome(
                    data, pattern_end, entry_date, max_horizon,
                    bar_of_day_map, regime_map,
                )
                if outcome is not None:
                    raw[pattern_key].append(outcome)

        # Step 3: Compute statistics
        result: dict[PatternKey, PatternStats] = {}
        for key, outcomes in raw.items():
            stats = self._compute_stats(key, outcomes)
            result[key] = stats

        return result

    def _measure_outcome(
        self,
        data: list[IntradayPriceData],
        pattern_end_index: int,
        entry_date: str,
        max_horizon: int,
        bar_of_day_map: list[int],
        regime_map: dict[str, MarketRegime] | None,
    ) -> PatternOutcome | None:
        """Measure what happened after the pattern ended."""
        entry_close = float(data[pattern_end_index].close)
        if entry_close == 0:
            return None

        returns: dict[int, float] = {}
        mae = 0.0
        mfe = 0.0

        # Scan forward up to max_horizon, stopping at session boundary
        actual_max = 0
        for j in range(1, max_horizon + 1):
            idx = pattern_end_index + j
            if idx >= len(data):
                break
            future_date = data[idx].date[:10]
            if future_date != entry_date:
                break  # session boundary -- stop

            actual_max = j
            bar = data[idx]
            # Track MAE/MFE using high/low
            low_pct = (float(bar.low) - entry_close) / entry_close * 100.0
            high_pct = (float(bar.high) - entry_close) / entry_close * 100.0
            if low_pct < mae:
                mae = low_pct
            if high_pct > mfe:
                mfe = high_pct

        # Record returns at each horizon that we could measure
        for h in self._horizons:
            target_idx = pattern_end_index + h
            if target_idx >= len(data):
                break
            if data[target_idx].date[:10] != entry_date:
                break
            future_close = float(data[target_idx].close)
            ret = (future_close - entry_close) / entry_close * 100.0
            returns[h] = ret

        if not returns:
            return None  # couldn't measure any horizon

        # Build context
        bod = bar_of_day_map[pattern_end_index]
        regime = None
        if regime_map is not None:
            regime = regime_map.get(data[pattern_end_index].date)

        context = self._encoder.encode_context(
            data, pattern_end_index, bod, regime=regime,
        )

        return PatternOutcome(
            bar_index=pattern_end_index,
            returns=returns,
            mae=mae,
            mfe=mfe,
            context=context,
        )

    def _compute_stats(
        self,
        key: PatternKey,
        outcomes: list[PatternOutcome],
    ) -> PatternStats:
        """Compute aggregate statistics for a pattern."""
        stats = PatternStats(
            key=key,
            occurrences=len(outcomes),
            outcomes=outcomes,
        )

        for horizon in self._horizons:
            horizon_returns = [
                o.returns[horizon]
                for o in outcomes
                if horizon in o.returns
            ]
            n = len(horizon_returns)
            if n < 2:
                continue

            mean = statistics.mean(horizon_returns)
            stdev = statistics.stdev(horizon_returns)
            median = statistics.median(horizon_returns)

            stats.avg_return[horizon] = mean
            stats.median_return[horizon] = median
            stats.win_rate[horizon] = sum(1 for r in horizon_returns if r > 0) / n
            stats.std_return[horizon] = stdev

            # T-test: is mean significantly different from 0?
            if stdev > 0 and n >= 5:
                t = mean / (stdev / math.sqrt(n))
                stats.t_stat[horizon] = t
                stats.p_value[horizon] = float(
                    2 * t_dist.sf(abs(t), df=n - 1)
                )

            # Walk-forward stability: 3-fold expanding window
            #   Fold 1: train [0:33%], test [33%:66%]
            #   Fold 2: train [0:50%], test [50%:75%]
            #   Fold 3: train [0:67%], test [67%:100%]
            fold_splits = [
                (n // 3, 2 * n // 3),
                (n // 2, 3 * n // 4),
                (2 * n // 3, n),
            ]
            horizon_returns_ordered = [
                o.returns[horizon]
                for o in outcomes
                if horizon in o.returns
            ]
            fold_wrs: list[float] = []
            for split_start, split_end in fold_splits:
                test_rets = horizon_returns_ordered[split_start:split_end]
                if test_rets:
                    fold_wr = sum(1 for r in test_rets if r > 0) / len(test_rets)
                    fold_wrs.append(fold_wr)

            if fold_wrs:
                stats.fold_win_rates[horizon] = fold_wrs

            # Backward-compat: first/second half from first/last fold
            mid = len(outcomes) // 2
            first_half = outcomes[:mid]
            second_half = outcomes[mid:]

            fh_returns = [
                o.returns[horizon]
                for o in first_half
                if horizon in o.returns
            ]
            sh_returns = [
                o.returns[horizon]
                for o in second_half
                if horizon in o.returns
            ]
            if fh_returns:
                stats.first_half_wr[horizon] = (
                    sum(1 for r in fh_returns if r > 0) / len(fh_returns)
                )
            if sh_returns:
                stats.second_half_wr[horizon] = (
                    sum(1 for r in sh_returns if r > 0) / len(sh_returns)
                )

        # MAE / MFE averages
        mae_vals = [o.mae for o in outcomes]
        mfe_vals = [o.mfe for o in outcomes]
        if mae_vals:
            stats.avg_mae = statistics.mean(mae_vals)
        if mfe_vals:
            stats.avg_mfe = statistics.mean(mfe_vals)

        return stats
