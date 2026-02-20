"""Cross-timeframe signal alignment.

Evaluates :class:`AtomicSignalGenerator` instances on data from multiple
timeframes, using :class:`TimeframeAggregator` to derive higher-timeframe
(HTF) bars from the underlying 5-minute data.

Each HTF signal is evaluated only on **completed** candles (via
:meth:`TimeframeAggregator.as_price_data_through`) to prevent look-ahead bias.
Results are cached per ``(generator_name, timeframe)`` to avoid re-evaluating
when multiple 5-minute bars fall within the same HTF candle.

Usage::

    from stockdownloader.strategy.signals.multi_timeframe_aligner import (
        MultiTimeframeAligner, TimeframeSignalSpec,
    )

    aligner = MultiTimeframeAligner(aggregator)
    specs = [
        TimeframeSignalSpec(generator=rsi_gen, timeframe=Timeframe.M5, weight=1.0),
        TimeframeSignalSpec(generator=macd_gen, timeframe=Timeframe.H1, weight=1.5),
    ]
    aligned = aligner.get_aligned_signals(specs, current_5m_index, hub_m5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalResult,
)
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.timeframe_aggregator import Timeframe

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData
    from stockdownloader.util.timeframe_aggregator import TimeframeAggregator


@dataclass(frozen=True, slots=True)
class TimeframeSignalSpec:
    """Binds a generator to a specific timeframe and weight.

    Attributes
    ----------
    generator:
        The atomic signal generator to evaluate.
    timeframe:
        The timeframe on which to evaluate (M5 uses raw data, others
        use aggregated bars from :class:`TimeframeAggregator`).
    weight:
        Relative weight in the stacking engine (default 1.0).
    """

    generator: AtomicSignalGenerator
    timeframe: Timeframe
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class AlignedSignal:
    """A signal result tagged with timeframe metadata.

    Attributes
    ----------
    result:
        The signal evaluation result.
    timeframe:
        Which timeframe produced this result.
    generator_name:
        Identifier of the generator that produced the signal.
    category:
        Signal category (momentum, trend, volatility, volume).
    weight:
        Weight from the spec for use in aggregation.
    htf_bar_index:
        Index of the HTF bar that was evaluated (-1 for M5).
    htf_bar_count:
        Total number of HTF bars available at evaluation time.
    """

    result: SignalResult
    timeframe: Timeframe
    generator_name: str
    category: str
    weight: float
    htf_bar_index: int = -1
    htf_bar_count: int = 0


class MultiTimeframeAligner:
    """Evaluates signal generators across multiple timeframes.

    For M5 specs, generators are evaluated directly on the raw 5-minute
    data using the shared hub.  For HTF specs (M15, M30, H1, H4, DAILY),
    aggregated bars are obtained from the :class:`TimeframeAggregator` and
    generators are evaluated on the last completed HTF bar.

    **Caching**: HTF results are cached by ``(generator_name, timeframe,
    htf_bar_count)`` to avoid re-evaluation when multiple 5-minute bars
    fall within the same HTF candle.

    Parameters
    ----------
    aggregator:
        The :class:`TimeframeAggregator` holding the 5-minute data.
    """

    __slots__ = ("_aggregator", "_htf_hubs", "_htf_cache", "_hub_factory")

    def __init__(
        self,
        aggregator: TimeframeAggregator,
        hub_factory: Callable[[], IndicatorHub] | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._hub_factory = hub_factory or IndicatorHub
        # Per-timeframe IndicatorHub instances (HTF data is different list)
        self._htf_hubs: dict[Timeframe, IndicatorHub] = {}
        # Cache: (generator_name, timeframe, htf_bar_count) → AlignedSignal
        self._htf_cache: dict[tuple[str, Timeframe, int], AlignedSignal] = {}

    def get_aligned_signals(
        self,
        specs: Sequence[TimeframeSignalSpec],
        current_5m_index: int,
        hub_m5: IndicatorHub,
    ) -> list[AlignedSignal]:
        """Evaluate all specs and return aligned signals.

        Parameters
        ----------
        specs:
            List of (generator, timeframe, weight) specifications.
        current_5m_index:
            The current bar index in the 5-minute data.
        hub_m5:
            Shared :class:`IndicatorHub` for M5 evaluations.

        Returns
        -------
        list[AlignedSignal]
            One aligned signal per spec.  Specs whose HTF data has
            insufficient warmup produce neutral results.
        """
        results: list[AlignedSignal] = []
        for spec in specs:
            if spec.timeframe == Timeframe.M5:
                aligned = self._evaluate_m5(spec, current_5m_index, hub_m5)
            else:
                aligned = self._evaluate_htf(spec, current_5m_index)
            results.append(aligned)
        return results

    def clear_cache(self) -> None:
        """Clear all HTF caches (e.g. between backtest runs)."""
        self._htf_cache.clear()
        self._htf_hubs.clear()

    # ------------------------------------------------------------------
    # Internal evaluation methods
    # ------------------------------------------------------------------

    def _evaluate_m5(
        self,
        spec: TimeframeSignalSpec,
        index: int,
        hub: IndicatorHub,
    ) -> AlignedSignal:
        """Evaluate a generator directly on M5 data."""
        # Get 5m data as PriceData through current index
        m5_data = self._aggregator.as_price_data_through(Timeframe.M5, index)
        if not m5_data:
            return self._make_neutral(spec)

        gen = spec.generator
        bar_idx = len(m5_data) - 1
        result = gen.evaluate(m5_data, bar_idx, hub)
        return AlignedSignal(
            result=result,
            timeframe=Timeframe.M5,
            generator_name=gen.name,
            category=gen.category,
            weight=spec.weight,
            htf_bar_index=bar_idx,
            htf_bar_count=len(m5_data),
        )

    def _evaluate_htf(
        self,
        spec: TimeframeSignalSpec,
        current_5m_index: int,
    ) -> AlignedSignal:
        """Evaluate a generator on aggregated HTF data."""
        tf = spec.timeframe
        gen = spec.generator

        # Get completed HTF bars through current 5m index
        htf_data = self._aggregator.as_price_data_through(tf, current_5m_index)
        if not htf_data:
            return self._make_neutral(spec)

        bar_count = len(htf_data)
        cache_key = (gen.name, tf, bar_count)

        # Return cached result if the same HTF bar is still current
        if cache_key in self._htf_cache:
            return self._htf_cache[cache_key]

        # Get or create per-timeframe hub
        if tf not in self._htf_hubs:
            self._htf_hubs[tf] = self._hub_factory()
        htf_hub = self._htf_hubs[tf]

        bar_idx = bar_count - 1
        warmup = gen.warmup_period

        if bar_idx < warmup:
            aligned = self._make_neutral(spec, htf_bar_index=bar_idx,
                                          htf_bar_count=bar_count)
        else:
            result = gen.evaluate(htf_data, bar_idx, htf_hub)
            aligned = AlignedSignal(
                result=result,
                timeframe=tf,
                generator_name=gen.name,
                category=gen.category,
                weight=spec.weight,
                htf_bar_index=bar_idx,
                htf_bar_count=bar_count,
            )

        self._htf_cache[cache_key] = aligned
        return aligned

    @staticmethod
    def _make_neutral(
        spec: TimeframeSignalSpec,
        htf_bar_index: int = -1,
        htf_bar_count: int = 0,
    ) -> AlignedSignal:
        """Create a neutral aligned signal for warmup / no-data cases."""
        return AlignedSignal(
            result=SignalResult.neutral(),
            timeframe=spec.timeframe,
            generator_name=spec.generator.name,
            category=spec.generator.category,
            weight=spec.weight,
            htf_bar_index=htf_bar_index,
            htf_bar_count=htf_bar_count,
        )
