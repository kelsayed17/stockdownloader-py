"""Daily strategy wrapper bridging signal stacks to backtest engines.

Evaluates a signal stack on single-timeframe data and produces
:class:`~stockdownloader.strategy.trading_strategy.Signal` (BUY/SELL/HOLD).

Usage::

    config = StackConfig(buy_threshold=0.3, sell_threshold=0.3)
    specs = [TimeframeSignalSpec(rsi_gen, Timeframe.M5, 1.0)]
    daily = StackedDailyStrategy("MyStack", config, specs)
    signal = daily.evaluate(data, index)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.strategy.signals.multi_timeframe_aligner import (
    AlignedSignal,
    TimeframeSignalSpec,
)
from stockdownloader.strategy.signals.stacked_signal_engine import (
    StackConfig,
    StackedSignalEngine,
)
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.indicators.hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.core.models.price import PriceData


class StackedDailyStrategy(TradingStrategy):
    """Evaluates a signal stack on single-timeframe data.

    All specs are evaluated on the provided data (assumed to be a
    single timeframe, typically M5 after aggregation or daily).
    Compatible with :class:`DailyToIntradayAdapter` for intraday backtests,
    or :class:`BacktestEngine` for daily backtests.

    Parameters
    ----------
    name:
        Display name for the strategy.
    config:
        Stacking engine configuration.
    specs:
        Signal specs (timeframe field is ignored — all evaluate on
        the provided data).
    """

    def __init__(
        self,
        name: str,
        config: StackConfig,
        specs: list[TimeframeSignalSpec],
        hub: IndicatorHub | None = None,
    ) -> None:
        self._name = name
        self._config = config
        self._specs = specs
        self._engine = StackedSignalEngine(config)
        self._hub = hub or IndicatorHub()

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.warmup_period:
            return Signal.HOLD

        # Evaluate each generator directly on the provided data
        aligned: list[AlignedSignal] = []
        for spec in self._specs:
            gen = spec.generator
            result = gen.evaluate(data, current_index, self._hub)
            cat_name = gen.category
            aligned.append(AlignedSignal(
                result=result,
                timeframe=spec.timeframe,
                generator_name=gen.name,
                category=cat_name,
                weight=spec.weight,
                htf_bar_index=current_index,
                htf_bar_count=len(data),
            ))

        stack_result = self._engine.evaluate(aligned)

        if stack_result.buy_signal:
            return Signal.BUY
        if stack_result.sell_signal:
            return Signal.SELL
        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        if not self._specs:
            return 0
        return max(spec.generator.warmup_period for spec in self._specs)
