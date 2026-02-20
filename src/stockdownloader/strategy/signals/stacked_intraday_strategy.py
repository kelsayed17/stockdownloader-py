"""Intraday strategy wrapper bridging multi-timeframe signal stacks to
:class:`~stockdownloader.backtest.intraday_backtest_engine.IntradayBacktestEngine`.

Lazily initializes a :class:`TimeframeAggregator` and
:class:`MultiTimeframeAligner` on first evaluation.  Produces
:class:`IntradaySignal` with ATR-based stop-loss and take-profit.

Usage::

    config = StackConfig(buy_threshold=0.3, sell_threshold=0.3)
    specs = [TimeframeSignalSpec(rsi_gen, Timeframe.M5, 1.0)]
    intraday = StackedIntradayStrategy("MyMTF", config, specs,
                                        sl_atr_mult=1.5, rr=1.5)
    signal = intraday.evaluate(data, index)
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import (
    IntradayAction,
    IntradaySignal,
    HOLD,
)
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.signals.multi_timeframe_aligner import (
    AlignedSignal,
    MultiTimeframeAligner,
    TimeframeSignalSpec,
)
from stockdownloader.strategy.signals.stacked_signal_engine import (
    StackConfig,
    StackedSignalEngine,
)
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.timeframe_aggregator import Timeframe, TimeframeAggregator
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData

class StackedIntradayStrategy(IntradayTradingStrategy):
    """Full multi-timeframe signal stacking strategy.

    Lazily initializes a :class:`TimeframeAggregator` and
    :class:`MultiTimeframeAligner` on first evaluation.  Produces
    :class:`IntradaySignal` with ATR-based stop-loss and take-profit.

    Parameters
    ----------
    name:
        Display name for the strategy.
    config:
        Stacking engine configuration.
    specs:
        Signal specs binding generators to timeframes and weights.
    sl_atr_mult:
        ATR multiplier for stop-loss distance (default: 1.5).
    rr:
        Risk/reward ratio for take-profit (default: 1.5).
    sl_cap:
        Maximum stop-loss distance in dollars (default: $2.00).
    allow_shorts:
        Whether SELL composites should enter short positions.
    atr_period:
        ATR lookback period in bars.
    """

    def __init__(
        self,
        name: str,
        config: StackConfig,
        specs: list[TimeframeSignalSpec],
        sl_atr_mult: Decimal = Decimal("1.5"),
        rr: Decimal = Decimal("1.5"),
        sl_cap: Decimal = Decimal("2.00"),
        allow_shorts: bool = False,
        atr_period: int = 14,
        hub: IndicatorHub | None = None,
    ) -> None:
        self._name = name
        self._config = config
        self._specs = specs
        self._engine = StackedSignalEngine(config)
        self._sl_atr_mult = sl_atr_mult
        self._rr = rr
        self._sl_cap = sl_cap
        self._allow_shorts = allow_shorts
        self._atr_period = atr_period

        # Performance: detect if all specs use M5 (fast-path avoids aligner)
        self._all_m5 = all(s.timeframe == Timeframe.M5 for s in specs)

        # Lazy-initialized on first evaluate
        self._hub_m5: IndicatorHub = hub or IndicatorHub()
        self._aggregator: TimeframeAggregator | None = None
        self._aligner: MultiTimeframeAligner | None = None
        self._data_id: int | None = None

        # Position tracking
        self._in_position = False
        self._position_is_long = False
        self._current_session: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def warmup_period(self) -> int:
        if not self._specs:
            return 0
        # HTF generators need more bars: warmup × timeframe factor
        max_warmup = 0
        for spec in self._specs:
            gen_warmup = spec.generator.warmup_period
            if spec.timeframe == Timeframe.M5:
                max_warmup = max(max_warmup, gen_warmup)
            else:
                factor = spec.timeframe.factor if spec.timeframe.factor > 0 else 78
                max_warmup = max(max_warmup, gen_warmup * factor)
        return max(max_warmup, self._atr_period + 1)

    def on_session_start(self, trading_date: str) -> None:
        self._current_session = trading_date
        # Reset position tracking — any overnight position was force-closed
        # by the engine at end-of-data or should not carry across sessions.
        self._in_position = False
        self._position_is_long = False

    # ------------------------------------------------------------------
    # Engine callbacks — keep internal state in sync with actual trades.
    # ------------------------------------------------------------------

    def on_position_opened(self, is_long: bool) -> None:
        self._in_position = True
        self._position_is_long = is_long

    def on_position_closed(self) -> None:
        self._in_position = False
        self._position_is_long = False

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        # Detect session boundary
        bar = data[current_index]
        trading_date = bar.trading_date
        if trading_date != self._current_session:
            self.on_session_start(trading_date)

        # Warmup check
        if current_index < self.warmup_period:
            return HOLD

        # Lazy-init hub (once per data list); re-create to clear stale caches
        did = id(data)
        if self._data_id != did:
            self._data_id = did
            self._hub_m5 = IndicatorHub()  # fresh hub for new data
            if not self._all_m5:
                self._aggregator = TimeframeAggregator(data)
                self._aligner = MultiTimeframeAligner(self._aggregator)

        # Fast path: all M5 specs — evaluate directly on data (no aligner)
        if self._all_m5:
            aligned = self._evaluate_m5_fast(data, current_index)
        else:
            assert self._aligner is not None
            aligned = self._aligner.get_aligned_signals(
                self._specs, current_index, self._hub_m5,
            )
        stack_result = self._engine.evaluate(aligned)

        # ATR-based stops
        atr_val = self._hub_m5.atr(data, current_index, self._atr_period)
        if atr_val <= ZERO:
            return HOLD

        sl_distance = min(
            (atr_val * self._sl_atr_mult).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            self._sl_cap,
        )
        tp_distance = (sl_distance * self._rr).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        entry_price = bar.close

        # --- BUY composite ---
        if stack_result.buy_signal:
            if self._in_position and not self._position_is_long:
                # Close existing short
                return IntradaySignal(
                    action=IntradayAction.EXIT,
                    reason=f"{self._name} BUY composite → exit short",
                )
            if not self._in_position:
                # Request long entry — actual flag update happens in
                # on_position_opened() callback from the engine.
                return IntradaySignal(
                    action=IntradayAction.ENTER_LONG,
                    mode=self._name,
                    stop_loss=entry_price - sl_distance,
                    take_profit=entry_price + tp_distance,
                    risk_per_share=sl_distance,
                    confluence_score=int(stack_result.fire_count),
                    max_score=stack_result.total_signals,
                    reason=(
                        f"{self._name} BUY (score={stack_result.composite_score:.2f}, "
                        f"fires={stack_result.fire_count}/{stack_result.total_signals})"
                    ),
                )
            return HOLD  # already long

        # --- SELL composite ---
        if stack_result.sell_signal:
            if self._in_position and self._position_is_long:
                # Close existing long
                return IntradaySignal(
                    action=IntradayAction.EXIT,
                    reason=f"{self._name} SELL composite → exit long",
                )
            if not self._in_position and self._allow_shorts:
                # Request short entry — actual flag update happens in
                # on_position_opened() callback from the engine.
                return IntradaySignal(
                    action=IntradayAction.ENTER_SHORT,
                    mode=self._name,
                    stop_loss=entry_price + sl_distance,
                    take_profit=entry_price - tp_distance,
                    risk_per_share=sl_distance,
                    confluence_score=int(stack_result.fire_count),
                    max_score=stack_result.total_signals,
                    reason=(
                        f"{self._name} SELL (score={stack_result.composite_score:.2f}, "
                        f"fires={stack_result.fire_count}/{stack_result.total_signals})"
                    ),
                )
            return HOLD

        return HOLD

    def _evaluate_m5_fast(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> list[AlignedSignal]:
        """Fast path for all-M5 specs: evaluate directly on data.

        Avoids :class:`MultiTimeframeAligner` and the expensive
        ``as_price_data_through()`` list-copy on every bar.
        """
        aligned: list[AlignedSignal] = []
        for spec in self._specs:
            gen = spec.generator
            result = gen.evaluate(data, current_index, self._hub_m5)
            aligned.append(AlignedSignal(
                result=result,
                timeframe=Timeframe.M5,
                generator_name=gen.name,
                category=gen.category,
                weight=spec.weight,
                htf_bar_index=current_index,
                htf_bar_count=current_index + 1,
            ))
        return aligned
