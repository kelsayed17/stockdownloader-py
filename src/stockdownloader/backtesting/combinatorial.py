"""Exhaustive combinatorial signal-stack tester.

Generates all valid subsets of signal generators (2–5 generators per combo),
pairs them with timeframe, weight, mode, and threshold settings, then
backtests each configuration through the :class:`IntradayBacktestEngine`.

Pruning filters keep the search space manageable:

* Minimum/maximum combo size
* Category diversity requirement (at least 2 of 4 categories)
* Configurable timeframe, weight, mode, and threshold grids

Supports **parallel execution** via ``concurrent.futures.ProcessPoolExecutor``
for significant speedup on multi-core machines.  Each configuration is
completely independent, making the workload embarrassingly parallel.

Usage::

    from stockdownloader.backtesting.combinatorial import (
        CombinatorialTester, CombinatorialConfig,
    )

    tester = CombinatorialTester(config, data, initial_capital)
    results = tester.run()          # single-process (default)
    results = tester.run(workers=8) # parallel with 8 worker processes

Output is written to both stdout and a log file (``TeeWriter`` pattern)
for monitoring via ``tail -f``.
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal
from operator import attrgetter
from pathlib import Path
from typing import TextIO

from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.core.io import TeeWriter
from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.optimization.scoring import score_v2 as _score
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.signals.timeframe_aligner import TimeframeSignalSpec
from stockdownloader.strategies.registry import (
    SignalGeneratorEntry,
    SignalGeneratorRegistry,
)
from stockdownloader.signals.engine import (
    AggregationMode,
    StackConfig,
)
from stockdownloader.signals.intraday_adapter import (
    StackedIntradayStrategy,
)
from stockdownloader.core.timeframe import Timeframe

logger = logging.getLogger(__name__)

# Module-level variable for worker processes (set by _init_worker)
_WORKER_DATA: list[IntradayPriceData] | None = None
_WORKER_TRADING_DAYS: int = 0


def _init_worker(
    data: list[IntradayPriceData],
    trading_days: int,
) -> None:
    """Initialize worker process with shared data (called once per worker)."""
    global _WORKER_DATA, _WORKER_TRADING_DAYS
    _WORKER_DATA = data
    _WORKER_TRADING_DAYS = trading_days
    # Trigger generator registration in the worker
    import stockdownloader.signals.generators  # noqa: F401


@dataclass(frozen=True, slots=True)
class ComboResult:
    """One combination's backtest result.

    Attributes
    ----------
    generator_names:
        Tuple of generator registry names in this combo.
    timeframe:
        The timeframe used (uniform per combo for speed).
    weight:
        Weight applied to all generators.
    mode:
        Aggregation mode.
    buy_threshold:
        BUY threshold.
    sell_threshold:
        SELL threshold.
    result:
        Full backtest result.
    score:
        Composite fitness score from :func:`optimizer_scoring.score`.
    """

    generator_names: tuple[str, ...]
    timeframe: Timeframe
    weight: float
    mode: AggregationMode
    buy_threshold: float
    sell_threshold: float
    result: BacktestResult
    score: float


@dataclass(slots=True)
class CombinatorialConfig:
    """Configuration for the combinatorial search.

    Attributes
    ----------
    min_combo_size:
        Minimum number of generators in a combo (default: 2).
    max_combo_size:
        Maximum number of generators in a combo (default: 4).
    min_category_diversity:
        Minimum number of distinct signal categories required (default: 2).
    timeframes:
        Timeframes to test (default: M5 only for speed).
    weights:
        Weight values to test (uniform across all generators in a combo).
    modes:
        Aggregation modes to test.
    buy_thresholds:
        Buy threshold values to test.
    sell_thresholds:
        Sell threshold values to test.
    require_fire:
        Whether to require fire events (default: True).
    allow_shorts:
        Whether to enable short positions (default: True).
    initial_capital:
        Starting capital for backtests.
    risk_per_trade:
        Risk per trade fraction.
    generator_names:
        Optional subset of generator names to include.
        If None, all registered generators are used.
    """

    min_combo_size: int = 2
    max_combo_size: int = 4
    min_category_diversity: int = 2
    timeframes: list[Timeframe] = field(
        default_factory=lambda: [Timeframe.M5],
    )
    weights: list[float] = field(
        default_factory=lambda: [1.0],
    )
    modes: list[AggregationMode] = field(
        default_factory=lambda: [
            AggregationMode.WEIGHTED_AVERAGE,
            AggregationMode.UNANIMOUS,
        ],
    )
    buy_thresholds: list[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.5],
    )
    sell_thresholds: list[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.5],
    )
    require_fire: bool = True
    allow_shorts: bool = True
    initial_capital: Decimal = Decimal("100000")
    risk_per_trade: Decimal = Decimal("0.01")
    generator_names: list[str] | None = None


def _run_single_config(
    combo: tuple[str, ...],
    tf: Timeframe,
    weight: float,
    mode: AggregationMode,
    buy_th: float,
    sell_th: float,
    initial_capital: Decimal,
    risk_per_trade: Decimal,
    require_fire: bool,
    allow_shorts: bool,
) -> ComboResult | None:
    """Run a single backtest config — picklable top-level function for workers.

    Uses module-level ``_WORKER_DATA`` and ``_WORKER_TRADING_DAYS`` set by
    :func:`_init_worker` so that the 39K-bar dataset is loaded once per worker
    process rather than pickled with every task submission.

    Returns a :class:`ComboResult` on success, ``None`` on failure.
    """
    data = _WORKER_DATA
    trading_days = _WORKER_TRADING_DAYS
    if data is None:
        return None

    entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}

    specs: list[TimeframeSignalSpec] = []
    for name in combo:
        entry = entries[name]
        gen = entry.factory(**entry.default_kwargs)
        specs.append(TimeframeSignalSpec(
            generator=gen,
            timeframe=tf,
            weight=weight,
        ))

    stack_config = StackConfig(
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        mode=mode,
        require_fire=require_fire,
    )

    strategy_name = (
        f"{'+'.join(combo)} | {tf.label} | "
        f"w={weight} | {mode.value} | "
        f"buy={buy_th} sell={sell_th}"
    )

    strategy = StackedIntradayStrategy(
        name=strategy_name,
        config=stack_config,
        specs=specs,
        allow_shorts=allow_shorts,
    )

    try:
        engine = IntradayBacktestEngine(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )
        bt_result = engine.run(strategy, data)
        combo_score = _score(bt_result, trading_days)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None

    return ComboResult(
        generator_names=combo,
        timeframe=tf,
        weight=weight,
        mode=mode,
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        result=bt_result,
        score=combo_score,
    )


class CombinatorialTester:
    """Exhaustive combinatorial signal-stack tester.

    Parameters
    ----------
    config:
        Search space configuration.
    data:
        5-minute bar data for backtesting.
    log_file:
        Optional writable text file for progress output.
    """

    def __init__(
        self,
        config: CombinatorialConfig,
        data: list[IntradayPriceData],
        log_file: TextIO | None = None,
    ) -> None:
        self._config = config
        self._data = data
        self._tee = TeeWriter(log_file) if log_file else None

        # Import generators to trigger registration
        import stockdownloader.signals.generators  # noqa: F401

    def _print(self, msg: str) -> None:
        """Print to stdout and optional log file."""
        if self._tee:
            self._tee.write(msg + "\n")
        else:
            logger.info("%s", msg)

    def run(self, workers: int = 1) -> list[ComboResult]:
        """Run the exhaustive combinatorial search.

        Parameters
        ----------
        workers:
            Number of parallel worker processes.  ``1`` (default) runs
            single-threaded in-process.  Values > 1 use a
            :class:`~concurrent.futures.ProcessPoolExecutor` for parallel
            backtest execution.  ``0`` means *auto* — uses ``os.cpu_count()``.

        Returns a list of :class:`ComboResult` sorted by score (descending).
        """
        cfg = self._config

        # Determine generator names
        if cfg.generator_names:
            gen_names = cfg.generator_names
        else:
            gen_names = SignalGeneratorRegistry.list_names()

        # Get entries for category lookup
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}

        # Generate valid combos
        combos = self._generate_combos(gen_names, entries, cfg)
        self._print(f"Generated {len(combos)} valid generator combinations")

        # Generate setting grid
        settings = list(itertools.product(
            cfg.timeframes,
            cfg.weights,
            cfg.modes,
            cfg.buy_thresholds,
            cfg.sell_thresholds,
        ))
        total_configs = len(combos) * len(settings)
        self._print(
            f"Testing {len(combos)} combos × {len(settings)} settings "
            f"= {total_configs} configurations"
        )

        # Compute trading days for scoring
        trading_days = len({
            d.date[:10] for d in self._data
        })

        # Resolve worker count
        if workers == 0:
            workers = os.cpu_count() or 4
        use_parallel = workers > 1

        if use_parallel:
            self._print(f"Running with {workers} parallel workers")
            results = self._run_parallel(
                combos, settings, total_configs, trading_days, cfg, workers,
            )
        else:
            results = self._run_sequential(
                combos, settings, total_configs, trading_days, cfg, entries,
            )

        # Sort by score descending
        results.sort(key=attrgetter("score"), reverse=True)
        return results

    def _run_sequential(
        self,
        combos: list[tuple[str, ...]],
        settings: list[tuple],
        total_configs: int,
        trading_days: int,
        cfg: CombinatorialConfig,
        entries: dict[str, SignalGeneratorEntry],
    ) -> list[ComboResult]:
        """Single-process sequential execution (original behaviour)."""
        engine = IntradayBacktestEngine(
            initial_capital=cfg.initial_capital,
            risk_per_trade=cfg.risk_per_trade,
        )

        results: list[ComboResult] = []
        start_time = time.time()
        tested = 0

        for combo_idx, combo in enumerate(combos):
            for tf, weight, mode, buy_th, sell_th in settings:
                tested += 1

                # Build specs
                specs: list[TimeframeSignalSpec] = []
                for name in combo:
                    entry = entries[name]
                    gen = entry.factory(**entry.default_kwargs)
                    specs.append(TimeframeSignalSpec(
                        generator=gen,
                        timeframe=tf,
                        weight=weight,
                    ))

                stack_config = StackConfig(
                    buy_threshold=buy_th,
                    sell_threshold=sell_th,
                    mode=mode,
                    require_fire=cfg.require_fire,
                )

                strategy_name = (
                    f"{'+'.join(combo)} | {tf.label} | "
                    f"w={weight} | {mode.value} | "
                    f"buy={buy_th} sell={sell_th}"
                )

                strategy = StackedIntradayStrategy(
                    name=strategy_name,
                    config=stack_config,
                    specs=specs,
                    allow_shorts=cfg.allow_shorts,
                )

                try:
                    bt_result = engine.run(strategy, self._data)
                    combo_score = _score(bt_result, trading_days)
                except (ValueError, ZeroDivisionError, ArithmeticError) as e:
                    logger.warning("Failed: %s — %s", strategy_name, e)
                    continue

                results.append(ComboResult(
                    generator_names=combo,
                    timeframe=tf,
                    weight=weight,
                    mode=mode,
                    buy_threshold=buy_th,
                    sell_threshold=sell_th,
                    result=bt_result,
                    score=combo_score,
                ))

                # Progress every 100 configs
                if tested % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = tested / elapsed if elapsed > 0 else 0
                    eta = (total_configs - tested) / rate if rate > 0 else 0
                    self._print(
                        f"  [{tested}/{total_configs}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} configs/s, "
                        f"ETA {eta:.0f}s"
                    )

            # Log after each combo set
            elapsed = time.time() - start_time
            self._print(
                f"  Combo {combo_idx + 1}/{len(combos)}: "
                f"{'+'.join(combo)} done ({elapsed:.0f}s total)"
            )

        elapsed = time.time() - start_time
        self._print(
            f"\nCompleted {tested} configurations in {elapsed:.1f}s "
            f"({tested / elapsed:.1f} configs/s)"
        )
        return results

    def _run_parallel(
        self,
        combos: list[tuple[str, ...]],
        settings: list[tuple],
        total_configs: int,
        trading_days: int,
        cfg: CombinatorialConfig,
        workers: int,
    ) -> list[ComboResult]:
        """Multi-process parallel execution using ProcessPoolExecutor.

        Each configuration is submitted as an independent task.  Tasks are
        batched by combo to keep memory usage reasonable and provide
        per-combo progress updates.
        """
        results: list[ComboResult] = []
        start_time = time.time()
        completed = 0

        # Submit all tasks in batches per combo for progress reporting.
        # Data is loaded once per worker via _init_worker (not pickled per task).
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self._data, trading_days),
        ) as pool:
            for combo_idx, combo in enumerate(combos):
                # Submit all settings for this combo
                futures = []
                for tf, weight, mode, buy_th, sell_th in settings:
                    fut = pool.submit(
                        _run_single_config,
                        combo=combo,
                        tf=tf,
                        weight=weight,
                        mode=mode,
                        buy_th=buy_th,
                        sell_th=sell_th,
                        initial_capital=cfg.initial_capital,
                        risk_per_trade=cfg.risk_per_trade,
                        require_fire=cfg.require_fire,
                        allow_shorts=cfg.allow_shorts,
                    )
                    futures.append(fut)

                # Collect results for this combo
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        result = fut.result()
                        if result is not None:
                            results.append(result)
                    except (ValueError, ZeroDivisionError, ArithmeticError) as e:
                        logger.warning("Worker failed: %s", e)

                    # Progress every 100 configs
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total_configs - completed) / rate if rate > 0 else 0
                        self._print(
                            f"  [{completed}/{total_configs}] "
                            f"{elapsed:.0f}s elapsed, "
                            f"{rate:.1f} configs/s, "
                            f"ETA {eta:.0f}s"
                        )

                # Log after each combo
                elapsed = time.time() - start_time
                self._print(
                    f"  Combo {combo_idx + 1}/{len(combos)}: "
                    f"{'+'.join(combo)} done ({elapsed:.0f}s total)"
                )

        elapsed = time.time() - start_time
        self._print(
            f"\nCompleted {completed} configurations in {elapsed:.1f}s "
            f"({completed / elapsed:.1f} configs/s)"
        )
        return results

    @staticmethod
    def _generate_combos(
        gen_names: list[str],
        entries: dict[str, SignalGeneratorEntry],
        cfg: CombinatorialConfig,
    ) -> list[tuple[str, ...]]:
        """Generate all valid generator combinations with pruning."""
        combos: list[tuple[str, ...]] = []
        sorted_names = sorted(gen_names)

        for k in range(cfg.min_combo_size, cfg.max_combo_size + 1):
            for combo in itertools.combinations(sorted_names, k):
                # Category diversity check
                categories = {entries[n].category for n in combo}
                if len(categories) < cfg.min_category_diversity:
                    continue
                combos.append(combo)

        return combos

    @staticmethod
    def format_results(
        results: list[ComboResult],
        top_n: int = 50,
    ) -> str:
        """Format the top N results as a readable table.

        Returns a string suitable for printing or writing to a file.
        """
        lines: list[str] = []
        lines.append("=" * 120)
        lines.append(f"TOP {min(top_n, len(results))} SIGNAL STACK COMBINATIONS")
        lines.append("=" * 120)
        lines.append(
            f"{'Rank':>4}  {'Score':>8}  {'P&L':>12}  {'WR%':>6}  "
            f"{'Trades':>6}  {'PF':>6}  {'DD%':>6}  "
            f"{'TF':>4}  {'Mode':>15}  {'Th':>8}  Generators"
        )
        lines.append("-" * 120)

        for i, r in enumerate(results[:top_n], 1):
            pnl = float(r.result.final_capital - r.result.initial_capital)
            wr = float(r.result.win_rate)
            pf = float(r.result.profit_factor)
            dd = float(r.result.max_drawdown)
            gens = "+".join(r.generator_names)
            th = f"{r.buy_threshold}/{r.sell_threshold}"

            lines.append(
                f"{i:>4}  {r.score:>8.2f}  ${pnl:>11,.0f}  "
                f"{wr:>5.1f}%  {r.result.total_trades:>6}  "
                f"{pf:>6.2f}  {dd:>5.1f}%  "
                f"{r.timeframe.label:>4}  {r.mode.value:>15}  "
                f"{th:>8}  {gens}"
            )

        lines.append("=" * 120)
        return "\n".join(lines)

    @staticmethod
    def frequency_analysis(
        results: list[ComboResult],
        top_n: int = 50,
    ) -> str:
        """Analyze which generators appear most often in the top N combos.

        Returns a string with frequency counts and percentages.
        """
        from collections import Counter

        counter: Counter[str] = Counter()
        for r in results[:top_n]:
            for name in r.generator_names:
                counter[name] += 1

        lines: list[str] = []
        lines.append("\n" + "=" * 60)
        lines.append(f"GENERATOR FREQUENCY (Top {min(top_n, len(results))} combos)")
        lines.append("=" * 60)
        lines.append(f"{'Generator':>20}  {'Count':>5}  {'%':>6}")
        lines.append("-" * 60)

        for name, count in counter.most_common():
            pct = count / min(top_n, len(results)) * 100
            lines.append(f"{name:>20}  {count:>5}  {pct:>5.1f}%")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def category_pair_analysis(
        results: list[ComboResult],
        entries: dict[str, SignalGeneratorEntry],
    ) -> str:
        """Find the best combo for each category pair.

        Returns a string showing the top combo per pair of categories.
        """
        categories = ["momentum", "trend", "volatility", "volume"]
        pair_best: dict[tuple[str, str], ComboResult] = {}

        for r in results:
            cats = {entries[n].category for n in r.generator_names if n in entries}
            for c1, c2 in itertools.combinations(sorted(cats), 2):
                pair = (c1, c2)
                if pair not in pair_best or r.score > pair_best[pair].score:
                    pair_best[pair] = r

        lines: list[str] = []
        lines.append("\n" + "=" * 100)
        lines.append("BEST COMBO PER CATEGORY PAIR")
        lines.append("=" * 100)
        lines.append(
            f"{'Categories':>30}  {'Score':>8}  {'P&L':>12}  "
            f"{'Trades':>6}  Generators"
        )
        lines.append("-" * 100)

        for pair in sorted(pair_best.keys()):
            r = pair_best[pair]
            pnl = float(r.result.final_capital - r.result.initial_capital)
            gens = "+".join(r.generator_names)
            lines.append(
                f"{pair[0]+' + '+pair[1]:>30}  {r.score:>8.2f}  "
                f"${pnl:>11,.0f}  {r.result.total_trades:>6}  {gens}"
            )

        lines.append("=" * 100)
        return "\n".join(lines)
