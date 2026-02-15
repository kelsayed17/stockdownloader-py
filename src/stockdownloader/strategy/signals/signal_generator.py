"""Atomic signal generator abstraction.

Each generator produces a directional score from a single technical
indicator or condition.  Generators are **stateless** pure functions of
``(data, index, hub)`` and carry their own parameter space for optimization.

Score semantics
---------------
* ``+1.0`` = maximally bullish
* ``-1.0`` = maximally bearish
* ``0.0``  = neutral / no opinion

The ``fired`` flag is separate from ``score``.  A generator can report a
non-zero score (state) without "firing" (event).  For example, RSI at 25
produces a bullish *score* but only *fires* on the bar where it crosses
below the oversold threshold.

Usage::

    from stockdownloader.strategy.signals.signal_generator import (
        AtomicSignalGenerator, SignalResult,
    )

    class MyGenerator(AtomicSignalGenerator):
        def evaluate(self, data, index, hub):
            ...
            return SignalResult(score=0.7, direction=SignalDirection.BULLISH,
                               fired=True)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stockdownloader.model.price_data import PriceData
    from stockdownloader.util.indicator_hub import IndicatorHub


class SignalDirection(Enum):
    """Directional bias of a signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class SignalResult:
    """Output of a single atomic signal generator at one bar.

    Attributes
    ----------
    score:
        Continuous directional score in ``[-1.0, +1.0]``.
    direction:
        Discrete direction derived from *score*.
    fired:
        Whether the signal crossed its activation threshold this bar.
    confidence:
        Optional confidence estimate ``[0.0, 1.0]`` for weighting.
    metadata:
        Arbitrary extra context (indicator values, thresholds, etc.).
    """

    score: float
    direction: SignalDirection
    fired: bool
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def neutral() -> SignalResult:
        """Return a neutral signal (warmup / no-data sentinel)."""
        return SignalResult(
            score=0.0,
            direction=SignalDirection.NEUTRAL,
            fired=False,
        )


class AtomicSignalGenerator(ABC):
    """Abstract base for atomic, reusable signal generators.

    **Contract**:

    * ``evaluate()`` must be **stateless** — all mutable state lives in
      the :class:`IndicatorHub` cache that is passed in.
    * Generators must tolerate any *index* in ``[0, len(data)-1]``.
      If there is insufficient warmup data, return ``SignalResult.neutral()``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier (e.g. ``'rsi_14_30_70'``)."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label (e.g. ``'RSI(14) [30/70]'``)."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Signal category: ``'trend'``, ``'momentum'``, ``'volatility'``, or ``'volume'``."""

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """Bars needed before meaningful output."""

    @abstractmethod
    def evaluate(
        self,
        data: Sequence[PriceData],
        index: int,
        hub: IndicatorHub,
    ) -> SignalResult:
        """Compute the signal at the given bar index.

        Must be **stateless**.  All caching is handled by *hub*.
        """

    @property
    @abstractmethod
    def param_space(self) -> dict[str, list[Any]]:
        """Return tunable parameters and their search values.

        Used by the combinatorial tester for exhaustive search.
        Example::

            {"period": [7, 10, 14, 21],
             "oversold": [20.0, 25.0, 30.0, 35.0]}
        """
