"""Unified VWAP strategy — single-infra multi-mode dispatcher.

Holds a SINGLE :class:`IntradayInfra` and delegates to existing
standalone strategies' ``_evaluate_entry()`` methods with priority
dispatch: PS > ORB > PB > REV.

The unified strategy uses a single ``SessionState``, so day-trade
counts, circuit breaker, and day-loss limits are shared across all
modes.  Trail strategy is swapped via ``set_active_trail`` before
each entry to match the mode that fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import IntradayExitManager
from stockdownloader.strategies.intraday.trail import (
    AtrChandelierTrail,
    BreakevenTrail,
    TrailStrategy,
    VwapRatchetTrail,
)
from stockdownloader.strategies.intraday.pullback import PullbackStrategy, PullbackStrategyConfig
from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy, PatternScalpStrategyConfig
from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy, ORBreakoutStrategyConfig
from stockdownloader.strategies.intraday.reversal import ReversalStrategy, ReversalStrategyConfig

if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.session import BarContext


# ── UnifiedVWAPConfig ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class UnifiedVWAPConfig(InfraExitConfig):
    """Configuration for the unified multi-mode VWAP strategy.

    Extends :class:`InfraExitConfig` with per-mode enable flags.
    All modes are enabled by default.
    """

    pb_enable: bool = True
    ps_enable: bool = True
    orb_enable: bool = True
    rev_enable: bool = True


# ── UnifiedVWAPStrategy ───────────────────────────────────────────────


class UnifiedVWAPStrategy(BaseIntradayStrategy):
    """Single-infra multi-mode VWAP strategy with priority dispatch.

    Constructs a single :class:`IntradayInfra` and delegates to
    existing standalone strategies' ``_evaluate_entry()`` methods.
    Priority order: PS > ORB > PB > REV.

    All modes share a single ``SessionState``, so day-trade counts,
    circuit breaker, and day-loss limits apply across all modes.
    """

    _ENTRY_FLAGS: dict[str, bool] = {}

    def __init__(
        self,
        config: UnifiedVWAPConfig | None = None,
        pb_overrides: dict | None = None,
        ps_overrides: dict | None = None,
        orb_overrides: dict | None = None,
        rev_overrides: dict | None = None,
        **shared_overrides: object,
    ) -> None:
        if config is not None and shared_overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if shared_overrides:
            c = UnifiedVWAPConfig(**shared_overrides)
        else:
            c = config or UnifiedVWAPConfig()
        self._c = c

        # Single shared IntradayInfra with VwapRatchetTrail as default
        self._infra = IntradayInfra(c, IntradayExitManager(VwapRatchetTrail()))

        # Shared risk fields that PB and REV check internally.
        # These must match the unified config so that mode-level
        # ready/spaced checks agree with the shared SessionState.
        _shared_risk = dict(
            max_day=c.max_day,
            spacing=c.spacing,
            circuit=c.circuit,
            day_loss=c.day_loss,
            can_trade_bar=c.can_trade_bar,
            adx_thresh=c.adx_thresh,
            allow_longs=c.allow_longs,
            allow_shorts=c.allow_shorts,
        )

        def _merge(overrides: dict | None) -> dict:
            """Merge shared risk fields with mode-specific overrides."""
            merged = dict(_shared_risk)
            if overrides:
                merged.update(overrides)
            return merged

        # Build mode list in priority order: PS > ORB > PB > REV
        self._modes: list[tuple[BaseIntradayStrategy, type[TrailStrategy]]] = []

        if c.ps_enable:
            ps_cfg = PatternScalpStrategyConfig(**_merge(ps_overrides))
            ps = PatternScalpStrategy(config=ps_cfg)
            self._modes.append((ps, BreakevenTrail))

        if c.orb_enable:
            orb_cfg = ORBreakoutStrategyConfig(**_merge(orb_overrides))
            orb = ORBreakoutStrategy(config=orb_cfg)
            self._modes.append((orb, AtrChandelierTrail))

        if c.pb_enable:
            pb_cfg = PullbackStrategyConfig(**_merge(pb_overrides))
            pb = PullbackStrategy(config=pb_cfg)
            self._modes.append((pb, VwapRatchetTrail))

        if c.rev_enable:
            rev_cfg = ReversalStrategyConfig(**_merge(rev_overrides))
            rev = ReversalStrategy(config=rev_cfg)
            self._modes.append((rev, BreakevenTrail))

    @property
    def name(self) -> str:
        return "Unified VWAP"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        """Priority dispatch: first non-None signal wins.

        Iterates through modes in priority order (PS > ORB > PB > REV).
        When a mode fires, swaps the active trail strategy on the shared
        exit manager before returning the signal.
        """
        for strategy, trail_cls in self._modes:
            sig = strategy._evaluate_entry(ctx)
            if sig is not None:
                self._infra.exit_mgr.set_active_trail(trail_cls())
                return sig
        return None
