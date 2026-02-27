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

from stockdownloader.strategies.intraday.base import InfraExitConfig


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
