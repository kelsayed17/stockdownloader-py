"""Centralized strategy registration for StrategyRegistry.

Call :func:`ensure_registered` from any entry point that needs the registry
populated (CLI apps, tests that query by name).  Idempotent — safe to call
multiple times.

All strategy defaults, param_space, and factory mappings are defined in
JSON config files under ``config/strategy/``.  The loader
(:mod:`~stockdownloader.strategy.registration_loader`) reads these at
startup and registers each strategy dynamically.
"""

from __future__ import annotations

_registered = False


def ensure_registered() -> None:
    """Populate :class:`StrategyRegistry` with all built-in strategies.

    Loads strategy definitions from JSON config files:

    - ``config/strategy/daily_registrations.json`` (7 daily strategies)
    - ``config/strategy/options_registrations.json`` (2 options strategies)
    - ``config/strategy/intraday_registrations.json`` (7 intraday strategies)
    """
    global _registered
    if _registered:
        return
    _registered = True

    from stockdownloader.strategy.registration_loader import load_registrations

    load_registrations("daily", "config/strategy/daily_registrations.json")
    load_registrations("options", "config/strategy/options_registrations.json")
    load_registrations("intraday", "config/strategy/intraday_registrations.json")
