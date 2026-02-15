"""Central strategy registry for auto-discovery and CLI resolution.

Provides a mapping from short CLI names (case-insensitive) to factory
functions that create strategy instances with default parameters.

Each registration also carries an optional ``param_space`` dict used by
the optimizer to define tunable parameters and their search ranges.

Usage::

    from stockdownloader.strategy.registry import StrategyRegistry

    # Resolve by CLI name
    entry = StrategyRegistry.get("rsi")
    strategy = StrategyRegistry.create("rsi")

    # With overrides
    strategy = StrategyRegistry.create("rsi", period=21, oversold=25.0)

    # List all
    for entry in StrategyRegistry.all_entries(category="daily"):
        print(entry.name, entry.display_name)
"""
from __future__ import annotations

from typing import ClassVar

from stockdownloader.strategy.base_registry import BaseRegistry, RegistryEntry

StrategyEntry = RegistryEntry


class StrategyRegistry(BaseRegistry[RegistryEntry]):
    """Singleton-style registry.

    Strategies self-register at import time via their subpackage
    ``__init__.py`` files.
    """

    _entries: ClassVar[dict[str, RegistryEntry]] = {}
    _entry_cls: ClassVar[type[RegistryEntry]] = RegistryEntry
    _label: ClassVar[str] = "strategy"
