"""Registry for atomic signal generators.

Built on :class:`~stockdownloader.strategy.base_registry.BaseRegistry`,
sharing the same lookup, creation, and filtering API as
:class:`~stockdownloader.strategy.registry.StrategyRegistry`.

Generators self-register at import time via their sub-package
``__init__.py``.

Usage::

    from stockdownloader.strategy.signals.signal_registry import (
        SignalGeneratorRegistry,
    )

    entry = SignalGeneratorRegistry.get("rsi")
    gen   = SignalGeneratorRegistry.create("rsi", period=7)

    for e in SignalGeneratorRegistry.all_entries(category="momentum"):
        print(e.name, e.display_name)
"""
from __future__ import annotations

from typing import ClassVar

from stockdownloader.strategy.base_registry import BaseRegistry, RegistryEntry

SignalGeneratorEntry = RegistryEntry


class SignalGeneratorRegistry(BaseRegistry[RegistryEntry]):
    """Singleton-style registry for atomic signal generators."""

    _entries: ClassVar[dict[str, RegistryEntry]] = {}
    _entry_cls: ClassVar[type[RegistryEntry]] = RegistryEntry
    _label: ClassVar[str] = "signal generator"
