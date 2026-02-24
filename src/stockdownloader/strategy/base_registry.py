"""Generic registry base for auto-discovery and CLI resolution.

Provides a template for singleton-style registries that map short
case-insensitive names to factory functions.  Used by both
:class:`StrategyRegistry` and :class:`SignalGeneratorRegistry`.

Usage::

    from stockdownloader.strategy.base_registry import BaseRegistry, RegistryEntry

    class MyEntry(RegistryEntry):
        ...

    class MyRegistry(BaseRegistry[MyEntry]):
        _entries: dict[str, MyEntry] = {}
        _entry_cls = MyEntry

Concrete Registries
-------------------

:class:`StrategyRegistry`
    Central strategy registry for auto-discovery and CLI resolution.
    Provides a mapping from short CLI names (case-insensitive) to factory
    functions that create strategy instances with default parameters.

    Usage::

        from stockdownloader.strategy.base_registry import StrategyRegistry

        # Resolve by CLI name
        entry = StrategyRegistry.get("rsi")
        strategy = StrategyRegistry.create("rsi")

        # With overrides
        strategy = StrategyRegistry.create("rsi", period=21, oversold=25.0)

        # List all
        for entry in StrategyRegistry.all_entries(category="daily"):
            print(entry.name, entry.display_name)

:class:`SignalGeneratorRegistry`
    Registry for atomic signal generators. Shares the same lookup, creation,
    and filtering API as StrategyRegistry.

    Usage::

        from stockdownloader.strategy.base_registry import SignalGeneratorRegistry

        entry = SignalGeneratorRegistry.get("rsi")
        gen   = SignalGeneratorRegistry.create("rsi", period=7)

        for e in SignalGeneratorRegistry.all_entries(category="momentum"):
            print(e.name, e.display_name)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Callable, ClassVar, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="RegistryEntry")


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """Base entry for a registered factory with its metadata."""

    name: str
    """Lookup key (lowercase)."""

    display_name: str
    """Human-readable label."""

    category: str
    """Grouping category (e.g. ``'daily'``, ``'intraday'``, ``'momentum'``)."""

    factory: Callable[..., Any]
    """Callable that creates an instance."""

    default_kwargs: dict[str, Any] = field(default_factory=dict)
    """Default constructor keyword arguments."""

    param_space: dict[str, list[Any]] = field(default_factory=dict)
    """Parameter search ranges: ``{param_name: [values]}``."""


class BaseRegistry(Generic[T]):
    """Singleton-style registry base class.

    Subclasses must declare their own ``_entries`` class variable
    (not inherited) so each registry has its own namespace::

        class MyRegistry(BaseRegistry[MyEntry]):
            _entries: ClassVar[dict[str, MyEntry]] = {}
            _entry_cls: ClassVar[type[MyEntry]] = MyEntry
    """

    _entries: ClassVar[dict[str, Any]]  # Must be overridden per subclass
    _entry_cls: ClassVar[type[RegistryEntry]]  # Must be overridden per subclass
    _label: ClassVar[str] = "entry"  # Human label for error messages

    @classmethod
    def register(
        cls,
        name: str,
        display_name: str,
        category: str,
        factory: Callable[..., Any],
        default_kwargs: dict[str, Any] | None = None,
        param_space: dict[str, list[Any]] | None = None,
    ) -> None:
        """Register a factory under *name* (case-insensitive)."""
        cls._entries[name.lower()] = cls._entry_cls(
            name=name.lower(),
            display_name=display_name,
            category=category,
            factory=factory,
            default_kwargs=default_kwargs or {},
            param_space=param_space or {},
        )

    @classmethod
    def get(cls, name: str) -> T:
        """Look up an entry by name.  Raises :class:`ValueError` if unknown."""
        key = name.lower()
        if key not in cls._entries:
            available = ", ".join(sorted(cls._entries.keys()))
            raise ValueError(
                f"Unknown {cls._label} '{name}'. Available: {available}"
            )
        return cls._entries[key]

    @classmethod
    def create(cls, name: str, **overrides: Any) -> Any:
        """Create an instance by name with optional parameter overrides."""
        entry = cls.get(name)
        kwargs = entry.default_kwargs | overrides
        return entry.factory(**kwargs)

    @classmethod
    def all_entries(cls, category: str | None = None) -> list[T]:
        """Return all entries, optionally filtered by category."""
        entries = list(cls._entries.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=attrgetter("name"))

    @classmethod
    def list_names(cls, category: str | None = None) -> list[str]:
        """Return sorted list of registered names."""
        return [e.name for e in cls.all_entries(category)]

    @classmethod
    def apply_config_overrides(cls, config_path: str | None = None) -> int:
        """Override ``default_kwargs`` for registered entries from a JSON config.

        The JSON file should have strategy names as top-level keys mapping
        to dicts of parameter overrides::

            {
                "sma": {"short_period": 12, "long_period": 50},
                "rsi": {"period": 10, "oversold": 25.0}
            }

        Only keys present in the JSON will be overridden — others keep their
        built-in defaults.  This enables users to customize strategy defaults
        without modifying Python code.

        Parameters
        ----------
        config_path:
            Path to a JSON config file.  Resolved via
            :func:`~stockdownloader.core.config.load_config`.

        Returns
        -------
        int
            Number of entries that were updated.
        """
        if not config_path:
            return 0

        from stockdownloader.core.config import load_config

        cfg = load_config(config_path)
        count = 0

        for name, entry in list(cls._entries.items()):
            overrides = cfg.get(name)
            if overrides and isinstance(overrides, dict):
                merged = entry.default_kwargs | overrides
                # Replace the frozen entry with updated defaults
                cls._entries[name] = cls._entry_cls(
                    name=entry.name,
                    display_name=entry.display_name,
                    category=entry.category,
                    factory=entry.factory,
                    default_kwargs=merged,
                    param_space=entry.param_space,
                )
                count += 1
                logger.debug(
                    "Applied config overrides for %s: %s", name, overrides
                )

        if count:
            logger.info("Applied config overrides to %d %s(s)", count, cls._label)
        return count

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (for testing only)."""
        cls._entries.clear()


# ── Concrete Registries ──────────────────────────────────────────────────────

StrategyEntry = RegistryEntry
"""Alias for :class:`RegistryEntry` used by :class:`StrategyRegistry`."""


class StrategyRegistry(BaseRegistry[RegistryEntry]):
    """Singleton-style registry for strategies.

    Strategies self-register at import time via their subpackage
    ``__init__.py`` files.
    """

    _entries: ClassVar[dict[str, RegistryEntry]] = {}
    _entry_cls: ClassVar[type[RegistryEntry]] = RegistryEntry
    _label: ClassVar[str] = "strategy"


SignalGeneratorEntry = RegistryEntry
"""Alias for :class:`RegistryEntry` used by :class:`SignalGeneratorRegistry`."""


class SignalGeneratorRegistry(BaseRegistry[RegistryEntry]):
    """Singleton-style registry for atomic signal generators.

    Generators self-register at import time via their sub-package
    ``__init__.py``.
    """

    _entries: ClassVar[dict[str, RegistryEntry]] = {}
    _entry_cls: ClassVar[type[RegistryEntry]] = RegistryEntry
    _label: ClassVar[str] = "signal generator"
