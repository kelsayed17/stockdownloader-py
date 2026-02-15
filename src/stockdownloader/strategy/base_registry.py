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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Callable, ClassVar, Generic, TypeVar

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
    def clear(cls) -> None:
        """Remove all registrations (for testing only)."""
        cls._entries.clear()
