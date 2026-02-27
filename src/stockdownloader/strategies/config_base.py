"""Mixin providing JSON serialization for frozen strategy config dataclasses.

Adds ``to_dict`` / ``from_dict`` / ``to_json`` / ``from_json`` / ``save`` /
``load`` to any ``@dataclass`` strategy config.  Handles :class:`~decimal.Decimal`
round-trips by serialising to string (preserving precision) and reconstructing
via field-default type detection.

Usage::

    from dataclasses import dataclass
    from decimal import Decimal
    from stockdownloader.strategies.config_base import StrategyConfigMixin

    @dataclass(frozen=True, slots=True)
    class MyConfig(StrategyConfigMixin):
        period: int = 14
        threshold: Decimal = Decimal("1.5")

    cfg = MyConfig(period=7)
    cfg.save("my_config.json")

    cfg2 = MyConfig.load("my_config.json")
    assert cfg2 == cfg
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    """Fallback serializer for :func:`json.dumps`.

    Converts :class:`~decimal.Decimal` to *string* (not float) so that
    precision is preserved across round-trips.
    """
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _coerce_field(default_val: Any, raw: Any) -> Any:
    """Coerce a JSON-decoded *raw* value to match the type of *default_val*.

    This is necessary because :func:`json.loads` produces plain ``int`` /
    ``float`` / ``str`` / ``bool`` values whereas the dataclass field may
    expect a :class:`~decimal.Decimal`.
    """
    if isinstance(default_val, Decimal):
        return Decimal(str(raw))
    # bool check must precede int check (bool is a subclass of int)
    if isinstance(default_val, bool) and isinstance(raw, bool):
        return raw
    if isinstance(default_val, int) and isinstance(raw, (int, float)):
        return int(raw)
    return raw


class StrategyConfigMixin:
    """Mixin that adds JSON serialization to frozen strategy config dataclasses."""

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation (Decimals remain as Decimal)."""
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON string.  Decimals are serialized as strings."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    # -- Deserialization -------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfigMixin":
        """Reconstruct a config from a plain dict.

        Field types are inferred from the dataclass *default* values so that
        Decimals are properly restored from their string representation.
        """
        field_defaults: dict[str, Any] = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            if f.default is not dataclasses.MISSING:
                field_defaults[f.name] = f.default

        converted: dict[str, Any] = {}
        for k, v in data.items():
            if k in field_defaults:
                converted[k] = _coerce_field(field_defaults[k], v)
            else:
                # Unknown field — pass through, let the dataclass __init__
                # raise if it's truly unexpected.
                converted[k] = v
        return cls(**converted)  # type: ignore[return-value]

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyConfigMixin":
        """Reconstruct a config from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -- File I/O --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write this config to a JSON file at *path*."""
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "StrategyConfigMixin":
        """Load a config from a JSON file at *path*."""
        return cls.from_json(Path(path).read_text())
