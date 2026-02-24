"""JSON-driven strategy registration loader.

Reads strategy registrations from JSON config files and registers
them with :class:`StrategyRegistry`.  Decimal values are encoded
as strings prefixed with ``D:`` (e.g. ``"D:1.5"``).

Call :func:`ensure_registered` from any entry point that needs the registry
populated (CLI apps, tests that query by name).  Idempotent -- safe to call
multiple times.

All strategy defaults, param_space, and factory mappings are defined in
JSON config files under ``config/strategies/``.

Usage::

    from stockdownloader.strategies.loader import ensure_registered

    ensure_registered()
"""

from __future__ import annotations

import importlib
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_config_path(path: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidate = _PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {path} (tried {candidate})")


def _convert_decimals(value: Any) -> Any:
    """Recursively convert ``D:``-prefixed strings to :class:`Decimal`.

    - ``"D:1.5"`` → ``Decimal("1.5")``
    - Lists are converted element-wise.
    - Dicts are converted value-wise.
    - All other types pass through unchanged.
    """
    if isinstance(value, str) and value.startswith("D:"):
        return Decimal(value[2:])
    if isinstance(value, list):
        return [_convert_decimals(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_decimals(v) for k, v in value.items()}
    return value


def load_registrations(category: str, config_path: str) -> None:
    """Load strategy registrations from a JSON config file.

    Parameters
    ----------
    category:
        Category label (``"daily"``, ``"intraday"``, ``"options"``).
    config_path:
        Path to the JSON config file, relative to the project root.
    """
    from stockdownloader.strategies.registry import StrategyRegistry

    resolved = _resolve_config_path(config_path)
    with open(resolved) as f:
        data: dict[str, Any] = json.load(f)

    for name, entry in data.items():
        if name.startswith("_"):
            continue  # Skip metadata keys like "_doc"

        module_path = entry["module"]
        class_name = entry["class"]

        try:
            module = importlib.import_module(module_path)
            factory = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            logger.error(
                "Failed to load strategy %s from %s.%s: %s",
                name, module_path, class_name, exc,
            )
            continue

        default_kwargs = _convert_decimals(entry.get("default_kwargs", {}))
        param_space = _convert_decimals(entry.get("param_space", {}))

        StrategyRegistry.register(
            name=name,
            display_name=entry["display_name"],
            category=category,
            factory=factory,
            default_kwargs=default_kwargs,
            param_space=param_space,
        )


# =========================================================================
# Idempotent entry point (formerly registrations.py)
# =========================================================================

_registered = False


def ensure_registered() -> None:
    """Populate :class:`StrategyRegistry` with all built-in strategies.

    Loads strategy definitions from JSON config files:

    - ``config/strategies/daily_registrations.json`` (7 daily strategies)
    - ``config/strategies/options_registrations.json`` (2 options strategies)
    - ``config/strategies/intraday_registrations.json`` (7 intraday strategies)
    """
    global _registered
    if _registered:
        return
    _registered = True

    load_registrations("daily", "config/strategies/daily_registrations.json")
    load_registrations("options", "config/strategies/options_registrations.json")
    load_registrations("intraday", "config/strategies/intraday_registrations.json")
