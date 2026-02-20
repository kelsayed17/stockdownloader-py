"""JSON-file-backed persistence for signal advisories.

Stores advisory history per symbol, supports deduplication, and
provides retrieval by symbol with newest-first ordering.

Usage::

    from stockdownloader.analysis.alert_store import AlertStore

    store = AlertStore("output/alert_history.json")
    is_new = store.save(advisory)   # True if not a duplicate
    latest = store.get_latest("SPY")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stockdownloader.model.signal_advisory import SignalAdvisory
from stockdownloader.util.config_loader import DEFAULT_ALERT_HISTORY

logger = logging.getLogger(__name__)

_DEFAULT_PATH = DEFAULT_ALERT_HISTORY


class AlertStore:
    """Persists :class:`SignalAdvisory` instances to a JSON file.

    The file format is a dict keyed by symbol, each containing a list of
    advisory dicts sorted newest-first.  Duplicate advisories (same
    :attr:`~SignalAdvisory.dedup_key`) are silently skipped.

    Parameters
    ----------
    filepath:
        Path to the JSON file.  Created on first write.
    max_history:
        Maximum number of advisories to keep per symbol.  Oldest are
        pruned when the limit is exceeded.
    """

    def __init__(
        self,
        filepath: str | Path = _DEFAULT_PATH,
        max_history: int = 500,
    ) -> None:
        self._path = Path(filepath)
        self._max_history = max_history
        self._data: dict[str, list[dict[str, Any]]] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, advisory: SignalAdvisory) -> bool:
        """Save *advisory* to the store.

        Returns ``True`` if the advisory was new (saved), ``False`` if it
        was a duplicate and was therefore skipped.
        """
        if self.is_duplicate(advisory):
            return False

        symbol = advisory.symbol
        entry = advisory.to_dict()

        if symbol not in self._data:
            self._data[symbol] = []

        # Insert newest-first
        self._data[symbol].insert(0, entry)

        # Prune oldest
        if len(self._data[symbol]) > self._max_history:
            self._data[symbol] = self._data[symbol][: self._max_history]

        self._write()
        return True

    def get_history(
        self,
        symbol: str,
        last_n: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the most recent *last_n* advisories for *symbol*.

        Returned newest-first.
        """
        entries = self._data.get(symbol, [])
        return entries[:last_n]

    def get_latest(self, symbol: str) -> dict[str, Any] | None:
        """Return the most recent advisory for *symbol*, or ``None``."""
        entries = self._data.get(symbol, [])
        return entries[0] if entries else None

    def is_duplicate(self, advisory: SignalAdvisory) -> bool:
        """Check whether an advisory with the same dedup key already exists."""
        key = advisory.dedup_key
        for entry in self._data.get(advisory.symbol, []):
            # Reconstruct dedup key from stored dict
            stored_key = _dedup_key_from_dict(entry)
            if stored_key == key:
                return True
        return False

    def clear(self, symbol: str | None = None) -> int:
        """Remove stored advisories.

        Parameters
        ----------
        symbol:
            If provided, clear only that symbol's history.  Otherwise
            clear all symbols.

        Returns
        -------
        int
            Number of advisories removed.
        """
        if symbol is not None:
            removed = len(self._data.pop(symbol, []))
        else:
            removed = sum(len(v) for v in self._data.values())
            self._data.clear()

        self._write()
        return removed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        """Load existing data from disk, returning empty dict on failure."""
        if not self._path.exists():
            return {}

        try:
            text = self._path.read_text(encoding="utf-8")
            raw = json.loads(text)
            if not isinstance(raw, dict):
                logger.warning("Alert store file is not a dict; starting fresh.")
                return {}
            return raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Error loading alert store %s: %s", self._path, exc)
            return {}

    def _write(self) -> None:
        """Persist current data to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, default=_json_default),
            encoding="utf-8",
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dedup_key_from_dict(entry: dict[str, Any]) -> str:
    """Reconstruct a dedup key from a stored advisory dict."""
    symbol = entry.get("symbol", "")
    action = entry.get("action", "")
    ts = entry.get("timestamp", "")
    date_part = ts[:10] if len(ts) >= 10 else ts
    return f"{symbol}:{action}:{date_part}"


def _json_default(obj: object) -> Any:
    """Fallback serializer for json.dumps."""
    from decimal import Decimal

    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
