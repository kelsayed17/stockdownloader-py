"""PineScript strategy catalog -- application-layer PineScript code generation.

This package contains pre-built strategy definitions for Pine Script generation.
These modules import from the ``strategy/`` layer to delegate to Python strategy
``to_pinescript()`` methods, so they belong in ``app/`` (not ``util/``).

Public API::

    from stockdownloader.app.pinescript_catalog import (
        STRATEGY_CATALOG,
        COMPOSITE_STRATEGY_CATALOG,
    )
"""

from stockdownloader.app.pinescript_catalog.catalogs import (
    COMPOSITE_STRATEGY_CATALOG,
    STRATEGY_CATALOG,
)

__all__ = [
    "COMPOSITE_STRATEGY_CATALOG",
    "STRATEGY_CATALOG",
]
