"""Grand tournament -- re-exports from tournament package.

All logic has moved to :mod:`stockdownloader.app.tournament`.  This
thin wrapper keeps the ``pyproject.toml`` entry-point paths
(``stockdownloader.app.tournament_app:main``) working.
"""
from stockdownloader.app.tournament.cli import (  # noqa: F401
    _parse_args,
    main,
    main_baseline,
    main_greedy,
    main_walkforward,
)
from stockdownloader.app.tournament.helpers import (  # noqa: F401
    _box_title,
    _build_skip_set,
    _status_label,
)

__all__ = ["main", "main_walkforward", "main_baseline", "main_greedy"]
