"""Grand tournament -- strategy evaluation framework."""
from stockdownloader.app.tournament.cli import (
    main,
    main_baseline,
    main_greedy,
    main_walkforward,
)

__all__ = ["main", "main_walkforward", "main_baseline", "main_greedy"]
