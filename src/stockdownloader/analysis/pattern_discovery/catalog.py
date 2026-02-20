"""Pattern catalog management -- load, save, merge, and lookup.

Provides the :class:`PatternCatalog` dataclass for persisting and
querying validated patterns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stockdownloader.analysis.pattern_encoder import BarFeatures

from stockdownloader.analysis.pattern_discovery.models import (
    DiscoveredPattern,
    PatternKey,
)


@dataclass(frozen=True, slots=True)
class PatternCatalog:
    """Immutable catalog of discovered patterns, ready for live evaluation.

    Serializable to/from JSON for persistence between discovery runs.
    """

    patterns: tuple[DiscoveredPattern, ...]
    discovery_date: str = ""
    data_range: str = ""
    total_bars_scanned: int = 0
    total_patterns_tested: int = 0

    def lookup(
        self,
        recent_features: tuple[BarFeatures, ...],
    ) -> DiscoveredPattern | None:
        """Find the best matching pattern for the given recent bar features.

        Checks all pattern lengths (longest first for specificity).
        Returns the pattern with highest |t-stat| if multiple match.
        """
        best: DiscoveredPattern | None = None

        for pattern in self.patterns:
            n = len(pattern.key)
            if len(recent_features) < n:
                continue
            # Check if the last N features match this pattern's key
            if recent_features[-n:] == pattern.key:
                if best is None or abs(pattern.t_stat) > abs(best.t_stat):
                    best = pattern

        return best

    def to_dict(self) -> dict[str, Any]:
        """Serialize catalog to a JSON-compatible dict."""
        patterns_list = []
        for p in self.patterns:
            key_list = [
                {
                    "body_type": bf.body_type,
                    "body_strength": bf.body_strength,
                    "wick_signal": bf.wick_signal,
                    "relative_size": bf.relative_size,
                    "volume_profile": bf.volume_profile,
                }
                for bf in p.key
            ]
            patterns_list.append({
                "key": key_list,
                "direction": p.direction,
                "best_horizon": p.best_horizon,
                "avg_return": p.avg_return,
                "win_rate": p.win_rate,
                "occurrences": p.occurrences,
                "t_stat": p.t_stat,
                "p_value": p.p_value,
                "avg_mae": p.avg_mae,
                "avg_mfe": p.avg_mfe,
                "risk_reward": p.risk_reward,
                "walk_forward_stable": p.walk_forward_stable,
                "human_label": p.human_label,
                "confirmation": p.confirmation,
                "timeframe": p.timeframe,
                "effect_size": p.effect_size,
                "dominant_regime": p.dominant_regime,
                "regime_breakdown": p.regime_breakdown,
            })

        return {
            "patterns": patterns_list,
            "discovery_date": self.discovery_date,
            "data_range": self.data_range,
            "total_bars_scanned": self.total_bars_scanned,
            "total_patterns_tested": self.total_patterns_tested,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternCatalog:
        """Deserialize catalog from a JSON-compatible dict."""
        patterns = []
        for pd in d.get("patterns", []):
            key = tuple(
                BarFeatures(**bf) for bf in pd["key"]
            )
            patterns.append(DiscoveredPattern(
                key=key,
                direction=pd["direction"],
                best_horizon=pd["best_horizon"],
                avg_return=pd["avg_return"],
                win_rate=pd["win_rate"],
                occurrences=pd["occurrences"],
                t_stat=pd["t_stat"],
                p_value=pd["p_value"],
                avg_mae=pd["avg_mae"],
                avg_mfe=pd["avg_mfe"],
                risk_reward=pd["risk_reward"],
                walk_forward_stable=pd["walk_forward_stable"],
                human_label=pd["human_label"],
                confirmation=pd.get("confirmation"),
                timeframe=pd.get("timeframe", "5m"),
                effect_size=pd.get("effect_size", 0.0),
                dominant_regime=pd.get("dominant_regime"),
                regime_breakdown=pd.get("regime_breakdown"),
            ))

        return cls(
            patterns=tuple(patterns),
            discovery_date=d.get("discovery_date", ""),
            data_range=d.get("data_range", ""),
            total_bars_scanned=d.get("total_bars_scanned", 0),
            total_patterns_tested=d.get("total_patterns_tested", 0),
        )

    def save(self, path: Path) -> None:
        """Save catalog to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PatternCatalog:
        """Load catalog from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
