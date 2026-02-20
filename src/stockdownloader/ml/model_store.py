"""Persist and retrieve trained ML models with metadata.

Uses ``joblib`` for model serialization and a companion JSON file for
human-readable metadata.

Usage::

    from stockdownloader.ml.model_store import ModelStore, ModelMetadata

    store = ModelStore("output/models/spy")
    path = store.save(trained_model, metadata)
    model, meta = store.load(path)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import joblib

    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    _HAS_JOBLIB = False

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Companion metadata stored alongside a persisted model."""

    symbol: str
    model_type: str
    feature_names: tuple[str, ...]
    training_date_range: tuple[str, str]
    oos_accuracy: float
    oos_roc_auc: float
    class_distribution: dict[str, int]
    trained_at: str
    config: dict[str, Any]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d["feature_names"] = list(d["feature_names"])
        d["training_date_range"] = list(d["training_date_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelMetadata:
        """Deserialize from a dict."""
        return cls(
            symbol=d["symbol"],
            model_type=d["model_type"],
            feature_names=tuple(d["feature_names"]),
            training_date_range=(
                d["training_date_range"][0],
                d["training_date_range"][1],
            ),
            oos_accuracy=d["oos_accuracy"],
            oos_roc_auc=d["oos_roc_auc"],
            class_distribution=d["class_distribution"],
            trained_at=d["trained_at"],
            config=d["config"],
        )


# ------------------------------------------------------------------
# Store
# ------------------------------------------------------------------


class ModelStore:
    """File-system store for trained models.

    Parameters
    ----------
    base_dir:
        Root directory for model files.
    """

    def __init__(self, base_dir: str | Path = "output/models") -> None:
        if not _HAS_JOBLIB:  # pragma: no cover
            raise ImportError(
                "joblib is required for model persistence. "
                "Install ML extras: pip install -e '.[ml]'"
            )
        self._base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, model: Any, metadata: ModelMetadata) -> Path:
        """Save model + metadata and return the model file path.

        Files are named ``{symbol}_{model_type}_{date}.joblib`` with
        a companion ``.json`` for metadata.
        """
        self._base_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"{metadata.symbol}_{metadata.model_type}_{date_str}"

        model_path = self._base_dir / f"{stem}.joblib"
        meta_path = self._base_dir / f"{stem}.json"

        joblib.dump(model, model_path)
        meta_path.write_text(
            json.dumps(metadata.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.info("Saved model to %s", model_path)
        return model_path

    def load(self, path: str | Path) -> tuple[Any, ModelMetadata]:
        """Load a model and its metadata from disk.

        Parameters
        ----------
        path:
            Path to the ``.joblib`` file.  The companion ``.json``
            is inferred automatically.

        Raises
        ------
        FileNotFoundError:
            If the model or metadata file does not exist.
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        meta_path = model_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        model = joblib.load(model_path)
        meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = ModelMetadata.from_dict(meta_dict)

        return model, metadata

    def latest_model(self, symbol: str) -> tuple[Any, ModelMetadata] | None:
        """Find and load the most recent model for *symbol*.

        Returns ``None`` if no matching model exists.
        """
        if not self._base_dir.exists():
            return None

        pattern = f"{symbol}_*.joblib"
        candidates = sorted(
            self._base_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not candidates:
            return None

        return self.load(candidates[0])
