"""Tests for ModelStore — model persistence and retrieval."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from stockdownloader.ml.model_store import ModelMetadata, ModelStore


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _metadata(symbol: str = "SPY") -> ModelMetadata:
    return ModelMetadata(
        symbol=symbol,
        model_type="gradient_boosting",
        feature_names=("feat_0", "feat_1", "feat_2"),
        training_date_range=("2024-01-01", "2024-12-31"),
        oos_accuracy=0.62,
        oos_roc_auc=0.65,
        class_distribution={"0": 120, "1": 80},
        trained_at="2024-12-31T23:59:59+00:00",
        config={"n_estimators": 50, "max_depth": 3},
    )


def _dummy_model() -> GradientBoostingClassifier:
    """Train a tiny model for serialization tests."""
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    return model


# ------------------------------------------------------------------
# Save / load round-trip
# ------------------------------------------------------------------


class TestModelStoreSaveLoad:
    def test_save_and_load(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        model = _dummy_model()
        meta = _metadata()

        path = store.save(model, meta)
        assert path.exists()
        assert path.suffix == ".joblib"

        loaded_model, loaded_meta = store.load(path)
        assert loaded_meta.symbol == "SPY"
        assert loaded_meta.oos_accuracy == 0.62

    def test_predictions_preserved(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        model = _dummy_model()
        meta = _metadata()

        rng = np.random.RandomState(42)
        X_test = rng.randn(10, 3)
        original_preds = model.predict(X_test)

        path = store.save(model, meta)
        loaded_model, _ = store.load(path)
        loaded_preds = loaded_model.predict(X_test)

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_metadata_json_file_created(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        path = store.save(_dummy_model(), _metadata())

        json_path = path.with_suffix(".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["symbol"] == "SPY"
        assert data["model_type"] == "gradient_boosting"

    def test_creates_parent_dirs(self, tmp_path: object) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"  # type: ignore[operator]
        store = ModelStore(base_dir=deep_dir)
        path = store.save(_dummy_model(), _metadata())
        assert path.exists()


# ------------------------------------------------------------------
# Metadata serialization
# ------------------------------------------------------------------


class TestModelMetadata:
    def test_round_trip(self) -> None:
        meta = _metadata()
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)

        assert restored.symbol == meta.symbol
        assert restored.feature_names == meta.feature_names
        assert restored.training_date_range == meta.training_date_range
        assert restored.oos_accuracy == meta.oos_accuracy
        assert restored.config == meta.config

    def test_feature_names_as_list_in_dict(self) -> None:
        meta = _metadata()
        d = meta.to_dict()
        # Should be list (JSON-compatible), not tuple
        assert isinstance(d["feature_names"], list)
        assert isinstance(d["training_date_range"], list)


# ------------------------------------------------------------------
# latest_model
# ------------------------------------------------------------------


class TestModelStoreLatest:
    def test_latest_model_found(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]

        # Save two models for same symbol
        import time

        store.save(_dummy_model(), _metadata(symbol="SPY"))
        time.sleep(0.05)  # ensure different mtime
        store.save(_dummy_model(), _metadata(symbol="SPY"))

        result = store.latest_model("SPY")
        assert result is not None
        _, meta = result
        assert meta.symbol == "SPY"

    def test_latest_model_none_when_empty(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        assert store.latest_model("SPY") is None

    def test_latest_model_none_when_no_dir(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path / "nonexistent")  # type: ignore[operator]
        assert store.latest_model("SPY") is None

    def test_latest_model_different_symbol(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        store.save(_dummy_model(), _metadata(symbol="SPY"))

        assert store.latest_model("AAPL") is None


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestModelStoreErrors:
    def test_load_missing_model(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            store.load(tmp_path / "nonexistent.joblib")  # type: ignore[operator]

    def test_load_missing_metadata(self, tmp_path: object) -> None:
        store = ModelStore(base_dir=tmp_path)  # type: ignore[arg-type]
        # Create model file without metadata
        import joblib

        model_path = tmp_path / "test.joblib"  # type: ignore[operator]
        joblib.dump(_dummy_model(), model_path)

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            store.load(model_path)
