"""Tests for DatasetBuilder — labeled ML dataset construction."""

from __future__ import annotations

import random
from decimal import Decimal

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig, MLDataset
from stockdownloader.ml.feature_extractor import FeatureExtractor
from stockdownloader.core.models.price import PriceData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 300, seed: int = 42) -> list[PriceData]:
    rng = random.Random(seed)
    data: list[PriceData] = []
    price = 100.0
    for i in range(n):
        price += (rng.random() - 0.48) * 2
        price = max(50, price)
        h = price + rng.random() * 2
        l = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(l, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


DATA_300 = _make_data(300)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDatasetBuilderBuild:
    def test_shapes_match(self) -> None:
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext)
        ds = builder.build(DATA_300)

        assert ds.X.shape[0] == ds.y.shape[0]
        assert ds.X.shape[0] == len(ds.dates)
        assert ds.X.shape[1] == 63

    def test_labels_are_binary(self) -> None:
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext)
        ds = builder.build(DATA_300)

        assert set(np.unique(ds.y)).issubset({0, 1})

    def test_feature_names_match(self) -> None:
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext)
        ds = builder.build(DATA_300)

        assert ds.feature_names == ext.FEATURE_NAMES
        assert len(ds.feature_names) == ds.X.shape[1]

    def test_label_config_stored(self) -> None:
        cfg = LabelConfig(forward_period=10, profit_threshold=0.01)
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext, label_config=cfg)
        ds = builder.build(DATA_300)

        assert ds.label_config.forward_period == 10
        assert ds.label_config.profit_threshold == 0.01

    def test_warmup_exclusion(self) -> None:
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext, label_config=LabelConfig(forward_period=5))
        ds = builder.build(DATA_300, warmup=201)

        # First date in dataset should correspond to bar 201
        expected_first_date = DATA_300[201].date
        assert ds.dates[0] == expected_first_date

    def test_no_look_ahead(self) -> None:
        """Labels at index i use only data[i + forward_period]."""
        ext = FeatureExtractor()
        cfg = LabelConfig(forward_period=5, profit_threshold=0.0)
        builder = DatasetBuilder(ext, label_config=cfg)

        # Manually check first label
        idx = 201
        current = float(DATA_300[idx].close)
        future = float(DATA_300[idx + 5].close)
        expected_label = 1 if (future - current) / current > 0.0 else 0

        ds = builder.build(DATA_300, warmup=201)
        assert ds.y[0] == expected_label

    def test_no_nan_in_features(self) -> None:
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext)
        ds = builder.build(DATA_300)

        assert not np.any(np.isnan(ds.X))
        assert not np.any(np.isinf(ds.X))


class TestDatasetBuilderErrors:
    def test_insufficient_data_raises(self) -> None:
        ext = FeatureExtractor()
        builder = DatasetBuilder(ext, label_config=LabelConfig(forward_period=5))
        short_data = _make_data(205)

        with pytest.raises(ValueError, match="Need at least"):
            builder.build(short_data, warmup=201)


class TestLabelConfig:
    def test_defaults(self) -> None:
        cfg = LabelConfig()
        assert cfg.forward_period == 10
        assert cfg.profit_threshold == 0.005
        assert cfg.use_atr_threshold is False
        assert cfg.atr_multiplier == 0.5

    def test_custom(self) -> None:
        cfg = LabelConfig(forward_period=10, profit_threshold=0.01)
        assert cfg.forward_period == 10

    def test_atr_threshold_config(self) -> None:
        cfg = LabelConfig(use_atr_threshold=True, atr_multiplier=0.8)
        assert cfg.use_atr_threshold is True
        assert cfg.atr_multiplier == 0.8


class TestATRRelativeLabels:
    def test_atr_labels_produce_fewer_positives(self) -> None:
        """ATR-relative threshold should be more selective (fewer class-1)."""
        ext = FeatureExtractor()
        # Build with standard threshold
        cfg_std = LabelConfig(forward_period=5, profit_threshold=0.0)
        builder_std = DatasetBuilder(ext, label_config=cfg_std)
        ds_std = builder_std.build(DATA_300, warmup=201)

        # Build with ATR-relative threshold
        cfg_atr = LabelConfig(
            forward_period=5, use_atr_threshold=True, atr_multiplier=0.5,
        )
        builder_atr = DatasetBuilder(ext, label_config=cfg_atr)
        ds_atr = builder_atr.build(DATA_300, warmup=201)

        pos_std = int(np.sum(ds_std.y == 1))
        pos_atr = int(np.sum(ds_atr.y == 1))
        # ATR threshold should produce same or fewer positives
        assert pos_atr <= pos_std

    def test_atr_labels_still_binary(self) -> None:
        ext = FeatureExtractor()
        cfg = LabelConfig(
            forward_period=5, use_atr_threshold=True, atr_multiplier=0.5,
        )
        builder = DatasetBuilder(ext, label_config=cfg)
        ds = builder.build(DATA_300, warmup=201)
        assert set(np.unique(ds.y)).issubset({0, 1})

    def test_higher_threshold_fewer_positives(self) -> None:
        """Higher profit threshold should yield fewer positives."""
        ext = FeatureExtractor()
        cfg_lo = LabelConfig(forward_period=5, profit_threshold=0.0)
        cfg_hi = LabelConfig(forward_period=5, profit_threshold=0.02)
        ds_lo = DatasetBuilder(ext, label_config=cfg_lo).build(DATA_300, warmup=201)
        ds_hi = DatasetBuilder(ext, label_config=cfg_hi).build(DATA_300, warmup=201)
        assert int(np.sum(ds_hi.y == 1)) <= int(np.sum(ds_lo.y == 1))
