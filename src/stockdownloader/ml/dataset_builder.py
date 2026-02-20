"""Builds labeled training datasets from historical price data.

Loops over historical bars, extracts feature vectors via
:class:`~stockdownloader.ml.feature_extractor.FeatureExtractor`, and
labels each bar from the N-bar forward return.

Usage::

    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.feature_extractor import FeatureExtractor

    builder = DatasetBuilder(FeatureExtractor())
    dataset = builder.build(daily_data)
    print(dataset.X.shape)  # (n_samples, 63)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from stockdownloader.ml.feature_extractor import FeatureExtractor
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Label construction parameters.

    Attributes
    ----------
    forward_period:
        Number of bars forward to compute the return.
    profit_threshold:
        Minimum return to be labeled class 1 (profitable).
    use_atr_threshold:
        When ``True``, label positive if forward return exceeds
        ``atr_multiplier * ATR / close`` instead of *profit_threshold*.
    atr_multiplier:
        How many ATRs the forward return must exceed (only used when
        *use_atr_threshold* is ``True``).
    """

    forward_period: int = 10
    profit_threshold: float = 0.005
    use_atr_threshold: bool = False
    atr_multiplier: float = 0.5


@dataclass(frozen=True, slots=True)
class MLDataset:
    """Training dataset with features, labels, and metadata.

    Attributes
    ----------
    X:
        Feature matrix — numpy ndarray of shape ``(n_samples, n_features)``.
    y:
        Label vector — numpy ndarray of shape ``(n_samples,)``.  Binary:
        1 = profitable, 0 = not profitable.
    dates:
        Corresponding bar dates.
    feature_names:
        Feature column names matching ``X`` columns.
    label_config:
        The :class:`LabelConfig` used to construct labels.
    """

    X: Any  # numpy ndarray
    y: Any  # numpy ndarray
    dates: tuple[str, ...]
    feature_names: tuple[str, ...]
    label_config: LabelConfig


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------


class DatasetBuilder:
    """Builds ``(X, y)`` arrays from price data for ML training.

    Parameters
    ----------
    extractor:
        Feature extractor instance.
    label_config:
        Label construction configuration.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        label_config: LabelConfig | None = None,
        hub: IndicatorHub | None = None,
    ) -> None:
        if np is None:
            raise ImportError(
                "numpy is required for ML features. "
                "Install: pip install -e '.[ml]'"
            )
        self._extractor = extractor
        self._label_config = label_config or LabelConfig()
        self._hub = hub or IndicatorHub()

    def build(
        self,
        data: list[PriceData],
        warmup: int = 201,
    ) -> MLDataset:
        """Build the full dataset from price data.

        Parameters
        ----------
        data:
            Daily price bars.  Needs at least
            ``warmup + forward_period + 1`` bars.
        warmup:
            First bar index to extract features from.

        Returns
        -------
        MLDataset

        Raises
        ------
        ValueError
            If there isn't enough data.
        """
        cfg = self._label_config
        min_bars = warmup + cfg.forward_period + 1

        if len(data) < min_bars:
            raise ValueError(
                f"Need at least {min_bars} bars, got {len(data)}"
            )

        last_feature_index = len(data) - cfg.forward_period - 1

        rows: list[list[float]] = []
        labels: list[int] = []
        dates: list[str] = []

        for i in range(warmup, last_feature_index + 1):
            fv = self._extractor.extract(data, i)
            label = self._compute_label(data, i, cfg, self._hub)

            rows.append(list(fv.values))
            labels.append(label)
            dates.append(fv.bar_date)

        X = np.array(rows, dtype=np.float64)
        y = np.array(labels, dtype=np.int64)

        logger.info(
            "Built dataset: %d samples, %d features, class distribution: %s",
            len(rows),
            X.shape[1] if X.ndim > 1 else 0,
            dict(zip(*np.unique(y, return_counts=True))),
        )

        return MLDataset(
            X=X,
            y=y,
            dates=tuple(dates),
            feature_names=self._extractor.FEATURE_NAMES,
            label_config=cfg,
        )

    @staticmethod
    def _compute_label(
        data: list[PriceData],
        index: int,
        config: LabelConfig,
        hub: IndicatorHub | None = None,
    ) -> int:
        """Compute binary label for bar at *index*.

        Label = 1 if forward return > threshold, else 0.
        When ``config.use_atr_threshold`` is True, the threshold is
        ``atr_multiplier * ATR(14) / close`` instead of *profit_threshold*.
        """
        current_close = float(data[index].close)
        future_close = float(data[index + config.forward_period].close)

        if current_close == 0:
            return 0

        forward_return = (future_close - current_close) / current_close

        if config.use_atr_threshold and hub is not None:
            atr_val = float(hub.atr(data, index, 14))
            threshold = config.atr_multiplier * atr_val / current_close if current_close != 0 else 0.0
        else:
            threshold = config.profit_threshold

        return 1 if forward_return > threshold else 0
