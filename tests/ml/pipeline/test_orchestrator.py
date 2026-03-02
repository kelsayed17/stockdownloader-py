"""Tests for the Pipeline Orchestrator."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import patch

import pytest

from stockdownloader.ml.pipeline.config import (
    BacktestConfig,
    DataConfig,
    HybridConfig,
    PipelineConfig,
    SelectionConfig,
    TrainingGridConfig,
)
from stockdownloader.ml.pipeline.orchestrator import MLPipelineOrchestrator
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
        low = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(low, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestMLPipelineOrchestrator:
    @patch("stockdownloader.app.helpers.fetch_daily_data")
    def test_full_pipeline_quick(
        self, mock_fetch: object, tmp_path: object,
    ) -> None:
        """Run the full pipeline in quick mode with mocked data."""
        mock_fetch.return_value = _make_data(300)  # type: ignore[attr-defined]

        config = PipelineConfig(
            data=DataConfig(
                symbol="SPY",
                range_="5y",
                use_cache=False,
            ),
            training=TrainingGridConfig(
                forward_periods=(5,),
                profit_thresholds=(0.0,),
                model_types=("gradient_boosting",),
                use_class_balance_options=(False,),
                use_atr_labels_options=(False,),
                n_estimators=10,
                max_depth=2,
            ),
            hybrid=HybridConfig(
                modes=("confirmed",),
                top_pairs=5,
            ),
            backtest=BacktestConfig(
                walk_forward_windows=0,
                top_for_walk_forward=0,
            ),
            selection=SelectionConfig(
                top_n=3,
                export_pine=False,
                output_dir=str(tmp_path),
            ),
        )

        output: list[str] = []
        pipeline = MLPipelineOrchestrator(
            config, print_fn=lambda *a: output.append(str(a)),
        )
        result = pipeline.run()

        # Verify pipeline completed
        assert result.data_result is not None
        assert result.data_result.bar_count == 300
        assert result.training_result is not None
        assert len(result.training_result.candidates) >= 1
        assert result.convergence_result is not None
        assert result.hybrid_result is not None
        assert result.backtest_result is not None
        assert len(result.backtest_result.entries) >= 1

        # Verify output has stage markers
        full_output = " ".join(output)
        assert "STAGE 1" in full_output
        assert "STAGE 2" in full_output
        assert "STAGE 3" in full_output
        assert "STAGE 4" in full_output
        assert "STAGE 5" in full_output
        assert "STAGE 6" in full_output
        assert "PIPELINE COMPLETE" in full_output

    @patch("stockdownloader.app.helpers.fetch_daily_data")
    def test_pipeline_no_data_aborts(self, mock_fetch: object) -> None:
        """Pipeline should raise if no data is fetched."""
        mock_fetch.return_value = []  # type: ignore[attr-defined]
        config = PipelineConfig(
            data=DataConfig(use_cache=False),
        )
        pipeline = MLPipelineOrchestrator(
            config, print_fn=lambda *a, **k: None,
        )
        with pytest.raises(RuntimeError, match="Could not fetch"):
            pipeline.run()

    @patch("stockdownloader.app.helpers.fetch_daily_data")
    def test_pipeline_with_pine_export(
        self, mock_fetch: object, tmp_path: object,
    ) -> None:
        """Pipeline exports PineScript when configured."""
        mock_fetch.return_value = _make_data(300)  # type: ignore[attr-defined]

        config = PipelineConfig(
            data=DataConfig(use_cache=False),
            training=TrainingGridConfig(
                forward_periods=(5,),
                profit_thresholds=(0.0,),
                model_types=("gradient_boosting",),
                use_class_balance_options=(False,),
                use_atr_labels_options=(False,),
                n_estimators=10,
                max_depth=2,
            ),
            hybrid=HybridConfig(
                modes=("confirmed",),
                top_pairs=3,
            ),
            backtest=BacktestConfig(
                walk_forward_windows=0,
                top_for_walk_forward=0,
            ),
            selection=SelectionConfig(
                top_n=3,
                export_pine=True,
                pine_depth=3,
                pine_top_features=5,
                output_dir=str(tmp_path),
            ),
        )

        pipeline = MLPipelineOrchestrator(
            config, print_fn=lambda *a, **k: None,
        )
        result = pipeline.run()

        assert result.pine_script is not None
        assert "//@version=6" in result.pine_script
        assert result.pine_path is not None
