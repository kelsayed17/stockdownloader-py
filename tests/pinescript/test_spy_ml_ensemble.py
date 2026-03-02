"""Tests for SPY ML Ensemble PineScript strategy with dynamic exits.

Uses a synthetic trained DeepSurrogateExporter to validate that the
strategy factory produces correct StrategyDefinition objects and that
the generated Pine Script contains required elements.
"""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
from stockdownloader.pinescript.ml_export import FEATURE_PINE_MAP
from stockdownloader.pinescript.models import StrategyDefinition


# ------------------------------------------------------------------
# Helpers -- synthetic trained exporter
# ------------------------------------------------------------------

# Use real feature names that exist in FEATURE_PINE_MAP.
_FEATURE_NAMES = (
    "rsi14_norm",
    "stoch_k_norm",
    "stoch_d_norm",
    "adx14_norm",
    "plus_di_norm",
    "minus_di_norm",
    "mfi14_norm",
    "williams_r_norm",
    "cci20_norm",
    "bb_percent_b",
    "bb_width_norm",
    "roc12_clamped",
    "macd_hist_atr",
    "macd_signal_spread_atr",
    "price_vs_sma20_atr",
)

_N_FEATURES = len(_FEATURE_NAMES)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _make_trained_exporter(
    n: int = 500,
    max_depth: int = 4,
    top_n: int = 10,
    seed: int = 42,
) -> DeepSurrogateExporter:
    """Create and train a DeepSurrogateExporter on synthetic data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, _N_FEATURES)
    y = (rng.rand(n) > 0.5).astype(np.int64)
    dates = tuple(f"2024-01-{i + 1:04d}" for i in range(n))

    # Ensemble probs = sigmoid(2*f0 + 1.5*f1 - f2 + noise)
    logit = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2]
    noise = rng.randn(n) * 0.3
    ensemble_probs = _sigmoid(logit + noise)

    dataset = MLDataset(
        X=X,
        y=y,
        dates=dates,
        feature_names=_FEATURE_NAMES,
        label_config=LabelConfig(),
    )

    importances: dict[str, float] = {}
    for i, name in enumerate(_FEATURE_NAMES):
        if i == 0:
            importances[name] = 0.35
        elif i == 1:
            importances[name] = 0.25
        elif i == 2:
            importances[name] = 0.15
        else:
            importances[name] = 0.01

    exporter = DeepSurrogateExporter(
        max_depth=max_depth,
        min_samples_leaf=10,
        top_n=top_n,
    )
    exporter.train_surrogate(
        dataset, importances, ensemble_probs=ensemble_probs,
    )
    return exporter


@pytest.fixture
def trained_exporter() -> DeepSurrogateExporter:
    """Trained DeepSurrogateExporter for tests."""
    return _make_trained_exporter()


@pytest.fixture
def strategy_def(trained_exporter: DeepSurrogateExporter) -> StrategyDefinition:
    """StrategyDefinition built from trained exporter."""
    from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
        spy_ml_ensemble_strategy,
    )

    return spy_ml_ensemble_strategy(trained_exporter)


@pytest.fixture
def pine_code(strategy_def: StrategyDefinition) -> str:
    """Generated Pine Script code from the strategy definition."""
    from stockdownloader.pinescript import PineScriptGenerator

    return PineScriptGenerator().generate(strategy_def)


# ------------------------------------------------------------------
# Test: Factory builds without error
# ------------------------------------------------------------------


class TestStrategyDefBuilds:
    """The factory function produces a valid StrategyDefinition."""

    def test_builds_without_error(
        self, trained_exporter: DeepSurrogateExporter,
    ) -> None:
        from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
            spy_ml_ensemble_strategy,
        )

        sd = spy_ml_ensemble_strategy(trained_exporter)
        assert isinstance(sd, StrategyDefinition)

    def test_raises_if_not_trained(self) -> None:
        from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
            spy_ml_ensemble_strategy,
        )

        exporter = DeepSurrogateExporter()
        with pytest.raises(RuntimeError, match="trained"):
            spy_ml_ensemble_strategy(exporter)


# ------------------------------------------------------------------
# Test: Name and short_name
# ------------------------------------------------------------------


class TestNameFields:
    """Strategy has expected name and short_name."""

    def test_name(self, strategy_def: StrategyDefinition) -> None:
        assert strategy_def.name == "SPY ML Ensemble"

    def test_short_name(self, strategy_def: StrategyDefinition) -> None:
        assert strategy_def.short_name == "SPY-ML-ENS"


# ------------------------------------------------------------------
# Test: Strategy mode
# ------------------------------------------------------------------


class TestStrategyMode:
    """strategy_mode is True."""

    def test_strategy_mode_is_true(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        assert strategy_def.strategy_mode is True


# ------------------------------------------------------------------
# Test: Required inputs
# ------------------------------------------------------------------


class TestInputs:
    """Strategy has all required user-configurable inputs."""

    def _input_names(self, sd: StrategyDefinition) -> set[str]:
        return {inp.name for inp in sd.inputs}

    def test_has_buy_thresh(self, strategy_def: StrategyDefinition) -> None:
        assert "buyThresh" in self._input_names(strategy_def)

    def test_has_sell_thresh(self, strategy_def: StrategyDefinition) -> None:
        assert "sellThresh" in self._input_names(strategy_def)

    def test_has_base_sl(self, strategy_def: StrategyDefinition) -> None:
        assert "baseSL" in self._input_names(strategy_def)

    def test_has_base_tp(self, strategy_def: StrategyDefinition) -> None:
        assert "baseTP" in self._input_names(strategy_def)

    def test_has_atr_len(self, strategy_def: StrategyDefinition) -> None:
        assert "atrLen" in self._input_names(strategy_def)

    def test_has_sl_cap(self, strategy_def: StrategyDefinition) -> None:
        assert "slCapDollars" in self._input_names(strategy_def)

    def test_has_circuit_max(self, strategy_def: StrategyDefinition) -> None:
        assert "circuitMax" in self._input_names(strategy_def)

    def test_has_vwap_confirm(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        assert "useVwapConfirm" in self._input_names(strategy_def)

    def test_has_close_eod(self, strategy_def: StrategyDefinition) -> None:
        assert "closeEOD" in self._input_names(strategy_def)

    def test_buy_thresh_default(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        inp = next(i for i in strategy_def.inputs if i.name == "buyThresh")
        assert inp.default == 0.55

    def test_sell_thresh_default(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        inp = next(i for i in strategy_def.inputs if i.name == "sellThresh")
        assert inp.default == 0.45


# ------------------------------------------------------------------
# Test: Entry conditions
# ------------------------------------------------------------------


class TestEntryConditions:
    """Entry conditions reference expected signal variables."""

    def test_long_entry_expr(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        assert strategy_def.long_entry is not None
        assert "buySignal" in strategy_def.long_entry.expr

    def test_short_entry_expr(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        assert strategy_def.short_entry is not None
        assert "sellSignal" in strategy_def.short_entry.expr


# ------------------------------------------------------------------
# Test: Custom thresholds
# ------------------------------------------------------------------


class TestCustomThresholds:
    """Factory respects custom buy/sell threshold arguments."""

    def test_custom_thresholds(
        self, trained_exporter: DeepSurrogateExporter,
    ) -> None:
        from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
            spy_ml_ensemble_strategy,
        )

        sd = spy_ml_ensemble_strategy(
            trained_exporter,
            buy_threshold=0.70,
            sell_threshold=0.30,
        )
        buy_inp = next(i for i in sd.inputs if i.name == "buyThresh")
        sell_inp = next(i for i in sd.inputs if i.name == "sellThresh")
        assert buy_inp.default == 0.70
        assert sell_inp.default == 0.30


# ------------------------------------------------------------------
# Test: Pine Script generation contains key strings
# ------------------------------------------------------------------


class TestPineCodeContent:
    """Generated Pine Script contains required elements."""

    def test_has_strategy_header(self, pine_code: str) -> None:
        assert 'strategy("SPY ML Ensemble"' in pine_code

    def test_has_ml_prob(self, pine_code: str) -> None:
        assert "float mlProb" in pine_code

    def test_has_tn0(self, pine_code: str) -> None:
        assert "tn0" in pine_code

    def test_has_confidence(self, pine_code: str) -> None:
        assert "confidence" in pine_code

    def test_has_dynamic_sl(self, pine_code: str) -> None:
        assert "dynamicSL" in pine_code

    def test_has_dynamic_tp(self, pine_code: str) -> None:
        assert "dynamicTP" in pine_code

    def test_has_bgcolor(self, pine_code: str) -> None:
        assert "bgcolor" in pine_code

    def test_has_strategy_entry(self, pine_code: str) -> None:
        assert "strategy.entry" in pine_code

    def test_has_strategy_exit(self, pine_code: str) -> None:
        assert "strategy.exit" in pine_code

    def test_has_label(self, pine_code: str) -> None:
        assert "label.new" in pine_code

    def test_has_vwap_request(self, pine_code: str) -> None:
        assert "request.security" in pine_code
        assert "ta.vwap" in pine_code

    def test_has_buy_signal(self, pine_code: str) -> None:
        assert "buySignal" in pine_code

    def test_has_sell_signal(self, pine_code: str) -> None:
        assert "sellSignal" in pine_code

    def test_has_version_6(self, pine_code: str) -> None:
        assert "//@version=6" in pine_code

    def test_no_indicator_declaration(self, pine_code: str) -> None:
        """Must NOT contain indicator( declaration (strategy mode only)."""
        assert "indicator(" not in pine_code

    def test_has_ml_probability_plot(self, pine_code: str) -> None:
        assert "ML Probability" in pine_code

    def test_has_ml_confidence_plot(self, pine_code: str) -> None:
        assert "ML Confidence" in pine_code


# ------------------------------------------------------------------
# Test: extra_code contains ML tree elements
# ------------------------------------------------------------------


class TestExtraCodeContent:
    """The extra_code list has expected ML tree injection elements."""

    def test_extra_code_has_ml_prob(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_code)
        assert "float mlProb = tn0" in text

    def test_extra_code_has_tree_nodes(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_code)
        assert "float tn" in text

    def test_extra_code_has_confidence(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_code)
        assert "confidence" in text

    def test_extra_code_has_vwap(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_code)
        assert "vwap15" in text

    def test_extra_code_has_buy_signal(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_code)
        assert "bool buySignal" in text

    def test_extra_code_has_sell_signal(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_code)
        assert "bool sellSignal" in text


# ------------------------------------------------------------------
# Test: extra_plots contains dynamic exit elements
# ------------------------------------------------------------------


class TestExtraPlotsContent:
    """The extra_plots list has expected dynamic exit elements."""

    def test_extra_plots_has_dynamic_sl(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "dynamicSL" in text

    def test_extra_plots_has_dynamic_tp(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "dynamicTP" in text

    def test_extra_plots_has_strategy_exit(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "strategy.exit" in text

    def test_extra_plots_has_bgcolor(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "bgcolor" in text

    def test_extra_plots_has_label(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "label.new" in text

    def test_extra_plots_has_entry_price_plot(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "Entry Price" in text

    def test_extra_plots_has_take_profit_plot(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "Take Profit" in text

    def test_extra_plots_has_stop_loss_plot(
        self, strategy_def: StrategyDefinition,
    ) -> None:
        text = "\n".join(strategy_def.extra_plots)
        assert "Stop Loss" in text


# ------------------------------------------------------------------
# Test: Full pipeline generates valid Pine Script
# ------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: train exporter -> build strategy -> generate Pine."""

    def test_full_pipeline_produces_string(self, pine_code: str) -> None:
        assert isinstance(pine_code, str)
        assert len(pine_code) > 500

    def test_pine_code_starts_with_comment(self, pine_code: str) -> None:
        assert pine_code.startswith("// This Pine Script")

    def test_pine_code_has_strategy_params(self, pine_code: str) -> None:
        assert "initial_capital=" in pine_code
        assert "overlay=true" in pine_code
