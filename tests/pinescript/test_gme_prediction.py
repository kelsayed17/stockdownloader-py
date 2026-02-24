"""Tests for the GME Quantitative Regime Prediction PineScript strategy."""

from __future__ import annotations

import pytest

from stockdownloader.app.pinescript_catalog.gme_prediction import gme_prediction_strategy
from stockdownloader.app.pinescript_catalog.catalogs import STRATEGY_CATALOG
from stockdownloader.pinescript import PineScriptGenerator


class TestGmePredictionStrategyDefinition:
    """Verify the strategy definition builds correctly."""

    def test_builds_without_error(self) -> None:
        defn = gme_prediction_strategy()
        assert defn is not None
        assert defn.name == "GME Quant Regime Prediction"
        assert defn.short_name == "GME-QRP"

    def test_has_12_inputs(self) -> None:
        defn = gme_prediction_strategy()
        assert len(defn.inputs) == 12

    def test_input_names(self) -> None:
        defn = gme_prediction_strategy()
        names = [inp.name for inp in defn.inputs]
        expected = [
            "volWindow", "volAnnFactor", "regimeHighMult", "regimeExtremeMult",
            "volAnomalyWindow", "volAnomalyZ", "obvDivLookback",
            "clusterThreshold", "clusterCount",
            "atrLen", "rsiLen", "stopMult",
        ]
        assert names == expected

    def test_input_groups(self) -> None:
        defn = gme_prediction_strategy()
        groups = {inp.group for inp in defn.inputs}
        assert "Volatility Regime" in groups
        assert "Volume Anomaly" in groups
        assert "OBV Divergence" in groups
        assert "Volatility Clustering" in groups
        assert "Risk Management" in groups

    def test_has_indicators(self) -> None:
        defn = gme_prediction_strategy()
        # RSI, ATR, OBV
        assert len(defn.indicators) == 3
        var_names = [ind.var_name for ind in defn.indicators]
        assert "rsiVal" in var_names
        assert "atrVal" in var_names
        assert "obvRaw" in var_names

    def test_has_long_entry(self) -> None:
        defn = gme_prediction_strategy()
        assert defn.long_entry is not None
        assert "regime" in defn.long_entry.expr
        assert "volAnomaly" in defn.long_entry.expr
        assert "obvTrend" in defn.long_entry.expr
        assert "rsiVal" in defn.long_entry.expr

    def test_has_short_entry(self) -> None:
        defn = gme_prediction_strategy()
        assert defn.short_entry is not None
        assert "regime" in defn.short_entry.expr
        assert "volAnomaly" in defn.short_entry.expr
        assert "obvTrend" in defn.short_entry.expr

    def test_has_explicit_exits(self) -> None:
        defn = gme_prediction_strategy()
        assert defn.long_exit is not None
        assert defn.short_exit is not None
        # Extreme regime triggers exit
        assert "regime == 3" in defn.long_exit.expr
        assert "regime == 3" in defn.short_exit.expr
        # OBV divergence triggers exit
        assert "obvBearDiv" in defn.long_exit.expr
        assert "obvBullDiv" in defn.short_exit.expr

    def test_exit_on_reverse_disabled(self) -> None:
        defn = gme_prediction_strategy()
        assert defn.exit_on_reverse is False

    def test_has_extra_code(self) -> None:
        defn = gme_prediction_strategy()
        assert len(defn.extra_code) > 0
        code_text = "\n".join(defn.extra_code)
        assert "realizedVol" in code_text
        assert "regime" in code_text
        assert "volZScore" in code_text
        assert "obvBearDiv" in code_text
        assert "clusterActive" in code_text

    def test_has_extra_plots(self) -> None:
        defn = gme_prediction_strategy()
        assert len(defn.extra_plots) > 0
        plots_text = "\n".join(defn.extra_plots)
        assert "bgcolor" in plots_text
        assert "plotshape" in plots_text

    def test_has_description(self) -> None:
        defn = gme_prediction_strategy()
        assert "regime" in defn.description.lower()
        assert "volatility" in defn.description.lower()

    def test_custom_labels(self) -> None:
        defn = gme_prediction_strategy()
        assert defn.long_label == "Regime Buy"
        assert defn.short_label == "Regime Sell"


class TestGmePredictionPineScriptOutput:
    """Verify the generated Pine Script contains expected code."""

    def setup_method(self) -> None:
        self.gen = PineScriptGenerator()
        self.defn = gme_prediction_strategy()
        self.pine = self.gen.generate(self.defn)

    def test_generates_valid_pine(self) -> None:
        assert "//@version=6" in self.pine
        assert 'indicator("GME Quant Regime Prediction"' in self.pine
        assert "posState" in self.pine
        assert "alertcondition" in self.pine

    def test_has_all_inputs(self) -> None:
        assert "input.int(20" in self.pine  # volWindow
        assert "input.int(252" in self.pine  # volAnnFactor
        assert "input.float(1.5" in self.pine  # regimeHighMult
        assert "input.float(2.5" in self.pine  # regimeExtremeMult
        assert "input.int(60" in self.pine  # volAnomalyWindow
        assert "input.float(3.0" in self.pine  # volAnomalyZ
        assert "input.int(14" in self.pine  # obvDivLookback / atrLen / rsiLen
        assert "input.float(2.0" in self.pine  # clusterThreshold / stopMult

    def test_input_groups_rendered(self) -> None:
        assert "Volatility Regime" in self.pine
        assert "Volume Anomaly" in self.pine
        assert "OBV Divergence" in self.pine
        assert "Volatility Clustering" in self.pine
        assert "Risk Management" in self.pine

    # --- Volatility Regime ---

    def test_realized_vol_calculation(self) -> None:
        assert "math.log(close / close[1])" in self.pine
        assert "ta.stdev(logRet, volWindow)" in self.pine
        assert "math.sqrt(volAnnFactor)" in self.pine

    def test_median_vol_approximation(self) -> None:
        assert "ta.percentile_nearest_rank(realizedVol, 200, 50)" in self.pine

    def test_regime_classification(self) -> None:
        assert "highThreshold" in self.pine
        assert "extremeThreshold" in self.pine
        assert "lowThreshold" in self.pine
        assert "regimeHighMult" in self.pine
        assert "regimeExtremeMult" in self.pine

    def test_regime_transitions(self) -> None:
        assert "regimeUp" in self.pine
        assert "regimeDown" in self.pine

    # --- Volume Anomaly ---

    def test_volume_zscore(self) -> None:
        assert "volMean" in self.pine
        assert "volStd" in self.pine
        assert "volZScore" in self.pine
        assert "volAnomaly" in self.pine
        assert "volAnomalyZ" in self.pine

    def test_volume_zscore_handles_zero_std(self) -> None:
        assert "volStd > 0" in self.pine

    # --- OBV Divergence ---

    def test_obv_smoothing(self) -> None:
        assert "ta.ema(obvRaw, 5)" in self.pine
        assert "obvSmooth" in self.pine
        assert "obvTrend" in self.pine

    def test_obv_bearish_divergence(self) -> None:
        assert "obvBearDiv" in self.pine
        assert "priceHigh" in self.pine
        assert "obvHigh" in self.pine

    def test_obv_bullish_divergence(self) -> None:
        assert "obvBullDiv" in self.pine
        assert "priceLow" in self.pine
        assert "obvLow" in self.pine

    # --- Volatility Clustering ---

    def test_clustering_detection(self) -> None:
        assert "absRet" in self.pine
        assert "avgAbsRet" in self.pine
        assert "largeMove" in self.pine
        assert "clusterLen" in self.pine
        assert "clusterActive" in self.pine
        assert "clusterThreshold" in self.pine
        assert "clusterCount" in self.pine

    # --- Dynamic Stops ---

    def test_dynamic_stops(self) -> None:
        assert "regimeStopScale" in self.pine
        assert "dynamicStop" in self.pine
        assert "longStopPrice" in self.pine
        assert "shortStopPrice" in self.pine
        assert "stopMult" in self.pine

    def test_regime_scaled_stops(self) -> None:
        # Verify different scales for different regimes
        assert "regime == 3 ? 3.0" in self.pine  # extreme
        assert "regime == 2 ? 2.0" in self.pine  # high

    # --- Entry/Exit Conditions ---

    def test_entry_conditions_in_signal_logic(self) -> None:
        assert "longCondition" in self.pine
        assert "shortCondition" in self.pine

    def test_exit_conditions_in_signal_logic(self) -> None:
        assert "exitLongSignal" in self.pine
        assert "exitShortSignal" in self.pine

    # --- Visual Overlays ---

    def test_regime_background_coloring(self) -> None:
        assert "regimeBg" in self.pine
        assert 'bgcolor(regimeBg' in self.pine

    def test_volume_anomaly_markers(self) -> None:
        assert 'plotshape(volAnomaly' in self.pine
        assert "shape.diamond" in self.pine

    def test_obv_divergence_markers(self) -> None:
        assert 'plotshape(obvBearDiv' in self.pine
        assert 'plotshape(obvBullDiv' in self.pine
        assert "shape.triangledown" in self.pine
        assert "shape.triangleup" in self.pine

    def test_stop_level_plots(self) -> None:
        assert "longStopPrice" in self.pine
        assert "shortStopPrice" in self.pine
        assert "plot.style_linebr" in self.pine

    def test_data_window_metrics(self) -> None:
        assert "display.data_window" in self.pine
        assert 'title="Realized Vol %"' in self.pine
        assert 'title="Vol Regime (0-3)"' in self.pine
        assert 'title="Volume Z-Score"' in self.pine
        assert 'title="Cluster Length"' in self.pine

    # --- Alerts ---

    def test_has_alerts(self) -> None:
        # At minimum: buy, sell, exit long, exit short, combined
        assert self.pine.count("alertcondition") >= 5

    def test_alert_messages(self) -> None:
        assert "Regime Buy signal triggered" in self.pine
        assert "Regime Sell signal triggered" in self.pine
        assert "Exit Long signal" in self.pine
        assert "Exit Short signal" in self.pine

    # --- Labels ---

    def test_custom_labels(self) -> None:
        assert '"Regime Buy"' in self.pine
        assert '"Regime Sell"' in self.pine

    # --- Description Header ---

    def test_description_in_header(self) -> None:
        assert "Quantitative regime-adaptive" in self.pine
        assert "Student-t df=1.95" in self.pine
        assert "VaR 99% = -18.4%" in self.pine

    # --- Reasonable Size ---

    def test_reasonable_output_size(self) -> None:
        lines = self.pine.splitlines()
        assert len(lines) > 100, "Output should be substantial"
        assert len(self.pine) > 3000, "Output should be at least 3KB"


class TestGmePredictionInCatalog:
    """Verify the strategy is registered in the catalog."""

    def test_registered_in_catalog(self) -> None:
        assert "gme_prediction" in STRATEGY_CATALOG

    def test_catalog_factory_works(self) -> None:
        factory = STRATEGY_CATALOG["gme_prediction"]
        defn = factory()
        assert defn.name == "GME Quant Regime Prediction"

    def test_catalog_generates_pine(self) -> None:
        gen = PineScriptGenerator()
        factory = STRATEGY_CATALOG["gme_prediction"]
        defn = factory()
        pine = gen.generate(defn)
        assert "//@version=6" in pine
        assert "alertcondition" in pine

    def test_catalog_count_updated(self) -> None:
        """Catalog should now have 21 strategies (14 + GME + 3 SPY + 3 SPY v2)."""
        assert len(STRATEGY_CATALOG) == 21
