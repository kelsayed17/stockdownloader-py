"""Export ML model decisions as a TradingView Pine Script v6 indicator.

Since sklearn models cannot run in Pine Script, this module trains a
simplified :class:`~sklearn.tree.DecisionTreeClassifier` as a *surrogate*
of the full ensemble model.  The tree's decision path is then translated
into nested Pine Script ternary expressions that approximate the model's
predicted probability using only native TradingView indicators.

Usage::

    from stockdownloader.util.pinescript.ml_export import (
        DecisionTreeExporter,
        ml_signal_strategy,
    )

    exporter = DecisionTreeExporter()
    exporter.train_surrogate(dataset, feature_importances, top_n=15)
    strategy = ml_signal_strategy("SPY", exporter)

    from stockdownloader.util.pinescript import PineScriptGenerator
    pine = PineScriptGenerator().generate(strategy)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, TYPE_CHECKING

from stockdownloader.util.pinescript.models import (
    Condition,
    Indicator,
    Input,
    StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.ml.dataset_builder import MLDataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Feature → Pine Script mapping
# ------------------------------------------------------------------

# Each entry is (pine_expression, list_of_pine_variable_declarations).
# Dependencies are declared once in the indicator section; the expression
# references those pre-computed variables.

FEATURE_PINE_MAP: dict[str, tuple[str, list[str]]] = {
    # Group A: Normalized indicators (26)
    # NOTE: Pine Script v6 does not allow variable names starting with
    # underscores, so all local variables use a 'p' prefix instead.
    "rsi14_norm": ("ta.rsi(close, 14) / 100.0", []),
    "stoch_k_norm": ("ta.stoch(close, high, low, 14) / 100.0", []),
    "stoch_d_norm": ("ta.sma(ta.stoch(close, high, low, 14), 3) / 100.0", []),
    "adx14_norm": ("pAdxVal / 100.0", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "plus_di_norm": ("pDiPlus / 100.0", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "minus_di_norm": ("pDiMinus / 100.0", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "mfi14_norm": ("ta.mfi(hlc3, 14) / 100.0", []),
    "williams_r_norm": ("ta.wpr(14) / 100.0", []),
    "cci20_norm": ("math.max(-2.0, math.min(2.0, ta.cci(close, 20) / 200.0))", []),
    "bb_percent_b": ("pBbPctB", [
        "[pBbMid, pBbUp, pBbLo] = ta.bb(close, 20, 2.0)",
        "float pBbPctB = (pBbUp - pBbLo) != 0 ? (close - pBbLo) / (pBbUp - pBbLo) : 0.5",
    ]),
    "bb_width_norm": ("(pBbUp - pBbLo) / close", [
        "[pBbMid, pBbUp, pBbLo] = ta.bb(close, 20, 2.0)",
    ]),
    "roc12_clamped": ("math.max(-10.0, math.min(10.0, ta.roc(close, 12)))", []),
    "macd_hist_atr": ("pMHist / pAtr14", [
        "[pMLine, pMSig, pMHist] = ta.macd(close, 12, 26, 9)",
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "macd_signal_spread_atr": ("(pMLine - pMSig) / pAtr14", [
        "[pMLine, pMSig, pMHist] = ta.macd(close, 12, 26, 9)",
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "price_vs_sma20_atr": ("(close - ta.sma(close, 20)) / pAtr14", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "price_vs_sma50_atr": ("(close - ta.sma(close, 50)) / pAtr14", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "price_vs_sma200_atr": ("(close - ta.sma(close, 200)) / pAtr14", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "price_vs_vwap_atr": ("(close - ta.vwap) / pAtr14", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "ema_cross_atr": ("(ta.ema(close, 12) - ta.ema(close, 26)) / pAtr14", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "volume_ratio": ("volume / ta.sma(volume, 20)", []),
    "atr_pct": ("pAtr14 / close", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "fib_position": ("pFibPos", [
        "float pFibHi = ta.highest(high, 50)",
        "float pFibLo = ta.lowest(low, 50)",
        "float pFibPos = (pFibHi - pFibLo) != 0 ? (close - pFibLo) / (pFibHi - pFibLo) : 0.5",
    ]),
    "bb_position": ("pBbPctB", [
        "[pBbMid, pBbUp, pBbLo] = ta.bb(close, 20, 2.0)",
        "float pBbPctB = (pBbUp - pBbLo) != 0 ? (close - pBbLo) / (pBbUp - pBbLo) : 0.5",
    ]),
    "obv_rising_flag": ("(ta.obv > ta.obv[5]) ? 1.0 : 0.0", []),
    "sar_bullish_flag": ("(close > ta.sar(0.02, 0.02, 0.2)) ? 1.0 : 0.0", []),
    "price_above_cloud_flag": ("pAboveCloud", [
        "float pTenkan = (ta.highest(high, 9) + ta.lowest(low, 9)) / 2",
        "float pKijun = (ta.highest(high, 26) + ta.lowest(low, 26)) / 2",
        "float pSenkouA = ((pTenkan + pKijun) / 2)",
        "float pSenkouB = (ta.highest(high, 52) + ta.lowest(low, 52)) / 2",
        "float pCloudTop = math.max(pSenkouA, pSenkouB)",
        "float pAboveCloud = close > pCloudTop ? 1.0 : 0.0",
    ]),
    # Group B: Signal scores — approximated as simpler thresholds
    "rsi_score": ("(ta.rsi(close, 14) - 50.0) / 50.0", []),
    "macd_score": ("pMHist > 0 ? math.min(pMHist / pAtr14, 1.0) : math.max(pMHist / pAtr14, -1.0)", [
        "[pMLine, pMSig, pMHist] = ta.macd(close, 12, 26, 9)",
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "stochastic_score": ("(ta.stoch(close, high, low, 14) - 50.0) / 50.0", []),
    "cci_score": ("math.max(-1.0, math.min(1.0, ta.cci(close, 20) / 200.0))", []),
    "williams_r_score": ("(ta.wpr(14) + 50.0) / 50.0", []),
    "sma_cross_score": ("ta.sma(close, 20) > ta.sma(close, 50) ? 1.0 : -1.0", []),
    "adx_score": ("pAdxVal > 25 ? (pDiPlus > pDiMinus ? 1.0 : -1.0) : 0.0", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "ema_trend_score": ("ta.ema(close, 12) > ta.ema(close, 26) ? 1.0 : -1.0", []),
    "ichimoku_score": ("pAboveCloud == 1.0 ? 1.0 : -1.0", [
        "float pTenkan = (ta.highest(high, 9) + ta.lowest(low, 9)) / 2",
        "float pKijun = (ta.highest(high, 26) + ta.lowest(low, 26)) / 2",
        "float pSenkouA = ((pTenkan + pKijun) / 2)",
        "float pSenkouB = (ta.highest(high, 52) + ta.lowest(low, 52)) / 2",
        "float pCloudTop = math.max(pSenkouA, pSenkouB)",
        "float pAboveCloud = close > pCloudTop ? 1.0 : 0.0",
    ]),
    "sar_score": ("(close > ta.sar(0.02, 0.02, 0.2)) ? 1.0 : -1.0", []),
    "vwap_score": ("close > ta.vwap ? 1.0 : -1.0", []),
    "sma_position_score": ("(close - ta.sma(close, 200)) / pAtr14", [
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "bb_touch_score": ("pBbPctB > 0.95 ? -1.0 : (pBbPctB < 0.05 ? 1.0 : 0.0)", [
        "[pBbMid, pBbUp, pBbLo] = ta.bb(close, 20, 2.0)",
        "float pBbPctB = (pBbUp - pBbLo) != 0 ? (close - pBbLo) / (pBbUp - pBbLo) : 0.5",
    ]),
    "bb_squeeze_score": ("(pBbUp - pBbLo) / close < 0.03 ? 1.0 : 0.0", [
        "[pBbMid, pBbUp, pBbLo] = ta.bb(close, 20, 2.0)",
    ]),
    "obv_score": ("(ta.obv > ta.obv[10]) ? 1.0 : -1.0", []),
    "mfi_score": ("(ta.mfi(hlc3, 14) - 50.0) / 50.0", []),
    "volume_surge_score": ("volume > ta.sma(volume, 20) * 1.5 ? 1.0 : 0.0", []),
    # Group C: Regime features (5)
    "regime_confidence": ("0.5", []),  # Placeholder — regime not available in Pine
    "regime_adx_norm": ("pAdxVal / 100.0", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "regime_bb_width_pctl": ("0.5", []),  # Percentile not easily computed in Pine
    "regime_trend_slope": ("ta.linreg(close, 20, 0) > close[20] ? 1.0 : -1.0", []),
    "regime_sma200_dist_norm": ("(close - ta.sma(close, 200)) / (ta.sma(close, 200)) * 100.0 / 100.0", []),
    # Group D: Enhanced features (15)
    "return_1d": ("(close - close[1]) / nz(close[1])", []),
    "return_3d": ("(close - close[3]) / nz(close[3])", []),
    "return_5d": ("(close - close[5]) / nz(close[5])", []),
    "return_10d": ("(close - close[10]) / nz(close[10])", []),
    "realized_vol_5d": ("pRealVol5d", [
        "float pR1 = (close - close[1]) / nz(close[1])",
        "float pR2 = (close[1] - close[2]) / nz(close[2])",
        "float pR3 = (close[2] - close[3]) / nz(close[3])",
        "float pR4 = (close[3] - close[4]) / nz(close[4])",
        "float pR5 = (close[4] - close[5]) / nz(close[5])",
        "float pMeanR5 = (pR1 + pR2 + pR3 + pR4 + pR5) / 5.0",
        "float pRealVol5d = math.sqrt((math.pow(pR1 - pMeanR5, 2) + math.pow(pR2 - pMeanR5, 2) + math.pow(pR3 - pMeanR5, 2) + math.pow(pR4 - pMeanR5, 2) + math.pow(pR5 - pMeanR5, 2)) / 5.0)",
    ]),
    "vol_ratio_5d_20d": ("pRealVol5d / nz(ta.stdev(ta.roc(close, 1) / 100.0, 20))", [
        "float pR1 = (close - close[1]) / nz(close[1])",
        "float pR2 = (close[1] - close[2]) / nz(close[2])",
        "float pR3 = (close[2] - close[3]) / nz(close[3])",
        "float pR4 = (close[3] - close[4]) / nz(close[4])",
        "float pR5 = (close[4] - close[5]) / nz(close[5])",
        "float pMeanR5 = (pR1 + pR2 + pR3 + pR4 + pR5) / 5.0",
        "float pRealVol5d = math.sqrt((math.pow(pR1 - pMeanR5, 2) + math.pow(pR2 - pMeanR5, 2) + math.pow(pR3 - pMeanR5, 2) + math.pow(pR4 - pMeanR5, 2) + math.pow(pR5 - pMeanR5, 2)) / 5.0)",
    ]),
    "intraday_range_pct": ("(high - low) / close", []),
    "rsi_roc_3": ("(ta.rsi(close, 14) - ta.rsi(close, 14)[3]) / 100.0", []),
    "macd_hist_slope_3": ("(pMHist - pMHist[3]) / pAtr14", [
        "[pMLine, pMSig, pMHist] = ta.macd(close, 12, 26, 9)",
        "float pAtr14 = nz(ta.atr(14), 1.0)",
    ]),
    "adx_slope_3": ("(pAdxVal - pAdxVal[3]) / 100.0", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "rsi_x_adx": ("(ta.rsi(close, 14) / 100.0) * (pAdxVal / 100.0)", [
        "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)",
    ]),
    "bb_width_x_vol_ratio": ("((pBbUp - pBbLo) / close) * (volume / ta.sma(volume, 20))", [
        "[pBbMid, pBbUp, pBbLo] = ta.bb(close, 20, 2.0)",
    ]),
    "regime_conf_x_slope": ("0.0", []),  # Requires regime classification
    "day_of_week_norm": ("dayofweek / 5.0", []),
    "month_of_year_norm": ("(month - 1) / 11.0", []),
    # Group E: Alternative data features (17)
    # These data sources are not available in Pine Script, so all map to
    # constant zero.  The PineScript surrogate tree should be trained on
    # only the first 63 (non-alt) features to avoid referencing these.
    "ftd_count_norm": ("0.0", []),
    "ftd_count_change_pct": ("0.0", []),
    "ftd_value_norm": ("0.0", []),
    "ftd_rolling_avg_20d": ("0.0", []),
    "short_interest_pct": ("0.0", []),
    "short_interest_change": ("0.0", []),
    "si_days_to_cover": ("0.0", []),
    "si_days_to_cover_change": ("0.0", []),
    "dark_pool_volume_pct": ("0.0", []),
    "dark_pool_volume_change": ("0.0", []),
    "dark_pool_vs_avg": ("0.0", []),
    "institutional_ownership_pct": ("0.0", []),
    "ownership_concentration": ("0.0", []),
    "institution_count_norm": ("0.0", []),
    "borrow_rate_proxy": ("0.0", []),
    "borrow_rate_change": ("0.0", []),
    "ftd_x_si": ("0.0", []),
    # Group F: HMM regime features (5)
    # HMM regime detection is not available in Pine Script, so all map
    # to constant placeholders.  The surrogate tree should be trained
    # on Pine-compatible features only.
    "hmm_regime_id": ("0.5", []),
    "hmm_prob_bull": ("0.33", []),
    "hmm_prob_bear": ("0.33", []),
    "hmm_prob_sideways": ("0.34", []),
    "hmm_regime_duration_norm": ("0.0", []),
}


# ------------------------------------------------------------------
# Decision tree → Pine Script converter
# ------------------------------------------------------------------


class DecisionTreeExporter:
    """Train a surrogate decision tree and export it to Pine Script.

    The surrogate tree is a simplified :class:`DecisionTreeClassifier`
    trained on the top-N most important features from the full ensemble.
    Its decision path is translated into nested Pine Script ternary
    expressions.

    Parameters
    ----------
    max_depth:
        Maximum tree depth (default 6).
    min_samples_leaf:
        Minimum samples per leaf (default 20).
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._tree: Any = None
        self._feature_names: tuple[str, ...] = ()
        self._top_indices: list[int] = []

    @property
    def is_trained(self) -> bool:
        """Whether a surrogate tree has been trained."""
        return self._tree is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of features used by the surrogate."""
        return self._feature_names

    def train_surrogate(
        self,
        dataset: MLDataset,
        feature_importances: dict[str, float],
        top_n: int = 15,
    ) -> None:
        """Train a surrogate DecisionTree on the top *top_n* features.

        Parameters
        ----------
        dataset:
            The full ML dataset (same one used for training the ensemble).
        feature_importances:
            Feature importance dict from the main model.
        top_n:
            Number of top features to use in the surrogate.
        """
        from sklearn.tree import DecisionTreeClassifier

        # Select top N features by importance
        sorted_feats = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True,
        )
        top_names = [name for name, _ in sorted_feats[:top_n]]

        # Map names to column indices
        name_to_idx = {
            name: i for i, name in enumerate(dataset.feature_names)
        }
        self._top_indices = [name_to_idx[n] for n in top_names if n in name_to_idx]
        self._feature_names = tuple(
            dataset.feature_names[i] for i in self._top_indices
        )

        X_sub = dataset.X[:, self._top_indices]

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree.fit(X_sub, dataset.y)

        self._tree = tree
        logger.info(
            "Surrogate tree: depth=%d, leaves=%d, features=%d",
            tree.get_depth(), tree.get_n_leaves(), len(self._feature_names),
        )

    def tree_to_pine(self) -> str:
        """Convert the trained tree to a Pine Script ternary expression.

        Returns a Pine Script expression that computes the probability
        of class 1 (bullish).

        Raises :class:`RuntimeError` if no tree has been trained.
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")

        tree = self._tree.tree_
        return self._node_to_pine(tree, 0)

    def tree_to_pine_lines(self) -> list[str]:
        """Convert tree to multiple Pine Script lines using intermediate vars.

        For deep trees (depth > 3) this avoids excessively long ternary
        expressions that can exceed TradingView's line-length limits.

        Returns a list of Pine Script lines ending with the final variable
        ``tn0`` holding the probability.
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")

        tree = self._tree.tree_
        lines: list[str] = []
        self._node_to_pine_lines(tree, 0, lines)
        return lines

    @staticmethod
    def _safe_pine_expr(pine_expr: str) -> str:
        """Wrap Pine expression in parentheses if it contains a ternary.

        Prevents operator precedence issues when used in comparisons like
        ``expr <= threshold ? ...`` where ``expr`` is itself a ternary.
        """
        if "?" in pine_expr:
            return f"({pine_expr})"
        return pine_expr

    def _node_to_pine(self, tree: Any, node_id: int) -> str:
        """Recursively convert a tree node to Pine Script ternary."""
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # Leaf node — return probability of class 1
        if left == right:  # leaf
            values = tree.value[node_id][0]
            total = sum(values)
            prob = values[1] / total if total > 0 and len(values) > 1 else 0.0
            return f"{prob:.4f}"

        # Internal node — get feature and threshold
        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = self._feature_names[feat_idx]

        # Get Pine Script expression for this feature
        pine_expr, _ = FEATURE_PINE_MAP.get(feat_name, (feat_name, []))
        safe_expr = self._safe_pine_expr(pine_expr)

        left_expr = self._node_to_pine(tree, left)
        right_expr = self._node_to_pine(tree, right)

        return f"({safe_expr} <= {threshold:.6f} ? {left_expr} : {right_expr})"

    def _node_to_pine_lines(
        self, tree: Any, node_id: int, lines: list[str],
    ) -> str:
        """Recursively convert tree nodes to named intermediate Pine vars.

        Each non-leaf node produces a line ``float _nX = ...`` so deeply
        nested trees stay under TradingView's per-line character limit.
        Returns the variable name for *node_id*.
        """
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # Leaf node — return literal
        if left == right:
            values = tree.value[node_id][0]
            total = sum(values)
            prob = values[1] / total if total > 0 and len(values) > 1 else 0.0
            return f"{prob:.4f}"

        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = self._feature_names[feat_idx]
        pine_expr, _ = FEATURE_PINE_MAP.get(feat_name, (feat_name, []))
        safe_expr = self._safe_pine_expr(pine_expr)

        left_var = self._node_to_pine_lines(tree, left, lines)
        right_var = self._node_to_pine_lines(tree, right, lines)

        var_name = f"tn{node_id}"
        lines.append(
            f"float {var_name} = {safe_expr} <= {threshold:.6f} "
            f"? {left_var} : {right_var}"
        )
        return var_name

    def get_pine_dependencies(self) -> list[str]:
        """Collect all Pine Script variable declarations needed by the tree.

        Returns a de-duplicated, ordered list of Pine Script lines.
        """
        if self._tree is None:
            return []

        seen: set[str] = set()
        deps: list[str] = []

        for name in self._feature_names:
            _, dep_lines = FEATURE_PINE_MAP.get(name, (name, []))
            for line in dep_lines:
                if line not in seen:
                    seen.add(line)
                    deps.append(line)

        return deps


# ------------------------------------------------------------------
# Strategy factory
# ------------------------------------------------------------------


def ml_signal_strategy(
    symbol: str,
    exporter: DecisionTreeExporter,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> StrategyDefinition:
    """Build a :class:`StrategyDefinition` for an ML probability indicator.

    Parameters
    ----------
    symbol:
        Ticker symbol for labeling.
    exporter:
        A trained :class:`DecisionTreeExporter`.
    buy_threshold:
        Probability above which a buy signal fires.
    sell_threshold:
        Probability below which a sell signal fires.

    Returns
    -------
    StrategyDefinition
        A strategy definition ready for PineScriptGenerator.
    """
    if not exporter.is_trained:
        raise RuntimeError("DecisionTreeExporter must be trained first")

    # Inputs
    inputs = [
        Input.float_("buyThresh", buy_threshold, "Buy Threshold",
                      max_val=1.0, step=0.05, group="ML Signal"),
        Input.float_("sellThresh", sell_threshold, "Sell Threshold",
                      max_val=1.0, step=0.05, group="ML Signal"),
    ]

    # Get all Pine dependencies from the tree
    dep_lines = exporter.get_pine_dependencies()

    # No indicators needed — dependencies are raw Pine declarations
    indicators: list[Indicator] = []

    # Build the tree as intermediate variable lines to avoid
    # deeply nested ternary expressions that exceed TradingView
    # line-length limits.
    tree_lines = exporter.tree_to_pine_lines()

    # Extra code: dependency declarations + tree node vars + mlProb
    # Dependencies are already complete Pine Script declarations
    # (e.g. "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)")
    # so they go into extra_code, not indicators.
    extra_code = [
        *dep_lines,
        *tree_lines,
        "float mlProb = tn0",
    ]

    # Plots — overlay chart cannot usefully display mlProb (0-1 range)
    # so we only plot it as display=display.data_window for tooltips.
    # The bgcolor already provides visual feedback for bullish/bearish zones.
    extra_plots = [
        "plot(mlProb, title=\"ML Probability\", display=display.data_window)",
    ]

    # Conditions
    long_entry = Condition(
        expr="mlProb > buyThresh",
        description="ML probability above buy threshold",
    )
    short_entry = Condition(
        expr="mlProb < sellThresh",
        description="ML probability below sell threshold",
    )

    features_desc = ", ".join(exporter.feature_names[:10])
    if len(exporter.feature_names) > 10:
        features_desc += f" (+{len(exporter.feature_names) - 10} more)"

    return StrategyDefinition(
        name=f"{symbol} ML Signal Indicator",
        short_name=f"{symbol.lower()}_ml_signal",
        indicators=indicators,
        inputs=inputs,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=None,
        short_exit=None,
        exit_on_reverse=True,
        extra_code=extra_code,
        extra_plots=extra_plots,
        strategy_mode=False,  # indicator, not strategy
    )
