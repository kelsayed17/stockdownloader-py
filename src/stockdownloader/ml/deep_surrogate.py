"""Deep surrogate decision tree that approximates ensemble probabilities.

Unlike :class:`~stockdownloader.pinescript.ml_export.DecisionTreeExporter`
which trains a classifier on binary labels, this module trains a
:class:`~sklearn.tree.DecisionTreeRegressor` on the ensemble's continuous
probability predictions.  The resulting regression tree can approximate
the full ensemble's probability surface with higher fidelity, especially
when deeper trees are allowed.

Usage::

    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter

    exporter = DeepSurrogateExporter(max_depth=10)
    exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
    pine_lines = exporter.tree_to_pine_lines()
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, TYPE_CHECKING

from stockdownloader.pinescript.ml_export import FEATURE_PINE_MAP

if TYPE_CHECKING:
    from stockdownloader.ml.dataset_builder import MLDataset

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DeepSurrogateExporter:
    """Train a regression surrogate tree and export it to Pine Script.

    The surrogate tree is a :class:`DecisionTreeRegressor` trained on
    the top-N most important features, targeting the ensemble's continuous
    probability predictions (values in [0, 1]).

    Parameters
    ----------
    max_depth:
        Maximum tree depth (default 10).
    min_samples_leaf:
        Minimum samples per leaf (default 10).
    top_n:
        Number of top features to select by importance (default 25).
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        top_n: int = 25,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.top_n = top_n
        self._tree: Any = None
        self._feature_names: tuple[str, ...] = ()
        self._top_indices: list[int] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether a surrogate tree has been trained."""
        return self._tree is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of features used by the surrogate."""
        return self._feature_names

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_surrogate(
        self,
        dataset: MLDataset,
        feature_importances: dict[str, float],
        *,
        ensemble_probs: Any,
    ) -> None:
        """Train a surrogate DecisionTreeRegressor on ensemble probabilities.

        Parameters
        ----------
        dataset:
            The full ML dataset (same one used for training the ensemble).
        feature_importances:
            Feature importance dict from the main model.
        ensemble_probs:
            Continuous probability predictions from the ensemble, shape
            ``(n_samples,)`` with values in [0, 1].
        """
        from sklearn.tree import DecisionTreeRegressor

        # Select top N features by importance
        sorted_feats = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True,
        )
        top_names = [name for name, _ in sorted_feats[: self.top_n]]

        # Map names to column indices
        name_to_idx = {
            name: i for i, name in enumerate(dataset.feature_names)
        }
        self._top_indices = [
            name_to_idx[n] for n in top_names if n in name_to_idx
        ]
        self._feature_names = tuple(
            dataset.feature_names[i] for i in self._top_indices
        )

        X_sub = dataset.X[:, self._top_indices]

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree.fit(X_sub, ensemble_probs)

        self._tree = tree
        logger.info(
            "Deep surrogate tree: depth=%d, leaves=%d, features=%d",
            tree.get_depth(),
            tree.get_n_leaves(),
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Prediction & evaluation
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> Any:
        """Predict using the surrogate on the top_n feature subset.

        Parameters
        ----------
        X:
            Full feature matrix with *all* feature columns (same layout as
            ``dataset.X``).  The method selects the top_n columns internally.

        Returns
        -------
        numpy.ndarray
            Predictions clipped to [0, 1].
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")

        X_sub = X[:, self._top_indices]
        preds = self._tree.predict(X_sub)
        return np.clip(preds, 0.0, 1.0)

    def fidelity_r_squared(self, X: Any, ensemble_probs: Any) -> float:
        """Measure R-squared between surrogate predictions and ensemble probs.

        Parameters
        ----------
        X:
            Full feature matrix (all columns).
        ensemble_probs:
            The ensemble's continuous probability predictions.

        Returns
        -------
        float
            Coefficient of determination (R-squared).
        """
        preds = self.predict(X)
        ss_res = float(np.sum((ensemble_probs - preds) ** 2))
        ss_tot = float(np.sum((ensemble_probs - np.mean(ensemble_probs)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Pine Script export
    # ------------------------------------------------------------------

    def tree_to_pine_lines(self) -> list[str]:
        """Convert tree to multiple Pine Script lines using intermediate vars.

        Each non-leaf node produces a line ``float tnX = ...`` so deeply
        nested trees stay under TradingView's per-line character limit.

        Returns a list of Pine Script lines ending with the final variable
        ``tn0`` holding the predicted probability.
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")

        tree = self._tree.tree_
        lines: list[str] = []
        self._node_to_pine_lines(tree, 0, lines)
        return lines

    def tree_to_pine(self) -> str:
        """Convert the trained tree to a single Pine Script ternary expression.

        Suitable for small/shallow trees.  For deep trees, prefer
        :meth:`tree_to_pine_lines`.

        Returns a Pine Script expression that computes the predicted
        probability.
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")

        tree = self._tree.tree_
        return self._node_to_pine(tree, 0)

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
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_pine_expr(pine_expr: str) -> str:
        """Wrap Pine expression in parentheses if it contains a ternary."""
        if "?" in pine_expr:
            return f"({pine_expr})"
        return pine_expr

    def _node_to_pine(self, tree: Any, node_id: int) -> str:
        """Recursively convert a tree node to Pine Script ternary."""
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # Leaf node -- return mean prediction clipped to [0, 1]
        if left == right:
            value = float(tree.value[node_id][0][0])
            value = max(0.0, min(1.0, value))
            return f"{value:.4f}"

        # Internal node
        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = self._feature_names[feat_idx]

        pine_expr, _ = FEATURE_PINE_MAP.get(feat_name, (feat_name, []))
        safe_expr = self._safe_pine_expr(pine_expr)

        left_expr = self._node_to_pine(tree, left)
        right_expr = self._node_to_pine(tree, right)

        return f"({safe_expr} <= {threshold:.6f} ? {left_expr} : {right_expr})"

    def _node_to_pine_lines(
        self, tree: Any, node_id: int, lines: list[str],
    ) -> str:
        """Recursively convert tree nodes to named intermediate Pine vars.

        Each non-leaf node produces a line ``float tnX = ...``.
        Returns the variable name for *node_id*.
        """
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # Leaf node -- return literal clipped to [0, 1]
        if left == right:
            value = float(tree.value[node_id][0][0])
            value = max(0.0, min(1.0, value))
            return f"{value:.4f}"

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
