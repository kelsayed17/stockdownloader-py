"""Tests for spy-ml-mega CLI pipeline."""

from __future__ import annotations

import pytest

from stockdownloader.app.spy_ml_mega import _build_parser, _FULL_GRID, _QUICK_GRID


class TestMegaGrid:
    """Tests for model grid definitions."""

    def test_full_grid_has_18_configs(self) -> None:
        assert len(_FULL_GRID) == 18

    def test_quick_grid_has_6_configs(self) -> None:
        assert len(_QUICK_GRID) == 6

    def test_full_grid_tuples(self) -> None:
        for item in _FULL_GRID:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], bool)

    def test_full_grid_model_types(self) -> None:
        types = {mt for mt, _ in _FULL_GRID}
        expected = {
            "gradient_boosting", "random_forest", "extra_trees",
            "hist_gradient_boosting", "xgboost", "lightgbm",
            "catboost", "logistic_regression", "svm", "mlp", "knn",
        }
        assert types == expected

    def test_quick_grid_model_types(self) -> None:
        types = {mt for mt, _ in _QUICK_GRID}
        expected = {
            "xgboost", "lightgbm", "catboost",
            "random_forest", "logistic_regression", "svm",
        }
        assert types == expected

    def test_full_grid_balanced_variants(self) -> None:
        """Boosting models should have both balanced and unbalanced."""
        for mt in ["gradient_boosting", "random_forest", "extra_trees",
                    "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]:
            variants = [(m, b) for m, b in _FULL_GRID if m == mt]
            assert len(variants) == 2, f"{mt} should have 2 variants"
            balanced_flags = {b for _, b in variants}
            assert balanced_flags == {True, False}, f"{mt} missing balanced variant"

    def test_quick_grid_all_unbalanced(self) -> None:
        """Quick grid should only have unbalanced configs."""
        for _, use_balance in _QUICK_GRID:
            assert use_balance is False


class TestMegaParser:
    """Tests for argument parser."""

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.top_models == 5
        assert args.ensemble_method == "both"
        assert args.diversity_weight == 0.4
        assert args.depth == 10
        assert args.buy_thresh == 0.55
        assert args.sell_thresh == 0.45
        assert args.initial_capital == 100_000.0
        assert args.min_r2 == 0.70
        assert args.no_tournament is False

    def test_ensemble_method_choices(self) -> None:
        parser = _build_parser()
        for method in ["soft_vote", "stacking", "both"]:
            args = parser.parse_args(["--ensemble-method", method])
            assert args.ensemble_method == method

    def test_quick_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--quick"])
        assert args.quick is True

    def test_custom_diversity_weight(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--diversity-weight", "0.7"])
        assert args.diversity_weight == 0.7

    def test_custom_top_models(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--top-models", "7"])
        assert args.top_models == 7

    def test_output_dir(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/test_mega"])
        assert args.output_dir == "/tmp/test_mega"

    def test_no_tournament_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--no-tournament"])
        assert args.no_tournament is True

    def test_no_pine_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--no-pine"])
        assert args.no_pine is True

    def test_custom_thresholds(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--buy-thresh", "0.60", "--sell-thresh", "0.40"])
        assert args.buy_thresh == 0.60
        assert args.sell_thresh == 0.40

    def test_walk_forward_windows_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.walk_forward_windows == 5

    def test_walk_forward_windows_custom(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--walk-forward-windows", "3"])
        assert args.walk_forward_windows == 3

    def test_no_walk_forward_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.no_walk_forward is False

    def test_no_walk_forward_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--no-walk-forward"])
        assert args.no_walk_forward is True

    def test_direct_ensemble_default_true(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.direct_ensemble is True

    def test_use_surrogate_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--use-surrogate"])
        assert args.use_surrogate is True

    def test_long_only_default_true(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.long_only is True

    def test_allow_shorts_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--allow-shorts"])
        assert args.allow_shorts is True

    def test_crash_avoidance_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.crash_avoidance is False

    def test_crash_avoidance_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--crash-avoidance"])
        assert args.crash_avoidance is True

    def test_crash_exit_thresh_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.crash_exit_thresh == 0.35

    def test_crash_exit_thresh_custom(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--crash-exit-thresh", "0.25"])
        assert args.crash_exit_thresh == 0.25

    def test_re_entry_thresh_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.re_entry_thresh == 0.50

    def test_re_entry_thresh_custom(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--re-entry-thresh", "0.55"])
        assert args.re_entry_thresh == 0.55
