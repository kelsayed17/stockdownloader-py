"""Tests for GME options CLI."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from stockdownloader.gme.options.__main__ import build_parser


class TestCLIParser:
    def test_fetch_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["fetch", "--start", "2022-01-01", "--end", "2023-01-01"])
        assert args.command == "fetch"
        assert args.start == "2022-01-01"
        assert args.end == "2023-01-01"

    def test_build_state_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["build-state"])
        assert args.command == "build-state"

    def test_backtest_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["backtest", "--strategy", "wheel"])
        assert args.command == "backtest"
        assert args.strategy == "wheel"

    def test_backtest_all(self):
        parser = build_parser()
        args = parser.parse_args(["backtest", "--all"])
        assert args.command == "backtest"
        assert args.all is True

    def test_scorecard_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["scorecard", "--date", "2023-06-15"])
        assert args.command == "scorecard"
        assert args.date == "2023-06-15"

    def test_run_all_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["run-all"])
        assert args.command == "run-all"

    def test_snapshot_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["snapshot"])
        assert args.command == "snapshot"

    def test_enrich_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["enrich"])
        assert args.command == "enrich"
