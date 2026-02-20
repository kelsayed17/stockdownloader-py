"""Tests for retry_executor utility functions."""

import logging

from stockdownloader.util.io_helpers import execute, execute_with_result

LOGGER = logging.getLogger(__name__)


def test_successful_execution_on_first_attempt():
    counter = {"value": 0}

    def action():
        counter["value"] += 1

    execute(action, max_retries=3, logger=LOGGER, context="test")

    assert counter["value"] == 1


def test_retries_on_failure():
    counter = {"value": 0}

    def action():
        counter["value"] += 1
        if counter["value"] < 3:
            raise RuntimeError("fail")

    execute(action, max_retries=3, logger=LOGGER, context="test")

    assert counter["value"] == 3


def test_stops_after_max_retries():
    counter = {"value": 0}

    def action():
        counter["value"] += 1
        raise RuntimeError("always fail")

    execute(action, max_retries=2, logger=LOGGER, context="test")

    assert counter["value"] == 3  # initial + 2 retries


def test_supplier_returns_result():
    result = execute_with_result(lambda: "hello", max_retries=3, logger=LOGGER, context="test")
    assert result == "hello"


def test_supplier_retries_and_returns_result():
    counter = {"value": 0}

    def action():
        counter["value"] += 1
        if counter["value"] < 2:
            raise RuntimeError("fail")
        return "success"

    result = execute_with_result(action, max_retries=3, logger=LOGGER, context="test")

    assert result == "success"
    assert counter["value"] == 2


def test_supplier_returns_none_after_all_retries_fail():
    result = execute_with_result(
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
        max_retries=2,
        logger=LOGGER,
        context="test",
    )

    assert result is None


def test_default_retry_count_used():
    counter = {"value": 0}

    def action():
        counter["value"] += 1
        raise RuntimeError("fail")

    execute(action, logger=LOGGER, context="test")

    assert counter["value"] == 4  # initial + 3 default retries
