"""Shared fixtures for live API integration tests.

All tests in this directory hit real external APIs and require network
access. They are excluded from normal test runs via the ``live`` marker
and the ``addopts = "-m 'not live'"`` setting in pyproject.toml.

Run explicitly with::

    pytest -m live -v
"""
from __future__ import annotations

import pytest

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper

pytestmark = pytest.mark.live

TEST_TICKER = "SPY"


@pytest.fixture(scope="session")
def yahoo_auth() -> YahooAuthHelper:
    """Authenticate once with Yahoo Finance and reuse across all tests."""
    auth = YahooAuthHelper()
    auth.authenticate()
    return auth
