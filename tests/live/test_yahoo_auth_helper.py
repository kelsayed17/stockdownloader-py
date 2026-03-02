"""Live tests for YahooAuthHelper cookie/crumb authentication."""
from __future__ import annotations

import pytest

from stockdownloader.data.market.yahoo_base_client import YahooAuthHelper

pytestmark = pytest.mark.live


class TestYahooAuthHelper:

    def test_authenticate_returns_true(self):
        auth = YahooAuthHelper()
        result = auth.authenticate()
        assert result is True

    def test_crumb_is_non_empty_string_after_auth(self):
        auth = YahooAuthHelper()
        auth.authenticate()
        assert auth.crumb is not None
        assert isinstance(auth.crumb, str)
        assert len(auth.crumb) > 0

    def test_session_has_cookies_after_auth(self):
        auth = YahooAuthHelper()
        auth.authenticate()
        assert len(auth.session.cookies) > 0
