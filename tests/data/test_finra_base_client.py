"""Unit tests for FinraBaseClient — shared OAuth2 auth, constructor, date normalization."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.finra_base_client import (
    FinraBaseClient,
    normalize_finra_date,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(
    tmp_path: Path,
    client_id: str = "test_id",
    client_secret: str = "test_secret",
) -> FinraBaseClient:
    return FinraBaseClient(
        client_id=client_id,
        client_secret=client_secret,
        data_dir=str(tmp_path),
    )


def _mock_response(
    json_data: object = None,
    status_code: int = 200,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text or (json.dumps(json_data) if json_data else "")
    return resp


# ------------------------------------------------------------------
# Tests: normalize_finra_date
# ------------------------------------------------------------------


class TestNormalizeFinraDate:
    """Tests for the module-level normalize_finra_date helper."""

    def test_iso_format_passthrough(self) -> None:
        assert normalize_finra_date("2024-01-15") == "2024-01-15"

    def test_slash_format(self) -> None:
        assert normalize_finra_date("01/15/2024") == "2024-01-15"

    def test_compact_format(self) -> None:
        assert normalize_finra_date("20240115") == "2024-01-15"

    def test_empty_string(self) -> None:
        assert normalize_finra_date("") == ""

    def test_invalid_returns_empty(self) -> None:
        assert normalize_finra_date("garbage") == ""

    def test_slash_format_single_digit(self) -> None:
        assert normalize_finra_date("1/5/2024") == "2024-01-05"

    def test_invalid_slash_format(self) -> None:
        assert normalize_finra_date("abc/def/ghi") == ""

    def test_none_like_empty(self) -> None:
        # Passing an empty string should be safe
        assert normalize_finra_date("") == ""


# ------------------------------------------------------------------
# Tests: Constructor defaults
# ------------------------------------------------------------------


class TestConstructorDefaults:
    """Tests for FinraBaseClient constructor defaults and configuration."""

    def test_default_data_dir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._data_dir == Path(str(tmp_path))

    def test_default_rate_limit_delay(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._rate_limit_delay == 0.5

    def test_default_headers(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        headers = dict(client._session.headers)
        assert "User-Agent" in headers
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    def test_credentials_stored(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="myid", client_secret="mysec")
        assert client._client_id == "myid"
        assert client._client_secret == "mysec"

    def test_access_token_initially_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._access_token is None

    def test_constructor_env_vars(self, tmp_path: Path) -> None:
        """Verify env var fallback for credentials when not provided."""
        with patch.dict(
            "os.environ",
            {"FINRA_CLIENT_ID": "env_id", "FINRA_CLIENT_SECRET": "env_secret"},
        ):
            client = FinraBaseClient(data_dir=str(tmp_path))
        assert client._client_id == "env_id"
        assert client._client_secret == "env_secret"

    def test_constructor_explicit_overrides_env(self, tmp_path: Path) -> None:
        """Explicit credentials take precedence over env vars."""
        with patch.dict(
            "os.environ",
            {"FINRA_CLIENT_ID": "env_id", "FINRA_CLIENT_SECRET": "env_secret"},
        ):
            client = FinraBaseClient(
                client_id="explicit_id",
                client_secret="explicit_secret",
                data_dir=str(tmp_path),
            )
        assert client._client_id == "explicit_id"
        assert client._client_secret == "explicit_secret"


# ------------------------------------------------------------------
# Tests: OAuth2 Authentication
# ------------------------------------------------------------------


class TestAuthentication:
    """Tests for FINRA OAuth2 authentication in the base client."""

    def test_authenticate_success(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_base_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"access_token": "tok123"})
            result = client._authenticate()

        assert result is True
        assert client._access_token == "tok123"
        assert client._session.headers["Authorization"] == "Bearer tok123"

    def test_authenticate_sends_correct_basic_header(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="cid", client_secret="csec")

        with patch("stockdownloader.data.finra_base_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"access_token": "tok"})
            client._authenticate()

        call_kwargs = mock_post.call_args
        auth_header = call_kwargs.kwargs.get("headers", {}).get("Authorization", "")
        expected = base64.b64encode(b"cid:csec").decode()
        assert auth_header == f"Basic {expected}"

    def test_authenticate_no_credentials(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="", client_secret="")
        result = client._authenticate()
        assert result is False
        assert client._access_token is None

    def test_authenticate_failure(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_base_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                {"error": "forbidden"}, status_code=403, text="Forbidden",
            )
            result = client._authenticate()

        assert result is False
        assert client._access_token is None

    def test_authenticate_network_error(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_base_client.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("network down")
            result = client._authenticate()

        assert result is False
        assert client._access_token is None

    def test_authenticate_missing_token_in_response(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_base_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"status": "ok"})
            result = client._authenticate()

        assert result is False
        assert client._access_token is None
