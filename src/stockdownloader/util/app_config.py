"""Centralized application configuration.

Reads API keys and service credentials from environment variables
once at startup.  Inject this config into components instead of
scattering ``os.environ`` calls throughout data clients and app modules.

Usage::

    config = AppConfig.from_env()
    client = PolygonDataClient(api_key=config.polygon_api_key)

    # Or override specific values:
    config = AppConfig.from_env(polygon_api_key="my-key")
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration read from environment.

    All fields default to empty strings so callers can check
    ``if config.polygon_api_key:`` without worrying about None.

    Preferred construction is via :meth:`from_env`, which reads
    environment variables once and stores the results immutably.
    """

    # -- Polygon.io ----------------------------------------------------------
    polygon_api_key: str = ""

    # -- FINRA (short interest, dark pool, reg-SHO) --------------------------
    finra_client_id: str = ""
    finra_client_secret: str = ""

    # -- Tradier (options chain data) ----------------------------------------
    tradier_api_token: str = ""
    tradier_sandbox_token: str = ""

    @classmethod
    def from_env(cls, **overrides: str) -> AppConfig:
        """Load configuration from environment variables.

        Any keyword argument overrides the corresponding env-var value,
        making it easy to merge CLI flags::

            config = AppConfig.from_env(polygon_api_key=args.polygon_key)

        Parameters
        ----------
        **overrides:
            Field-name / value pairs that take priority over the
            environment.  Empty-string or ``None`` overrides are ignored
            so that ``args.polygon_key or ""`` never clobbers a real
            env-var value.
        """

        def _get(field: str, env_var: str) -> str:
            override = overrides.get(field)
            if override:               # non-empty string wins
                return override
            return os.environ.get(env_var, "")

        return cls(
            polygon_api_key=_get("polygon_api_key", "POLYGON_API_KEY"),
            finra_client_id=_get("finra_client_id", "FINRA_CLIENT_ID"),
            finra_client_secret=_get("finra_client_secret", "FINRA_CLIENT_SECRET"),
            tradier_api_token=_get("tradier_api_token", "TRADIER_API_TOKEN"),
            tradier_sandbox_token=_get("tradier_sandbox_token", "TRADIER_SANDBOX_TOKEN"),
        )
