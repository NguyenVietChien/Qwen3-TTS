# coding=utf-8
"""Simple shared-secret API key auth for routes that should not be public."""

import os

from fastapi import Header, HTTPException, status

API_KEY = os.environ.get("API_KEY")


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Require a matching X-API-Key header if API_KEY is configured.

    If the API_KEY environment variable is unset, auth is disabled
    (local/dev usage without a key).
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")
