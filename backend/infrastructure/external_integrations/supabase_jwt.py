"""Supabase JWT verification.

Supports both HS256 (symmetric) and ES256/RS256 (asymmetric) tokens.
The algorithm is read from the token header at runtime so the correct
verification path is chosen automatically.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import jwt  # PyJWT
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

_JWT_SECRET: Optional[str] = None
_JWKS_CLIENT: Optional[PyJWKClient] = None


def get_jwt_secret() -> Optional[str]:
    """Lazily read the secret so tests/CI can run without it."""
    global _JWT_SECRET
    if _JWT_SECRET is None:
        _JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
    return _JWT_SECRET or None


def _get_jwks_client() -> Optional[PyJWKClient]:
    """Lazily create a PyJWKClient for asymmetric (ES256/RS256) verification."""
    global _JWKS_CLIENT
    if _JWKS_CLIENT is None:
        url = os.environ.get("SUPABASE_URL", "")
        if url:
            _JWKS_CLIENT = PyJWKClient(
                f"{url.rstrip('/')}/auth/v1/.well-known/jwks.json"
            )
    return _JWKS_CLIENT


def decode_supabase_jwt(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a Supabase access-token JWT."""
    secret = get_jwt_secret()
    if secret is None:
        logger.warning(
            "SUPABASE_JWT_SECRET not set – skipping JWT verification (dev mode)."
        )
        return None

    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        logger.warning("JWT header decode failed: %s", exc)
        return None

    alg = header.get("alg", "HS256")

    try:
        if alg == "HS256":
            return jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["exp", "sub"]},
            )

        if alg in ("ES256", "RS256"):
            client = _get_jwks_client()
            if client is None:
                logger.warning(
                    "SUPABASE_URL not set – cannot fetch JWKS for %s verification.",
                    alg,
                )
                return None
            signing_key = client.get_signing_key_from_jwt(token)
            result = jwt.decode(
                token,
                signing_key.key,
                algorithms=[alg],
                audience="authenticated",
                options={"require": ["exp", "sub"]},
            )
            return result

        logger.warning("Unsupported JWT algorithm: %s", alg)
        return None

    except jwt.ExpiredSignatureError:
        logger.warning("JWT verification failed: token expired.")
        return None
    except jwt.InvalidTokenError as exc:
        logger.warning("JWT verification failed: %s", exc)
        return None
