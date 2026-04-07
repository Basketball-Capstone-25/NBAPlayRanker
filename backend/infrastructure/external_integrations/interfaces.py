"""External integrations interfaces."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

class IDecodeSupabaseJWT(Protocol):
    """Decode a Supabase JWT and return the claims dict, or None."""

    def __call__(self, token: str) -> Optional[Dict[str, Any]]: ...

class IGetJWTSecret(Protocol):
    """Return the configured JWT secret, or None when not set (dev-mode)."""

    def __call__(self) -> Optional[str]: ...
