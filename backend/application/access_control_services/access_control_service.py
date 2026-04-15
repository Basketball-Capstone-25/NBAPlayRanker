"""Session validation and role-based access control."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from infrastructure.external_integrations import (
    decode_supabase_jwt,
    get_jwt_secret,
)

logger = logging.getLogger(__name__)

# Each key is a resource tag attached by the auth dependency.
# Values are the set of roles that may access it.
_ROLE_PERMISSIONS: Dict[str, set] = {
    "recommendation": {"coach"},
    "viz": {"coach"},
    "shotplan": {"coach"},
    "analytics": {"analyst"},
    "data": {"analyst"},
    "shot_analysis": {"analyst"},
    "export": {"coach", "analyst"},
    "meta": {"coach", "analyst"},
}

def validate_session(session_token: Optional[str]) -> bool:
    """Role-based access control service."""
    if get_jwt_secret() is None:
        logger.debug("access_control: dev-mode (no JWT secret) - allowing")
        return True
    if not session_token:
        return False
    claims = decode_supabase_jwt(session_token)
    return claims is not None

def get_user_role(session_token: str) -> Optional[str]:
    """Extract the user role (coach | analyst) from the JWT claims."""
    claims = decode_supabase_jwt(session_token)
    if claims is None:
        return None

    user_meta = claims.get("user_metadata", {})
    app_meta = claims.get("app_metadata", {})
    role = user_meta.get("role") or app_meta.get("role")

    if role in ("coach", "analyst"):
        return role
    return None

def check_user_access(user_role: Optional[str], resource: str) -> bool:
    """Return True if user_role may access resource."""
    if get_jwt_secret() is None:
        return True
    allowed_roles = _ROLE_PERMISSIONS.get(resource)
    if allowed_roles is None:
        return user_role is not None
    return user_role in allowed_roles
