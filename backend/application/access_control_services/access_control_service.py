"""Session validation and role-based access control."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from infrastructure.external_integrations import (
    decode_supabase_jwt,
    is_insecure_dev_auth_enabled,
)
from infrastructure.external_integrations.supabase_profiles import get_profile_role

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
    if is_insecure_dev_auth_enabled():
        logger.debug("access_control: explicit insecure local development mode")
        return True
    if not session_token:
        return False
    claims = decode_supabase_jwt(session_token)
    return claims is not None

def get_user_role(session_token: str) -> Optional[str]:
    """Look up the current protected profile role after verifying the token."""
    claims = decode_supabase_jwt(session_token)
    if claims is None:
        return None

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        return None
    return get_profile_role(session_token, user_id)

def check_user_access(user_role: Optional[str], resource: str) -> bool:
    """Return True if user_role may access resource."""
    if is_insecure_dev_auth_enabled():
        return True
    allowed_roles = _ROLE_PERMISSIONS.get(resource)
    if allowed_roles is None:
        return user_role is not None
    return user_role in allowed_roles
