"""FastAPI dependency for JWT authentication and role enforcement."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request

from application.access_control_services.access_control_service import (
    check_user_access,
    get_user_role,
    validate_session,
)

logger = logging.getLogger(__name__)

def _extract_token(request: Request) -> Optional[str]:
    """Pull the Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None

async def require_auth(request: Request) -> dict:
    """FastAPI dependency – ensures the request carries a valid session."""
    token = _extract_token(request)
    if not validate_session(token):
        raise HTTPException(status_code=401, detail="Invalid or missing session token.")

    role = get_user_role(token) if token else None
    return {"token": token, "role": role}

class _RequireRole:
    """Callable dependency factory: ``Depends(require_role("coach"))``."""

    def __init__(self, role: str) -> None:
        self._role = role

    async def __call__(self, user: dict = Depends(require_auth)) -> dict:
        if not check_user_access(user.get("role"), self._role):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user.get('role')}' cannot access '{self._role}' resources.",
            )
        return user

def require_role(role: str) -> _RequireRole:
    """Return a dependency that enforces *role* access."""
    return _RequireRole(role)
