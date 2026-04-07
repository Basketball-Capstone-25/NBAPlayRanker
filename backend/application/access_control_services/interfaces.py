"""Access-control service interfaces."""
from __future__ import annotations

from typing import Optional, Protocol

class IValidateSession(Protocol):
    """Protocol for session token validation."""

    def __call__(self, session_token: Optional[str]) -> bool: ...

class IGetUserRole(Protocol):
    """Protocol for extracting user role from a session token."""

    def __call__(self, session_token: str) -> Optional[str]: ...

class ICheckUserAccess(Protocol):
    """Protocol for role-based resource authorization."""

    def __call__(self, user_role: Optional[str], resource: str) -> bool: ...
