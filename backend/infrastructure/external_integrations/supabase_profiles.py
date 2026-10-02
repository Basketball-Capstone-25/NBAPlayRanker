"""Read current, administrator-assigned roles with the user's RLS-scoped token."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def get_profile_role(session_token: str, user_id: str) -> Optional[str]:
    """Fail closed if a protected profile cannot establish the user's role.

    Do not cache roles or fall back to JWT metadata: an administrator's role
    change must apply to the next request, including tokens issued earlier.
    The publishable key grants no extra privileges; RLS still uses this user's
    verified access token to restrict the query to their own profile.
    """
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get(
        "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", ""
    )
    if not url or not key or not user_id:
        logger.warning("Profile role lookup is not configured or has no user ID.")
        return None

    query = urlencode({"select": "role", "id": f"eq.{user_id}", "limit": "1"})
    request = Request(
        f"{url}/rest/v1/profiles?{query}",
        headers={
            "Authorization": f"Bearer {session_token}",
            "apikey": key,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            rows = json.load(response)
    except (URLError, OSError, ValueError):
        # Never log bearer tokens, keys, or remote response bodies.
        logger.warning("Unable to retrieve the current profile role.")
        return None

    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    role = rows[0].get("role")
    return role if role in ("coach", "analyst") else None
