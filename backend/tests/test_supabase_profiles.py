"""Role lookup keeps RLS credentials, current assignments, and failure isolation."""
import io
import json
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from infrastructure.external_integrations.supabase_profiles import get_profile_role


@pytest.fixture(autouse=True)
def profile_configuration(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "public-test-key")


def test_lookup_scopes_to_subject_with_user_bearer_and_observes_role_changes():
    responses = [io.BytesIO(json.dumps([{"role": role}]).encode()) for role in ("coach", "analyst")]
    with patch("infrastructure.external_integrations.supabase_profiles.urlopen", side_effect=responses) as http:
        assert get_profile_role("user-token", "user-123") == "coach"
        assert get_profile_role("user-token", "user-123") == "analyst"
    request = http.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer user-token"
    assert request.get_header("Apikey") == "public-test-key"
    assert parse_qs(urlparse(request.full_url).query) == {
        "select": ["role"], "id": ["eq.user-123"], "limit": ["1"]
    }
    assert http.call_args.kwargs["timeout"] == 5


@pytest.mark.parametrize("body", [[], [{"role": None}], [{"role": "admin"}], {"role": "coach"}, [None]])
def test_missing_pending_and_malformed_profiles_deny_access(body):
    with patch("infrastructure.external_integrations.supabase_profiles.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
        assert get_profile_role("user-token", "user-123") is None


def test_unavailable_profile_service_fails_closed():
    with patch("infrastructure.external_integrations.supabase_profiles.urlopen", side_effect=URLError("offline")):
        assert get_profile_role("user-token", "user-123") is None


def test_unconfigured_profile_service_fails_closed(monkeypatch):
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY")
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", raising=False)
    with patch("infrastructure.external_integrations.supabase_profiles.urlopen") as http:
        assert get_profile_role("user-token", "user-123") is None
        http.assert_not_called()
