"""Tests for access control service (role-based authorization)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from application.access_control_services.access_control_service import (
    check_user_access,
    get_user_role,
    validate_session,
)


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=True)
def test_validate_session_dev_mode_allows(mock_dev_mode):
    assert validate_session(None) is True


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=False)
def test_validate_session_rejects_none_token(mock_dev_mode):
    assert validate_session(None) is False


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=False)
@patch("application.access_control_services.access_control_service.decode_supabase_jwt", return_value=None)
def test_validate_session_rejects_invalid_jwt(mock_decode, mock_dev_mode):
    assert validate_session("bad.token.here") is False


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=False)
@patch(
    "application.access_control_services.access_control_service.decode_supabase_jwt",
    return_value={"sub": "u1", "exp": 9999999999},
)
def test_validate_session_accepts_valid_jwt(mock_decode, mock_dev_mode):
    assert validate_session("valid.token.here") is True


@patch(
    "application.access_control_services.access_control_service.decode_supabase_jwt",
    return_value=None,
)
def test_get_user_role_returns_none_on_invalid_token(mock_decode):
    assert get_user_role("bad") is None


@pytest.mark.parametrize("profile_role", ["coach", "analyst", None])
def test_profile_is_authority_despite_forged_or_stale_metadata(profile_role):
    service = "application.access_control_services.access_control_service"
    claims = {
        "sub": "user-123",
        "user_metadata": {"role": "coach"},
        "app_metadata": {"role": "analyst"},
    }
    with patch(f"{service}.decode_supabase_jwt", return_value=claims), patch(
        f"{service}.get_profile_role", return_value=profile_role
    ) as lookup:
        assert get_user_role("tok") == profile_role
        lookup.assert_called_once_with("tok", "user-123")


def test_missing_subject_never_looks_up_a_role():
    service = "application.access_control_services.access_control_service"
    with patch(f"{service}.decode_supabase_jwt", return_value={}), patch(
        f"{service}.get_profile_role"
    ) as lookup:
        assert get_user_role("tok") is None
        lookup.assert_not_called()


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=True)
def test_check_user_access_dev_mode_allows(mock_dev_mode):
    assert check_user_access(None, "analytics") is True


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=False)
def test_check_user_access_rejects_wrong_role(mock_dev_mode):
    assert check_user_access("coach", "analytics") is False


@patch("application.access_control_services.access_control_service.is_insecure_dev_auth_enabled", return_value=False)
def test_check_user_access_accepts_correct_role(mock_dev_mode):
    assert check_user_access("analyst", "analytics") is True


def test_missing_configuration_does_not_allow_access(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    assert validate_session(None) is False
    assert check_user_access(None, "analytics") is False


def test_explicit_local_development_opt_in(monkeypatch):
    monkeypatch.setenv("ALLOW_INSECURE_DEV_AUTH", "true")
    assert validate_session(None) is True
    assert check_user_access(None, "analytics") is True
