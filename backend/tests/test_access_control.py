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


@patch(
    "application.access_control_services.access_control_service.decode_supabase_jwt",
    return_value={"user_metadata": {"role": "coach"}, "app_metadata": {}},
)
def test_get_user_role_extracts_coach(mock_decode):
    assert get_user_role("tok") == "coach"


@patch(
    "application.access_control_services.access_control_service.decode_supabase_jwt",
    return_value={"user_metadata": {}, "app_metadata": {"role": "analyst"}},
)
def test_get_user_role_extracts_analyst_from_app_metadata(mock_decode):
    assert get_user_role("tok") == "analyst"


@patch(
    "application.access_control_services.access_control_service.decode_supabase_jwt",
    return_value={"user_metadata": {"role": "admin"}, "app_metadata": {}},
)
def test_get_user_role_rejects_unknown_role(mock_decode):
    assert get_user_role("tok") is None


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
