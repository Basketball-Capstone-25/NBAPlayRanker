"""Tests for access control service (role-based authorization)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from application.access_control_services.access_control_service import (
    check_user_access,
    get_user_role,
    validate_session,
)


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value=None)
def test_validate_session_dev_mode_allows(mock_secret):
    assert validate_session(None) is True


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value="real-secret")
def test_validate_session_rejects_none_token(mock_secret):
    assert validate_session(None) is False


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value="real-secret")
@patch("application.access_control_services.access_control_service.decode_supabase_jwt", return_value=None)
def test_validate_session_rejects_invalid_jwt(mock_decode, mock_secret):
    assert validate_session("bad.token.here") is False


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value="real-secret")
@patch(
    "application.access_control_services.access_control_service.decode_supabase_jwt",
    return_value={"sub": "u1", "exp": 9999999999},
)
def test_validate_session_accepts_valid_jwt(mock_decode, mock_secret):
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


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value=None)
def test_check_user_access_dev_mode_allows(mock_secret):
    assert check_user_access(None, "analytics") is True


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value="secret")
def test_check_user_access_rejects_wrong_role(mock_secret):
    assert check_user_access("coach", "analytics") is False


@patch("application.access_control_services.access_control_service.get_jwt_secret", return_value="secret")
def test_check_user_access_accepts_correct_role(mock_secret):
    assert check_user_access("analyst", "analytics") is True
