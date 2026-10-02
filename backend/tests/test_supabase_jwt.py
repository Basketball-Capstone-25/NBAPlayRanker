"""Cryptographic verification must work with public JWKS and fail closed."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from infrastructure.external_integrations import supabase_jwt


@pytest.fixture
def signing_setup(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://test-project.supabase.co")
    monkeypatch.setattr(supabase_jwt, "_JWT_SECRET", None)
    private_key = ec.generate_private_key(ec.SECP256R1())
    client = Mock()
    client.get_signing_key_from_jwt.return_value = SimpleNamespace(key=private_key.public_key())
    monkeypatch.setattr(supabase_jwt, "_get_jwks_client", lambda: client)
    claims = {
        "sub": "test-user",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        "aud": "authenticated",
        "iss": "https://test-project.supabase.co/auth/v1",
    }
    return private_key, client, claims


def sign(private_key, claims):
    return jwt.encode(claims, private_key, algorithm="ES256", headers={"kid": "test-key"})


def test_es256_verifies_without_legacy_secret(signing_setup):
    private_key, client, claims = signing_setup
    token = sign(private_key, claims)
    decoded = supabase_jwt.decode_supabase_jwt(token)
    assert decoded["sub"] == "test-user"
    client.get_signing_key_from_jwt.assert_called_once_with(token)


@pytest.mark.parametrize("changes", [
    {"iss": "https://other-project.supabase.co/auth/v1"},
    {"aud": "other-audience"},
    {"exp": 1},
])
def test_es256_rejects_invalid_claims(signing_setup, changes):
    private_key, _, claims = signing_setup
    assert supabase_jwt.decode_supabase_jwt(sign(private_key, {**claims, **changes})) is None


def test_es256_rejects_invalid_signature(signing_setup):
    _, _, claims = signing_setup
    wrong_key = ec.generate_private_key(ec.SECP256R1())
    assert supabase_jwt.decode_supabase_jwt(sign(wrong_key, claims)) is None


def test_jwks_failure_returns_unauthenticated(signing_setup):
    private_key, client, claims = signing_setup
    client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError("unavailable")
    assert supabase_jwt.decode_supabase_jwt(sign(private_key, claims)) is None


def test_hs256_is_rejected_without_configured_secret(signing_setup):
    _, _, claims = signing_setup
    token = jwt.encode(claims, "test-secret-that-is-at-least-32-bytes", algorithm="HS256")
    assert supabase_jwt.decode_supabase_jwt(token) is None
