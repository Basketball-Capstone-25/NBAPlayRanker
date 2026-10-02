"""Browser origin boundaries for local development and hosted deployments."""

import pytest
from fastapi.testclient import TestClient

from application.api_coordination.app import _cors_origins, app


def test_configured_frontend_replaces_local_origins(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", " https://nbaplayranker-seven.vercel.app/ ")
    assert _cors_origins() == ["https://nbaplayranker-seven.vercel.app"]


@pytest.mark.parametrize("origin", ["*", "https://*.vercel.app", "https://example.com/path"])
def test_invalid_deployment_origins_fail_closed(monkeypatch, origin):
    monkeypatch.setenv("FRONTEND_ORIGIN", origin)
    with pytest.raises(ValueError, match="explicit HTTP"):
        _cors_origins()


def test_unknown_origin_cannot_make_authenticated_browser_requests():
    client = TestClient(app)
    response = client.options(
        "/meta/options",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_allowed_origin_preflight_supports_authorization():
    client = TestClient(app)
    origin = _cors_origins()[0]
    response = client.options(
        "/meta/options",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"
