"""API-level access-control tests — verify backend blocks unauthorized roles
even when requests bypass the frontend middleware entirely."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from application.api_coordination.app import app

client = TestClient(app)


JWT_SECRET_PATCH = patch(
    "application.access_control_services.access_control_service.get_jwt_secret",
    return_value="test-secret",
)

def _patch_role(role: str):
    """Mock the JWT decode so the request is treated as *role*."""
    return patch(
        "application.access_control_services.access_control_service.decode_supabase_jwt",
        return_value={
            "sub": "user-123",
            "exp": 9999999999,
            "user_metadata": {"role": role},
            "app_metadata": {},
        },
    )


class TestCoachCannotBypassToAnalystEndpoints:
    """A coach token sent directly to analyst-only endpoints must get 403."""

    @JWT_SECRET_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_data_explorer(self, _decode, _secret):
        """Coach calls /data/team-playtypes directly → 403."""
        resp = client.get(
            "/data/team-playtypes",
            params={"season": "2019-20"},
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @JWT_SECRET_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_data_csv_export(self, _decode, _secret):
        """Coach calls /data/team-playtypes.csv directly → 403."""
        resp = client.get(
            "/data/team-playtypes.csv",
            params={"season": "2019-20"},
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @JWT_SECRET_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_model_metrics(self, _decode, _secret):
        """Coach calls /metrics/baseline-vs-ml directly → 403."""
        resp = client.get(
            "/metrics/baseline-vs-ml",
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @JWT_SECRET_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_ml_analysis(self, _decode, _secret):
        """Coach calls /analysis/ml directly → 403."""
        resp = client.get(
            "/analysis/ml",
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @JWT_SECRET_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_shot_metrics(self, _decode, _secret):
        """Coach calls /metrics/shot-models directly → 403."""
        resp = client.get(
            "/metrics/shot-models",
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403


class TestAnalystCannotBypassToCoachEndpoints:
    """An analyst token sent directly to coach-only endpoints must get 403."""

    @JWT_SECRET_PATCH
    @_patch_role("analyst")
    def test_analyst_blocked_from_baseline_ranking(self, _decode, _secret):
        """Analyst calls /rank-plays/baseline directly → 403."""
        resp = client.get(
            "/rank-plays/baseline",
            params={
                "season": "2019-20",
                "our": "TOR",
                "opp": "BOS",
            },
            headers={"Authorization": "Bearer fake.analyst.token"},
        )
        assert resp.status_code == 403

    @JWT_SECRET_PATCH
    @_patch_role("analyst")
    def test_analyst_blocked_from_context_ml(self, _decode, _secret):
        """Analyst calls /rank-plays/context-ml directly → 403."""
        resp = client.get(
            "/rank-plays/context-ml",
            params={
                "season": "2019-20",
                "our": "TOR",
                "opp": "BOS",
            },
            headers={"Authorization": "Bearer fake.analyst.token"},
        )
        assert resp.status_code == 403

    @JWT_SECRET_PATCH
    @_patch_role("analyst")
    def test_analyst_blocked_from_viz(self, _decode, _secret):
        """Analyst calls /viz/playtype-zones directly → 403."""
        resp = client.get(
            "/viz/playtype-zones",
            params={"season": "2019-20", "our": "TOR", "opp": "BOS"},
            headers={"Authorization": "Bearer fake.analyst.token"},
        )
        assert resp.status_code == 403


class TestUnauthenticatedRequestsBlocked:
    """Requests with no token must get 401 regardless of endpoint."""

    @JWT_SECRET_PATCH
    def test_no_token_data_explorer(self, _secret):
        """No Authorization header → 401."""
        resp = client.get(
            "/data/team-playtypes",
            params={"season": "2019-20"},
        )
        assert resp.status_code == 401

    @JWT_SECRET_PATCH
    def test_no_token_baseline(self, _secret):
        """No Authorization header → 401."""
        resp = client.get(
            "/rank-plays/baseline",
            params={"season": "2019-20", "our": "TOR", "opp": "BOS"},
        )
        assert resp.status_code == 401


class TestAuthorizedRolesSucceed:
    """Confirm the access-control service allows the correct role through."""

    @JWT_SECRET_PATCH
    @_patch_role("analyst")
    def test_analyst_can_access_data_explorer(self, _decode, _secret):
        """Analyst calls /data/team-playtypes → 200."""
        resp = client.get(
            "/data/team-playtypes",
            params={"season": "2019-20"},
            headers={"Authorization": "Bearer fake.analyst.token"},
        )
        assert resp.status_code == 200

    @JWT_SECRET_PATCH
    @_patch_role("coach")
    def test_coach_can_access_baseline(self, _decode, _secret):
        """Coach calls /rank-plays/baseline → 200."""
        resp = client.get(
            "/rank-plays/baseline",
            params={
                "season": "2019-20",
                "our": "TOR",
                "opp": "BOS",
            },
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 200
