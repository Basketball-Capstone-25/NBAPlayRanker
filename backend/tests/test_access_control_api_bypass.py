"""API-level access-control tests — verify backend blocks unauthorized roles
even when requests bypass the frontend middleware entirely."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from fastapi.testclient import TestClient

from application.api_coordination.app import app

client = TestClient(app)


DEV_BYPASS_PATCH = patch(
    "application.access_control_services.access_control_service.is_insecure_dev_auth_enabled",
    return_value=False,
)

@pytest.fixture(autouse=True)
def valid_test_session():
    with patch(
        "application.access_control_services.access_control_service.decode_supabase_jwt",
        return_value={"sub": "user-123", "exp": 9999999999},
    ):
        yield


def _patch_role(role: str):
    """A current database role, independent of anything the caller claims."""
    return patch(
        "application.access_control_services.access_control_service.get_profile_role",
        return_value=role,
    )


@pytest.mark.parametrize("path", ["/rank-plays/baseline", "/meta/options"])
def test_self_assigned_metadata_cannot_access_protected_api(path):
    service = "application.access_control_services.access_control_service"
    with patch(
        f"{service}.decode_supabase_jwt",
        return_value={"sub": "user-123", "user_metadata": {"role": "coach"}},
    ), patch(f"{service}.get_profile_role", return_value=None):
        response = client.get(
            path,
            params={"season": "2019-20", "our": "TOR", "opp": "BOS"},
            headers={"Authorization": "Bearer user.with.edited.metadata"},
        )
        assert response.status_code == 403


class TestCoachCannotBypassToAnalystEndpoints:
    """A coach token sent directly to analyst-only endpoints must get 403."""

    @DEV_BYPASS_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_data_explorer(self, _decode, _secret):
        """Coach calls /data/team-playtypes directly → 403."""
        resp = client.get(
            "/data/team-playtypes",
            params={"season": "2019-20"},
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @DEV_BYPASS_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_data_csv_export(self, _decode, _secret):
        """Coach calls /data/team-playtypes.csv directly → 403."""
        resp = client.get(
            "/data/team-playtypes.csv",
            params={"season": "2019-20"},
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @DEV_BYPASS_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_model_metrics(self, _decode, _secret):
        """Coach calls /metrics/baseline-vs-ml directly → 403."""
        resp = client.get(
            "/metrics/baseline-vs-ml",
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @DEV_BYPASS_PATCH
    @_patch_role("coach")
    def test_coach_blocked_from_ml_analysis(self, _decode, _secret):
        """Coach calls /analysis/ml directly → 403."""
        resp = client.get(
            "/analysis/ml",
            headers={"Authorization": "Bearer fake.coach.token"},
        )
        assert resp.status_code == 403

    @DEV_BYPASS_PATCH
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

    @DEV_BYPASS_PATCH
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

    @DEV_BYPASS_PATCH
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

    @DEV_BYPASS_PATCH
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

    @DEV_BYPASS_PATCH
    def test_no_token_data_explorer(self, _secret):
        """No Authorization header → 401."""
        resp = client.get(
            "/data/team-playtypes",
            params={"season": "2019-20"},
        )
        assert resp.status_code == 401

    @DEV_BYPASS_PATCH
    def test_no_token_baseline(self, _secret):
        """No Authorization header → 401."""
        resp = client.get(
            "/rank-plays/baseline",
            params={"season": "2019-20", "our": "TOR", "opp": "BOS"},
        )
        assert resp.status_code == 401


class TestAuthorizedRolesSucceed:
    """Confirm the access-control service allows the correct role through."""

    @DEV_BYPASS_PATCH
    @_patch_role("analyst")
    def test_analyst_can_access_data_explorer(self, _decode, _secret):
        """Analyst calls /data/team-playtypes → 200."""
        resp = client.get(
            "/data/team-playtypes",
            params={"season": "2019-20"},
            headers={"Authorization": "Bearer fake.analyst.token"},
        )
        assert resp.status_code == 200

    @DEV_BYPASS_PATCH
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
