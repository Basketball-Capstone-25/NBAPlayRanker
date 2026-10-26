"""Readiness is separate from process liveness and never exposes config values."""

from urllib.parse import urlsplit


def readiness_report(environ, *, source_loaded: bool, predictions_loaded: bool) -> dict:
    missing = []
    if not source_loaded:
        missing.append("synergy_playtypes_2019_2025_players.csv (missing or unreadable)")
    url = urlsplit(environ.get("SUPABASE_URL", ""))
    if url.scheme not in {"http", "https"} or not url.hostname:
        missing.append("SUPABASE_URL")
    if not (environ.get("SUPABASE_PUBLISHABLE_KEY") or environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")):
        missing.append("SUPABASE_PUBLISHABLE_KEY")
    if environ.get("ALLOW_INSECURE_DEV_AUTH", "").lower() == "true":
        missing.append("secure authentication (development bypass is enabled)")
    return {"status": "ready" if not missing else "not_ready", "ready": not missing,
            "missing": missing, "features": {"playtypes": source_loaded,
            "saved_playtype_predictions": predictions_loaded},
            "scope": "Core play-type data and authentication configuration. Shot data and saved shot results have separate checks."}
