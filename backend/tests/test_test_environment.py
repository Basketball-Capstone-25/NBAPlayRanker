"""Ordinary backend tests must not queue writes to a developer's database."""

import os
from pathlib import Path
import subprocess
import sys


def test_bootstrap_disables_live_request_logging():
    environment = {
        **os.environ,
        "SUPABASE_URL": "https://developer-project.example",
        "SUPABASE_PUBLISHABLE_KEY": "developer-primary-key",
        "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY": "developer-fallback-key",
    }
    script = """
import os
import runpy
import sys
from unittest.mock import patch

runpy.run_path(sys.argv[1])
from infrastructure.external_integrations import supabase_request_log as store

for name in (
    "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"
):
    assert os.environ[name] == "", f"Test bootstrap did not isolate {name}"

with patch.object(store._WRITER, "submit") as submit:
    store.submit_record("test-token", {"endpoint": "/rank-plays/baseline"})
    submit.assert_not_called()
assert store.flush(timeout=0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).with_name("conftest.py"))],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
