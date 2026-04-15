import sys
from pathlib import Path

BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

_BASELINE_PKG_DIR: Path = BACKEND_DIR / "domain" / "baseline_recommendation"
if str(_BASELINE_PKG_DIR) not in sys.path:
    sys.path.append(str(_BASELINE_PKG_DIR))

import os

from application.api_coordination.app import app as _app  # noqa: F401

os.environ.pop("SUPABASE_JWT_SECRET", None)

from infrastructure.external_integrations import supabase_jwt as _jwt_mod  # noqa: E402

_jwt_mod._JWT_SECRET = None  # reset cached value loaded by dotenv
