"""External integrations subsystem."""
from infrastructure.external_integrations.interfaces import (
    IDecodeSupabaseJWT,
    IGetJWTSecret,
)
from infrastructure.external_integrations.supabase_jwt import (
    decode_supabase_jwt,
    get_jwt_secret,
)
from infrastructure.external_integrations.nlp_parser import (
    parse_game_context,
    context_to_context_ml_params,
)
from infrastructure.external_integrations.nlp_explain import (
    explain_recommendations,
    explain_shotplan,
)

__all__ = [
    "IDecodeSupabaseJWT",
    "IGetJWTSecret",
    "decode_supabase_jwt",
    "get_jwt_secret",
    "parse_game_context",
    "context_to_context_ml_params",
    "explain_recommendations",
    "explain_shotplan",
]
