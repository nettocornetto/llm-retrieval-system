from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    if provided_key and provided_key in settings.allowed_api_keys:
        return provided_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key.",
    )
