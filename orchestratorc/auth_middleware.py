import httpx
import logging
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer()
settings = get_settings()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Verify JWT token via Auth Service /auth/verify."""

    if not settings.AUTH_ENABLED:
        logger.warning("⚠️ Auth disabled - dev mode")
        return {"sub": 0, "email": "dev@juriaid.lk", "role": "user"}

    token = credentials.credentials

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.AUTH_SERVICE_URL}/auth/verify",
                headers={"Authorization": f"Bearer {token}"},
            )

        if resp.status_code == 200:
            data = resp.json()
            return {
                "sub": data.get("user_id"),
                "email": data.get("email"),
                "role": data.get("role"),
            }

        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    except httpx.RequestError as e:
        logger.error(f"Auth service unreachable: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Auth service unavailable")