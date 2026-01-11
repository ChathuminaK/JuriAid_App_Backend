import httpx
from fastapi import HTTPException, Header
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://127.0.0.1:8001")

async def verify_user_token(authorization: Optional[str] = Header(None)):
    """Middleware to verify JWT token via auth_service"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{AUTH_SERVICE_URL}/auth/verify",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Auth service unavailable")