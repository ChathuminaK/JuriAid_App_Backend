from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List, Optional
import os, uuid

import database
from models import User
from schemas import (
    UserCreate,
    UserLogin,
    UserUpdate,
    UserResponse,
    Token,
    MessageResponse
)
from auth import (
    verify_password,
    create_access_token,
    get_current_user,
    get_current_active_admin,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from crud import (
    get_user_by_email,
    create_user,
    update_last_login,
    update_user_profile,
    get_all_users
)
from config import settings

app = FastAPI(
    title="JuriAid Auth Service",
    version="1.0.0",
    description="Authentication & User Management Service for JuriAid Legal AI System",
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Static file serving for uploaded profile icons
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads", "profile_icons")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "uploads")), name="static")


@app.on_event("startup")
def startup_event():
    print("🚀 Starting JuriAid Auth Service...")
    database.init_db()
    print("✅ Auth Service is ready!")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "JuriAid Auth Service", "version": "1.0.0"}


@app.post("/auth/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserCreate, db: Session = Depends(database.get_db)):
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    user = create_user(db, user_data)
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return Token(access_token=access_token, user=UserResponse.from_orm(user))


@app.post("/auth/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(database.get_db)):
    user = get_user_by_email(db, credentials.email)
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive. Contact support.")
    update_last_login(db, user.id)
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return Token(access_token=access_token, user=UserResponse.from_orm(user))


@app.get("/auth/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)


@app.put("/auth/me", response_model=UserResponse)
def update_profile(
    full_name: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """Update profile — accepts multipart/form-data"""
    icon_url = current_user.profile_icon_url  # keep existing if no new image

    if profile_image is not None:
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if profile_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG, PNG, WebP, or GIF images are allowed"
            )
        contents = profile_image.file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too large. Maximum size is 5MB."
            )
        ext = profile_image.filename.rsplit(".", 1)[-1].lower() if "." in profile_image.filename else "jpg"
        filename = f"{current_user.id}_{uuid.uuid4().hex}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        icon_url = f"{settings.BASE_URL}/static/profile_icons/{filename}"

    user_data = UserUpdate(
        full_name=full_name,
        phone=phone,
        profile_icon_url=icon_url
    )
    updated_user = update_user_profile(db, current_user.id, user_data)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse.from_orm(updated_user)


@app.get("/auth/verify")
def verify_token(current_user: User = Depends(get_current_user)):
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value,
        "subscription_tier": current_user.subscription_tier.value
    }


@app.post("/auth/logout", response_model=MessageResponse)
def logout(current_user: User = Depends(get_current_user)):
    return MessageResponse(message="Successfully logged out. Please delete your token.")


@app.get("/admin/users", response_model=List[UserResponse])
def list_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(database.get_db)
):
    users = get_all_users(db, skip=skip, limit=limit)
    return [UserResponse.from_orm(user) for user in users]


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Internal server error", "error": str(exc)}
    )
