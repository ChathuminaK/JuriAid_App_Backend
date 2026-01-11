from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List
from fastapi.responses import JSONResponse

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

# CORS configuration 
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("🚀 Starting JuriAid Auth Service...")
    database.init_db()
    print("✅ Auth Service is ready!")

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "JuriAid Auth Service",
        "version": "1.0.0"
    }

@app.post("/auth/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserCreate, db: Session = Depends(database.get_db)):
    """
    Register a new user account
    
    - **email**: Valid email address (must be unique)
    - **password**: Minimum 6 characters
    - **full_name**: Optional full name
    - **phone**: Optional phone number
    """
    # Check if user already exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user = create_user(db, user_data)
    
    # Generate JWT token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return Token(
        access_token=access_token,
        user=UserResponse.from_orm(user)
    )

@app.post("/auth/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(database.get_db)):
    """
    Login with email and password
    
    Returns JWT access token valid for 24 hours
    """
    # Find user by email
    user = get_user_by_email(db, credentials.email)
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Contact support."
        )
    
    # Update last login timestamp
    update_last_login(db, user.id)
    
    # Generate JWT token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return Token(
        access_token=access_token,
        user=UserResponse.from_orm(user)
    )

@app.get("/auth/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user profile
    
    Requires valid JWT token in Authorization header
    """
    return UserResponse.from_orm(current_user)

@app.put("/auth/me", response_model=UserResponse)
def update_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Update current user's profile
    
    - **full_name**: Update full name
    - **phone**: Update phone number
    """
    updated_user = update_user_profile(db, current_user.id, user_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(updated_user)

@app.get("/auth/verify")
def verify_token(current_user: User = Depends(get_current_user)):
    """
    Verify JWT token validity
    
    Used by orchestratorc to validate user authentication
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value,
        "subscription_tier": current_user.subscription_tier.value
    }

@app.post("/auth/logout", response_model=MessageResponse)
def logout(current_user: User = Depends(get_current_user)):
    """
    Logout user (client must delete token)
    
    This endpoint confirms the token is valid before logout.
    The client should delete the JWT token from storage.
    """
    return MessageResponse(message="Successfully logged out. Please delete your token.")

# Admin endpoints
@app.get("/admin/users", response_model=List[UserResponse])
def list_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(database.get_db)
):
    """
    List all users (Admin only)
    
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    """
    users = get_all_users(db, skip=skip, limit=limit)
    return [UserResponse.from_orm(user) for user in users]

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error",
            "error": str(exc)
        }
    )
