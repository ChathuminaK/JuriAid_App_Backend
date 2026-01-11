from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    password: str = Field(..., min_length=6, description="Minimum 6 characters")
    full_name: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=20)

class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    full_name: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=20)

class UserResponse(BaseModel):
    """Schema for user data in responses"""
    id: str
    email: str
    full_name: Optional[str]
    phone: Optional[str]
    role: str
    subscription_tier: str
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    """Schema for decoded JWT token data"""
    user_id: Optional[str] = None
    email: Optional[str] = None

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str