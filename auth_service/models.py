from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum
from datetime import datetime
import uuid
import enum

from database import Base

class SubscriptionTier(str, enum.Enum):
    """Subscription tiers for freemium model"""
    FREE = "free"
    PREMIUM = "premium"

class UserRole(str, enum.Enum):
    """User roles for access control"""
    LAWYER = "lawyer"
    ADMIN = "admin"

class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    role = Column(SQLEnum(UserRole), default=UserRole.LAWYER, nullable=False)
    subscription_tier = Column(SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"