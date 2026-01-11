from sqlalchemy.orm import Session
from datetime import datetime

from models import User
from auth import get_password_hash
from schemas import UserCreate, UserUpdate

def get_user_by_email(db: Session, email: str) -> User | None:
    """Get a user by email address"""
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: str) -> User | None:
    """Get a user by ID"""
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, user_data: UserCreate) -> User:
    """Create a new user"""
    hashed_password = get_password_hash(user_data.password)
    
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        phone=user_data.phone
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_last_login(db: Session, user_id: str) -> None:
    """Update user's last login timestamp"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = datetime.utcnow()
        db.commit()

def update_user_profile(db: Session, user_id: str, user_data: UserUpdate) -> User | None:
    """Update user profile information"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        return None
    
    if user_data.full_name is not None:
        user.full_name = user_data.full_name
    if user_data.phone is not None:
        user.phone = user_data.phone
    
    db.commit()
    db.refresh(user)
    return user

def get_all_users(db: Session, skip: int = 0, limit: int = 100):
    """Get all users (admin only)"""
    return db.query(User).offset(skip).limit(limit).all()

def deactivate_user(db: Session, user_id: str) -> User | None:
    """Deactivate a user account"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.is_active = False
        db.commit()
        db.refresh(user)
    return user