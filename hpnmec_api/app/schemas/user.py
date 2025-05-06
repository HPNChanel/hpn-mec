from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator, constr, ConfigDict
from datetime import date
import re

from app.models.enums import GenderEnum, UserRoleEnum
from app.schemas.base import TimestampSchema

# Base User Schema
class UserBase(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    full_name: constr(min_length=1, max_length=100)
    phone_number: Optional[constr(max_length=20)] = None
    date_of_birth: Optional[date] = None
    gender: Optional[GenderEnum] = None
    role: UserRoleEnum = UserRoleEnum.PATIENT
    address: Optional[constr(max_length=255)] = None
    profile_picture: Optional[constr(max_length=255)] = None
    
    @field_validator('username')
    @classmethod
    def username_validator(cls, v):
        # Check for whitespace
        if re.search(r'\s', v):
            raise ValueError('Username cannot contain whitespace')
        
        # Check for alphanumeric + underscores only
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        
        return v

# Creation Schema
class UserCreate(UserBase):
    password: constr(min_length=8, max_length=100)
    
    @field_validator('password')  # Changed from @validator
    @classmethod  # Add this decorator
    def password_complexity(cls, v):
        # Basic password complexity check
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

# Update Schema
class UserUpdate(BaseModel):
    username: Optional[constr(min_length=3, max_length=50)] = None
    email: Optional[EmailStr] = None
    full_name: Optional[constr(min_length=1, max_length=100)] = None
    phone_number: Optional[constr(max_length=20)] = None
    date_of_birth: Optional[date] = None
    gender: Optional[GenderEnum] = None
    address: Optional[constr(max_length=255)] = None
    profile_picture: Optional[constr(max_length=255)] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    
    model_config = ConfigDict(from_attributes=True)

# Password Update Schema
class UserPasswordUpdate(BaseModel):
    current_password: str
    new_password: constr(min_length=8, max_length=100)
    
    @field_validator('new_password')  # Changed from @validator
    @classmethod  # Add this decorator
    def password_complexity(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

# Response Schema
class UserResponse(UserBase, TimestampSchema):
    id: int
    is_active: bool
    is_verified: bool
    
    model_config = ConfigDict(from_attributes=True)

class UserLogin(BaseModel):
    email: EmailStr
    password: str