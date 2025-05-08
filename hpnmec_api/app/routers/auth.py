from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

# Verify correct schema imports
from app.crud.crud_user import user as crud_user # Renamed import
from app.schemas.user import UserResponse, UserLogin, UserCreate
from app.schemas.auth import TokenResponse, Token # Import new schemas
from app.db.database import get_db
from app.core.security import create_access_token # Import token creation function
from app.core.config import settings # Import settings
# Import dependency for getting current user
from app.api.deps import get_current_active_user 
from app.models.user import User # Import User model for dependency type hint
from app.schemas.response import ResponseModel

router = APIRouter()

@router.post("/login", response_model=ResponseModel)
def login(
    *,
    db: Session = Depends(get_db),
    login_data: UserLogin,
) -> Any:
    """
    Login with email and password, returns user info and access token.
    """
    user_obj = crud_user.authenticate(db, email=login_data.email, password=login_data.password)
    if not user_obj:
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không chính xác") # Vietnamese error
    
    if not crud_user.is_active(user_obj):
        raise HTTPException(status_code=400, detail="Tài khoản không hoạt động") # Vietnamese error
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user_obj.id, expires_delta=access_token_expires
    )
    
    # Return token and user data
    result = {
        "token": {
            "access_token": access_token,
            "token_type": "bearer"
        },
        "user": user_obj # UserResponse schema will handle formatting
    }
    return {"status": "success", "data": result, "message": "Login successful"}

@router.post(
    "/register", 
    response_model=ResponseModel,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email, password, and personal information.",
    responses={
        201: {"description": "User created successfully"},
        400: {"description": "Email or username already registered"},
        422: {"description": "Validation error in input data - check that all required fields are provided and username contains only letters, numbers, and underscores with no spaces"}
    }
)
def register(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Register a new user account.
    
    Args:
        user_in (UserCreate): The user registration information
    
    Returns:
        ResponseModel: The status, created user data, and message
        
    Raises:
        HTTPException: If email already exists (400) or validation fails (422)
    """
    # Check if user with this email already exists
    user_obj = crud_user.get_by_email(db, email=user_in.email)
    if user_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists."
        )
    
    # Check if user with this username already exists (if username is in the model)
    if hasattr(user_in, 'username') and user_in.username:
        user_obj = crud_user.get_by_username(db, username=user_in.username)
        if user_obj:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The user with this username already exists."
            )
    
    # Create user - this will handle password hashing through the CRUD method
    user_obj = crud_user.create(db, obj_in=user_in) # Use renamed crud_user
    return {"status": "success", "data": user_obj, "message": "Registration successful"}

@router.get("/me", response_model=ResponseModel)
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current logged-in user's information.
    """
    # The dependency already fetches and validates the user
    return {"status": "success", "data": current_user, "message": "User profile retrieved"}

# Optional: Add /refresh endpoint if needed
# @router.post("/refresh", response_model=Token)
# async def refresh_token(current_user: User = Depends(get_current_user)):
#     """
#     Generate a new access token for the current user.
#     """
#     access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         subject=current_user.id, expires_delta=access_token_expires
#     )
#     return {"access_token": access_token, "token_type": "bearer"}