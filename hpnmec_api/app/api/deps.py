from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.db.database import get_db
from app.crud.crud_user import user as crud_user
from app.models.user import User
# Remove UserResponse import if not directly used here
# from app.schemas.user import UserResponse 
from app.core.config import settings
from app.models.enums import UserRoleEnum

# Ensure the tokenUrl points to the correct login endpoint relative to the API prefix
# e.g., if prefix is /api/v1, tokenUrl should be "auth/login"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login") # Relative to /api/v1

# Common dependencies that can be used across endpoint modules
def get_db_session() -> Generator[Session, None, None]:
    """Dependency for getting DB session"""
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db:
            db.close()

async def get_current_user(
    db: Session = Depends(get_db_session), token: str = Depends(oauth2_scheme)
) -> User:
    """
    Decodes the JWT token and retrieves the user from the database.
    Raises HTTPException 401 if token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin đăng nhập", # Vietnamese: Could not validate credentials
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        # Add token type check if needed: token_type = payload.get("type")
    except JWTError:
        raise credentials_exception
    
    user = crud_user.get(db, id=int(user_id))
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Checks if the current user is active.
    Raises HTTPException 400 if the user is inactive.
    """
    if not crud_user.is_active(current_user):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tài khoản không hoạt động") # Vietnamese: Inactive account
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Checks if the current active user has the ADMIN role.
    Raises HTTPException 403 if the user is not an admin.
    """
    if current_user.role != UserRoleEnum.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền thực hiện hành động này", # Vietnamese: Not enough permissions
        )
    return current_user

async def get_current_doctor_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Checks if the current active user has the DOCTOR role.
    Raises HTTPException 403 if the user is not a doctor.
    """
    if current_user.role != UserRoleEnum.DOCTOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yêu cầu quyền bác sĩ", # Vietnamese: Doctor role required
        )
    return current_user

async def get_current_patient_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Checks if the current active user has the PATIENT role.
    Raises HTTPException 403 if the user is not a patient.
    """
    if current_user.role != UserRoleEnum.PATIENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yêu cầu quyền bệnh nhân", # Vietnamese: Patient role required
        )
    return current_user

# Example dependency for checking resource ownership (can be adapted)
# async def check_resource_owner(
#     resource_id: int, 
#     current_user: User = Depends(get_current_active_user),
#     db: Session = Depends(get_db_session)
# ) -> bool:
#     # Replace with actual logic to fetch the resource and check owner_id
#     # resource = crud_resource.get(db, id=resource_id) 
#     # if not resource or resource.owner_id != current_user.id:
#     #     raise HTTPException(status_code=403, detail="Không phải chủ sở hữu tài nguyên") # Vietnamese: Not resource owner
#     # return True
#     pass

# Add other common dependencies here
# These dependencies should not import from endpoint modules
