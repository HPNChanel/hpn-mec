from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# Import CRUD operations and schemas
from app.crud.crud_doctor_profile import doctor_profile
from app.crud.crud_user import user as crud_user # Renamed import
from app.schemas.doctor_profile import DoctorProfileResponse, DoctorProfileCreate, DoctorProfileUpdate
# Import UserResponse for type hint if needed, otherwise remove
# from app.schemas.user import UserResponse 
from app.db.database import get_db # Use the correct get_db from database.py
# Import dependencies for authentication and authorization
from app.api.deps import get_current_admin_user, get_current_active_user, get_db_session
from app.models.user import User # Import User model for dependency type hint
from app.schemas.response import ResponseModel

# Define the router with prefix and tags
router = APIRouter(
    prefix="/doctors",
    tags=["doctors"]
)

@router.get(
    "/", 
    response_model=List[DoctorProfileResponse],
    summary="Retrieve all doctor profiles",
    description="Fetches a list of all doctor profiles, available to any authenticated user."
)
def read_doctors(
    db: Session = Depends(get_db_session), # Use get_db_session
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user) # Ensure user is authenticated
) -> Any:
    """
    Retrieve doctors. Accessible by any authenticated user.
    """
    doctors = doctor_profile.get_multi(db, skip=skip, limit=limit)
    return doctors

@router.post(
    "/", 
    response_model=DoctorProfileResponse, 
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_current_admin_user)], # Enforce admin-only access
    summary="Create a new doctor profile (Admin only)",
    description="Creates a new doctor profile linked to an existing user. Requires admin privileges."
)
def create_doctor(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    doctor_in: DoctorProfileCreate,
    # current_user: User = Depends(get_current_admin_user) # Dependency already handles check
) -> Any:
    """
    Create new doctor profile. (Admin only)
    """
    # Check if user exists
    user_obj = crud_user.get(db, id=doctor_in.user_id) # Use renamed crud_user
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Người dùng không tồn tại" # Vietnamese error
        )
    
    # Check if doctor profile already exists for this user
    existing_doctor = doctor_profile.get_by_user_id(db, user_id=doctor_in.user_id)
    if existing_doctor:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hồ sơ bác sĩ đã tồn tại cho người dùng này" # Vietnamese error
        )
    
    # Check if license number is unique
    existing_license = doctor_profile.get_by_license_number(db, license_number=doctor_in.license_number)
    if existing_license:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bác sĩ với số giấy phép này đã tồn tại" # Vietnamese error
        )
    
    new_doctor = doctor_profile.create(db, obj_in=doctor_in)
    return new_doctor

@router.get(
    "/{doctor_id}", 
    response_model=DoctorProfileResponse,
    summary="Get doctor profile by ID",
    description="Retrieves details of a specific doctor profile by its ID. Available to any authenticated user."
)
def read_doctor(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    doctor_id: int,
    current_user: User = Depends(get_current_active_user) # Ensure user is authenticated
) -> Any:
    """
    Get doctor by ID. Accessible by any authenticated user.
    """
    doctor = doctor_profile.get(db, id=doctor_id)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bác sĩ" # Vietnamese error
        )
    return doctor

@router.put(
    "/{doctor_id}", 
    response_model=DoctorProfileResponse,
    dependencies=[Depends(get_current_admin_user)], # Enforce admin-only access
    summary="Update a doctor profile (Admin only)",
    description="Updates an existing doctor profile. Requires admin privileges."
)
def update_doctor(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    doctor_id: int,
    doctor_in: DoctorProfileUpdate,
    # current_user: User = Depends(get_current_admin_user) # Dependency already handles check
) -> Any:
    """
    Update a doctor. (Admin only)
    """
    doctor = doctor_profile.get(db, id=doctor_id)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bác sĩ" # Vietnamese error
        )
    # Check for license number conflict if license_number is being updated
    if doctor_in.license_number and doctor_in.license_number != doctor.license_number:
        existing_license = doctor_profile.get_by_license_number(db, license_number=doctor_in.license_number)
        if existing_license and existing_license.id != doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bác sĩ với số giấy phép này đã tồn tại" # Vietnamese error
            )

    updated_doctor = doctor_profile.update(db, db_obj=doctor, obj_in=doctor_in)
    return updated_doctor

@router.delete(
    "/{doctor_id}", 
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(get_current_admin_user)], # Enforce admin-only access
    summary="Delete a doctor profile (Admin only)",
    description="Deletes a doctor profile from the system. Requires admin privileges."
)
def delete_doctor(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    doctor_id: int,
    # current_user: User = Depends(get_current_admin_user) # Dependency already handles check
) -> None:
    """
    Delete a doctor. (Admin only)
    """
    doctor = doctor_profile.get(db, id=doctor_id)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bác sĩ" # Vietnamese error
        )
    doctor_profile.remove(db, id=doctor_id)
    # No content returned on successful deletion

@router.get(
    "/by-user/{user_id}", 
    response_model=DoctorProfileResponse,
    summary="Get doctor profile by User ID",
    description="Retrieves a doctor profile linked to a specific user ID. Available to any authenticated user."
)
def read_doctor_by_user(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    user_id: int,
    current_user: User = Depends(get_current_active_user) # Ensure user is authenticated
) -> Any:
    """
    Get doctor by user ID. Accessible by any authenticated user.
    """
    doctor = doctor_profile.get_by_user_id(db, user_id=user_id)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy hồ sơ bác sĩ cho người dùng này" # Vietnamese error
        )
    return doctor

@router.get(
    "/specialization/{specialization}", 
    response_model=List[DoctorProfileResponse],
    summary="Get doctors by specialization",
    description="Retrieves a list of doctors matching a specific specialization. Available to any authenticated user."
)
def read_doctors_by_specialization(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    specialization: str,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user) # Ensure user is authenticated
) -> Any:
    """
    Retrieve doctors by specialization. Accessible by any authenticated user.
    """
    doctors = doctor_profile.get_by_specialization(db, specialization=specialization, skip=skip, limit=limit)
    return doctors

@router.get(
    "/available/", 
    response_model=List[DoctorProfileResponse],
    summary="Get available doctors",
    description="Retrieves a list of doctors currently marked as available. Available to any authenticated user."
)
def read_available_doctors(
    db: Session = Depends(get_db_session), # Use get_db_session
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user) # Ensure user is authenticated
) -> Any:
    """
    Retrieve available doctors. Accessible by any authenticated user.
    """
    doctors = doctor_profile.get_available_doctors(db, skip=skip, limit=limit)
    return doctors

# Rating endpoint - Requires authenticated user, but not admin
@router.post(
    "/{doctor_id}/rating/{rating}", 
    response_model=DoctorProfileResponse,
    dependencies=[Depends(get_current_active_user)], # Requires authenticated user
    summary="Rate a doctor",
    description="Allows an authenticated user to submit a rating for a doctor."
)
def update_doctor_rating(
    *,
    db: Session = Depends(get_db_session), # Use get_db_session
    doctor_id: int,
    rating: float, # Use path parameter for rating
    current_user: User = Depends(get_current_active_user) # Inject current user
) -> Any:
    """
    Update doctor rating. Accessible by any authenticated user.
    """
    if not (0 <= rating <= 5): # Simplified check
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đánh giá phải từ 0 đến 5" # Vietnamese error
        )
    
    doctor = doctor_profile.get(db, id=doctor_id)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bác sĩ" # Vietnamese error
        )
    
    # Consider preventing users from rating multiple times or rating themselves if needed
    
    updated_doctor = doctor_profile.update_rating(db, doctor_id=doctor_id, new_rating=rating)
    return updated_doctor