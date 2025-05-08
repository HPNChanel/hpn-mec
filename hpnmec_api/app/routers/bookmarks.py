from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# Replace Bookmark with UserDoctorBookmarkResponse if needed
from app.crud.crud_user_doctor_bookmark import user_doctor_bookmark
from app.crud.crud_user import user
from app.crud.crud_doctor_profile import doctor_profile
from app.schemas.user_doctor_bookmark import UserDoctorBookmarkResponse, UserDoctorBookmarkCreate
from app.db.database import get_db
from app.schemas.response import ResponseModel

router = APIRouter()

@router.get("/", response_model=List[UserDoctorBookmarkResponse])
def read_bookmarks(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
) -> Any:
    """
    Retrieve bookmarks.
    """
    bookmarks = user_doctor_bookmark.get_multi(db, skip=skip, limit=limit)
    return bookmarks

@router.post("/", response_model=UserDoctorBookmarkResponse, status_code=status.HTTP_201_CREATED)
def create_bookmark(
    *,
    db: Session = Depends(get_db),
    bookmark_in: UserDoctorBookmarkCreate,
) -> Any:
    """
    Create new bookmark.
    """
    # Check if user exists
    user = user.get(db, id=bookmark_in.user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    # Check if doctor exists
    doctor = doctor_profile.get(db, id=bookmark_in.doctor_id)
    if not doctor:
        raise HTTPException(
            status_code=404,
            detail="Doctor not found"
        )
    
    # Check if bookmark already exists
    existing_bookmark = user_doctor_bookmark.get_by_user_doctor(
        db, user_id=bookmark_in.user_id, doctor_id=bookmark_in.doctor_id
    )
    if existing_bookmark:
        return existing_bookmark
    
    bookmark = user_doctor_bookmark.create(db, obj_in=bookmark_in)
    return bookmark

@router.get("/{bookmark_id}", response_model=UserDoctorBookmarkResponse)
def read_bookmark(
    *,
    db: Session = Depends(get_db),
    bookmark_id: int,
) -> Any:
    """
    Get bookmark by ID.
    """
    bookmark = user_doctor_bookmark.get(db, id=bookmark_id)
    if not bookmark:
        raise HTTPException(
            status_code=404,
            detail="Bookmark not found"
        )
    return bookmark

@router.delete("/{bookmark_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_bookmark(
    *,
    db: Session = Depends(get_db),
    bookmark_id: int,
) -> None:  # Changed return type from 'Any' to 'None'
    """
    Delete a bookmark.
    """
    bookmark = user_doctor_bookmark.get(db, id=bookmark_id)
    if not bookmark:
        raise HTTPException(
            status_code=404,
            detail="Bookmark not found"
        )
    user_doctor_bookmark.remove(db, id=bookmark_id)

@router.get("/user/{user_id}", response_model=List[UserDoctorBookmarkResponse])
def read_user_bookmarks(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve bookmarks for a specific user.
    """
    bookmarks = user_doctor_bookmark.get_by_user(db, user_id=user_id, skip=skip, limit=limit)
    return bookmarks

@router.get("/doctor/{doctor_id}", response_model=List[UserDoctorBookmarkResponse])
def read_doctor_bookmarks(
    *,
    db: Session = Depends(get_db),
    doctor_id: int,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve bookmarks for a specific doctor.
    """
    bookmarks = user_doctor_bookmark.get_by_doctor(db, doctor_id=doctor_id, skip=skip, limit=limit)
    return bookmarks

@router.delete("/user/{user_id}/doctor/{doctor_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_bookmark_by_user_doctor(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    doctor_id: int,
) -> None:  # Changed return type from 'Any' to 'None'
    """
    Delete a bookmark by user and doctor.
    """
    bookmark = user_doctor_bookmark.get_by_user_doctor(db, user_id=user_id, doctor_id=doctor_id)
    if not bookmark:
        raise HTTPException(
            status_code=404,
            detail="Bookmark not found"
        )
    user_doctor_bookmark.remove_by_user_doctor(db, user_id=user_id, doctor_id=doctor_id)