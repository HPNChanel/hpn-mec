from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from sqlalchemy.orm import Session
from app.schemas.user import UserResponse, UserUpdate
from app.db.database import get_db
from app.crud.crud_user import user as crud_user
from app.api.deps import get_current_admin_user
from app.schemas.response import ResponseModel

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(get_current_admin_user)], 
)

@router.get(
    "/",
    response_model=List[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="List all users (Admin only)",
    description="Retrieve a list of all registered users in the system. Requires admin privileges.",
    responses={
        200: {
            "description": "A list of users.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "username": "johndoe",
                            "email": "johndoe@example.com",
                            "full_name": "John Doe",
                            # ...other fields...
                        }
                    ]
                }
            }
        }
    }
)
async def list_users(db: Session = Depends(get_db)) -> List[UserResponse]:
    """
    Retrieve all users.

    Returns:
        List[UserResponse]: A list of user objects.
    """
    users = crud_user.get_multi(db, skip=0, limit=100)
    return users

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user by ID (Admin only)",
    description="Retrieve a user by their unique ID. Requires admin privileges.",
    responses={
        200: {
            "description": "User found.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "username": "johndoe",
                        "email": "johndoe@example.com",
                        "full_name": "John Doe",
                        # ...other fields...
                    }
                }
            }
        },
        404: {"description": "User not found."}
    }
)
async def get_user(user_id: int, db: Session = Depends(get_db)) -> UserResponse:
    """
    Get a user by ID.

    Args:
        user_id (int): The user's unique identifier.

    Returns:
        UserResponse: The user object if found.

    Raises:
        HTTPException: If user is not found.
    """
    user = crud_user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
    return user

@router.put(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a user (Admin only)",
    description="Update the details of an existing user. Requires admin privileges.",
    responses={
        200: {
            "description": "User updated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "username": "johndoe",
                        "email": "johndoe@example.com",
                        "full_name": "John Doe Updated",
                        # ...other fields...
                    }
                }
            }
        },
        404: {"description": "User not found."}
    }
)
async def update_user(user_id: int, user_in: UserUpdate, db: Session = Depends(get_db)) -> UserResponse:
    """
    Update a user's information.

    Args:
        user_id (int): The user's unique identifier.
        user_in (UserUpdate): The updated user information.

    Returns:
        UserResponse: The updated user object.

    Raises:
        HTTPException: If user is not found.
    """
    user = crud_user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
        
    user = crud_user.update(db, db_obj=user, obj_in=user_in)
    return user

@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a user (Admin only)",
    description="Delete a user from the system by their ID. Requires admin privileges.",
    responses={
        204: {"description": "User deleted successfully."},
        404: {"description": "User not found."}
    }
)
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete a user by ID.

    Args:
        user_id (int): The user's unique identifier.

    Returns:
        None

    Raises:
        HTTPException: If user is not found.
    """
    user = crud_user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
        
    crud_user.remove(db, id=user_id)