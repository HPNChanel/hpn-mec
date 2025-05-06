from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
# Change HealthRecord to HealthRecordResponse
from app.schemas.health_record import HealthRecordResponse, HealthRecordCreate, HealthRecordUpdate
from app.api import deps

router = APIRouter(
    prefix="/health-records",
    tags=["Health Records"],
    # dependencies=[Depends(get_current_active_user)],  # Uncomment when security is implemented
)

@router.get(
    "/",
    response_model=List[HealthRecordResponse],  # Update here
    status_code=status.HTTP_200_OK,
    summary="List all health records",
    description="Retrieve a list of all health records in the system.",
    responses={
        200: {
            "description": "A list of health records.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "user_id": 1,
                            "diagnosis": "Hypertension",
                            "notes": "Patient is responding well to medication.",
                            # ...other fields...
                        }
                    ]
                }
            }
        }
    }
)
async def list_health_records() -> List[HealthRecordResponse]:  # Update return type
    """
    Retrieve all health records.

    Returns:
        List[HealthRecordResponse]: A list of health record objects.
    """
    # ...existing code...

@router.post(
    "/",
    response_model=HealthRecordResponse,  # Update here
    status_code=status.HTTP_201_CREATED,
    summary="Create a new health record",
    description="Add a new health record for a user.",
    responses={
        201: {
            "description": "Health record created successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 2,
                        "user_id": 1,
                        "diagnosis": "Diabetes",
                        "notes": "Patient requires regular monitoring.",
                        # ...other fields...
                    }
                }
            }
        },
        400: {"description": "Invalid input."}
    }
)
async def create_health_record(record_in: HealthRecordCreate) -> HealthRecordResponse:  # Update return type
    """
    Create a new health record.

    Args:
        record_in (HealthRecordCreate): The health record information to create.

    Returns:
        HealthRecordResponse: The created health record object.
    """
    # ...existing code...

@router.get(
    "/{record_id}",
    response_model=HealthRecordResponse,  # Update here
    status_code=status.HTTP_200_OK,
    summary="Get health record by ID",
    description="Retrieve a health record by its unique ID.",
    responses={
        200: {
            "description": "Health record found.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "user_id": 1,
                        "diagnosis": "Hypertension",
                        "notes": "Patient is responding well to medication.",
                        # ...other fields...
                    }
                }
            }
        },
        404: {"description": "Health record not found."}
    }
)
async def read_health_record(record_id: int) -> HealthRecordResponse:  # Update return type
    """
    Get a health record by ID.

    Args:
        record_id (int): The health record's unique identifier.

    Returns:
        HealthRecordResponse: The health record object if found.

    Raises:
        HTTPException: If health record is not found.
    """
    # ...existing code...

@router.put(
    "/{record_id}",
    response_model=HealthRecordResponse,  # Update here
    status_code=status.HTTP_200_OK,
    summary="Update a health record",
    description="Update the details of an existing health record.",
    responses={
        200: {
            "description": "Health record updated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "user_id": 1,
                        "diagnosis": "Hypertension (controlled)",
                        "notes": "Medication adjusted.",
                        # ...other fields...
                    }
                }
            }
        },
        404: {"description": "Health record not found."}
    }
)
async def update_health_record(record_id: int, record_in: HealthRecordUpdate) -> HealthRecordResponse:  # Update return type
    """
    Update a health record's information.

    Args:
        record_id (int): The health record's unique identifier.
        record_in (HealthRecordUpdate): The updated health record information.

    Returns:
        HealthRecordResponse: The updated health record object.

    Raises:
        HTTPException: If health record is not found.
    """
    # ...existing code...

@router.delete(
    "/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a health record",
    description="Delete a health record from the system by its ID.",
    responses={
        204: {"description": "Health record deleted successfully."},
        404: {"description": "Health record not found."}
    }
)
async def delete_health_record(record_id: int):
    """
    Delete a health record by ID.

    Args:
        record_id (int): The health record's unique identifier.

    Returns:
        None

    Raises:
        HTTPException: If health record is not found.
    """
    # ...existing code...