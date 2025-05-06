from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from app.schemas.appointment import AppointmentResponse, AppointmentCreate, AppointmentUpdate
from app.api import deps

router = APIRouter(
    prefix="/appointments",
    tags=["Appointments"],
    # dependencies=[Depends(get_current_active_user)],  # Uncomment when security is implemented
)

@router.get(
    "/",
    response_model=List[AppointmentResponse],
    status_code=status.HTTP_200_OK,
    summary="List all appointments",
    description="Retrieve a list of all appointments in the system.",
    responses={
        200: {
            "description": "A list of appointments.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "user_id": 1,
                            "doctor_id": 2,
                            "scheduled_time": "2024-06-01T10:00:00",
                            "status": "scheduled",
                            # ...other fields...
                        }
                    ]
                }
            }
        }
    }
)
async def list_appointments() -> List[AppointmentResponse]:
    """
    Retrieve all appointments.

    Returns:
        List[Appointment]: A list of appointment objects.
    """
    # ...existing code...

@router.post(
    "/",
    response_model=AppointmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new appointment",
    description="Schedule a new appointment for a user.",
    responses={
        201: {
            "description": "Appointment created successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 2,
                        "user_id": 1,
                        "doctor_id": 2,
                        "scheduled_time": "2024-06-02T14:00:00",
                        "status": "scheduled",
                        # ...other fields...
                    }
                }
            }
        },
        400: {"description": "Invalid input."}
    }
)
async def create_appointment(appointment_in: AppointmentCreate) -> AppointmentResponse:
    """
    Create a new appointment.

    Args:
        appointment_in (AppointmentCreate): The appointment information to create.

    Returns:
        Appointment: The created appointment object.
    """
    # ...existing code...

@router.get(
    "/{appointment_id}",
    response_model=AppointmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get appointment by ID",
    description="Retrieve an appointment by its unique ID.",
    responses={
        200: {
            "description": "Appointment found.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "user_id": 1,
                        "doctor_id": 2,
                        "scheduled_time": "2024-06-01T10:00:00",
                        "status": "scheduled",
                        # ...other fields...
                    }
                }
            }
        },
        404: {"description": "Appointment not found."}
    }
)
async def get_appointment(appointment_id: int) -> AppointmentResponse:
    """
    Get an appointment by ID.

    Args:
        appointment_id (int): The appointment's unique identifier.

    Returns:
        Appointment: The appointment object if found.

    Raises:
        HTTPException: If appointment is not found.
    """
    # ...existing code...

@router.put(
    "/{appointment_id}",
    response_model=AppointmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Update an appointment",
    description="Update the details of an existing appointment.",
    responses={
        200: {
            "description": "Appointment updated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "user_id": 1,
                        "doctor_id": 2,
                        "scheduled_time": "2024-06-01T11:00:00",
                        "status": "rescheduled",
                        # ...other fields...
                    }
                }
            }
        },
        404: {"description": "Appointment not found."}
    }
)
async def update_appointment(appointment_id: int, appointment_in: AppointmentUpdate) -> AppointmentResponse:
    """
    Update an appointment's information.

    Args:
        appointment_id (int): The appointment's unique identifier.
        appointment_in (AppointmentUpdate): The updated appointment information.

    Returns:
        Appointment: The updated appointment object.

    Raises:
        HTTPException: If appointment is not found.
    """
    # ...existing code...

@router.delete(
    "/{appointment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an appointment",
    description="Delete an appointment from the system by its ID.",
    responses={
        204: {"description": "Appointment deleted successfully."},
        404: {"description": "Appointment not found."}
    }
)
async def delete_appointment(appointment_id: int):
    """
    Delete an appointment by ID.

    Args:
        appointment_id (int): The appointment's unique identifier.

    Returns:
        None

    Raises:
        HTTPException: If appointment is not found.
    """
    # ...existing code...