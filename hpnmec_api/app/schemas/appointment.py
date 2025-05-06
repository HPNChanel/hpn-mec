from typing import Optional
from pydantic import BaseModel, validator, constr, conint, ConfigDict
from datetime import datetime

from app.models.enums import AppointmentStatusEnum
from app.schemas.base import TimestampSchema

# Base Appointment Schema
class AppointmentBase(BaseModel):
    patient_id: int
    doctor_id: int
    appointment_datetime: datetime
    duration_minutes: conint(ge=15, le=120) = 30
    reason: Optional[str] = None
    is_virtual: bool = False
    
    @validator('appointment_datetime')
    def validate_appointment_datetime(cls, v):
        if v < datetime.now():
            raise ValueError('Appointment datetime cannot be in the past')
        return v

# Creation Schema
class AppointmentCreate(AppointmentBase):
    pass

# Update Schema
class AppointmentUpdate(BaseModel):
    appointment_datetime: Optional[datetime] = None
    duration_minutes: Optional[conint(ge=15, le=120)] = None
    status: Optional[AppointmentStatusEnum] = None
    reason: Optional[str] = None
    notes: Optional[str] = None
    is_virtual: Optional[bool] = None
    meeting_link: Optional[constr(max_length=255)] = None
    
    @validator('appointment_datetime')
    def validate_appointment_datetime(cls, v):
        if v is not None and v < datetime.now():
            raise ValueError('Appointment datetime cannot be in the past')
        return v
    
    model_config = ConfigDict(from_attributes=True)

# Response Schema
class AppointmentResponse(AppointmentBase, TimestampSchema):
    id: int
    status: AppointmentStatusEnum
    notes: Optional[str] = None
    meeting_link: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)