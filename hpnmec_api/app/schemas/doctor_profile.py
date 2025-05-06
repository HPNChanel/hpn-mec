from typing import Optional
from pydantic import BaseModel, validator, conint, confloat, constr, ConfigDict
from app.schemas.base import TimestampSchema

# Base Doctor Profile Schema
class DoctorProfileBase(BaseModel):
    specialization: constr(max_length=100)
    qualification: constr(max_length=255)
    experience_years: conint(ge=0)
    license_number: constr(max_length=50)
    hospital_affiliation: Optional[constr(max_length=255)] = None
    consultation_fee: Optional[confloat(ge=0)] = None
    bio: Optional[str] = None
    consultation_hours_start: Optional[constr(max_length=5)] = None  # "HH:MM"
    consultation_hours_end: Optional[constr(max_length=5)] = None    # "HH:MM"
    is_available: bool = True
    
    @validator('consultation_hours_start', 'consultation_hours_end')
    def validate_time_format(cls, v):
        if v is not None:
            if not (len(v) == 5 and v[2] == ":" and 
                    v[0:2].isdigit() and v[3:5].isdigit() and
                    0 <= int(v[0:2]) <= 23 and 0 <= int(v[3:5]) <= 59):
                raise ValueError('Time must be in format "HH:MM"')
        return v

# Creation Schema
class DoctorProfileCreate(DoctorProfileBase):
    user_id: int

# Update Schema
class DoctorProfileUpdate(BaseModel):
    specialization: Optional[constr(max_length=100)] = None
    qualification: Optional[constr(max_length=255)] = None
    experience_years: Optional[conint(ge=0)] = None
    hospital_affiliation: Optional[constr(max_length=255)] = None
    consultation_fee: Optional[confloat(ge=0)] = None
    bio: Optional[str] = None
    consultation_hours_start: Optional[constr(max_length=5)] = None
    consultation_hours_end: Optional[constr(max_length=5)] = None
    is_available: Optional[bool] = None
    
    @validator('consultation_hours_start', 'consultation_hours_end')
    def validate_time_format(cls, v):
        if v is not None:
            if not (len(v) == 5 and v[2] == ":" and 
                    v[0:2].isdigit() and v[3:5].isdigit() and
                    0 <= int(v[0:2]) <= 23 and 0 <= int(v[3:5]) <= 59):
                raise ValueError('Time must be in format "HH:MM"')
        return v
    
    model_config = ConfigDict(from_attributes=True)

# Response Schema
class DoctorProfileResponse(DoctorProfileBase, TimestampSchema):
    id: int
    user_id: int
    rating: float = 0.0
    total_ratings: int = 0
    
    model_config = ConfigDict(from_attributes=True)