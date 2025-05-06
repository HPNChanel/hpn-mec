from typing import Optional, Dict, Any
from pydantic import BaseModel, confloat, conint, validator, ConfigDict
from datetime import date

from app.models.enums import BloodTypeEnum
from app.schemas.base import TimestampSchema

# Base Health Record Schema
class HealthRecordBase(BaseModel):
    user_id: int
    record_date: date
    
    # Basic health vitals
    height: Optional[confloat(gt=0)] = None  # in cm
    weight: Optional[confloat(gt=0)] = None  # in kg
    bmi: Optional[confloat(gt=0)] = None
    blood_pressure_systolic: Optional[conint(ge=0)] = None  # in mmHg
    blood_pressure_diastolic: Optional[conint(ge=0)] = None  # in mmHg
    heart_rate: Optional[conint(ge=0)] = None  # in bpm
    temperature: Optional[confloat(gt=0)] = None  # in Celsius
    blood_oxygen: Optional[confloat(ge=0, le=100)] = None  # SpO2 in percentage
    blood_glucose: Optional[confloat(ge=0)] = None  # in mg/dL
    blood_type: Optional[BloodTypeEnum] = None
    
    # Medical history
    allergies: Optional[str] = None
    chronic_diseases: Optional[str] = None
    current_medications: Optional[str] = None
    family_medical_history: Optional[str] = None
    
    # Additional data
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    notes: Optional[str] = None
    
    @validator('blood_pressure_systolic', 'blood_pressure_diastolic')
    def validate_blood_pressure(cls, v):
        if v is not None and (v < 0 or v > 300):
            raise ValueError('Blood pressure must be between 0 and 300 mmHg')
        return v
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if v is not None and (v < 0 or v > 250):
            raise ValueError('Heart rate must be between 0 and 250 bpm')
        return v

# Creation Schema
class HealthRecordCreate(HealthRecordBase):
    lab_results: Optional[Dict[str, Any]] = None
    appointment_id: Optional[int] = None

# Update Schema
class HealthRecordUpdate(BaseModel):
    record_date: Optional[date] = None
    height: Optional[confloat(gt=0)] = None
    weight: Optional[confloat(gt=0)] = None
    bmi: Optional[confloat(gt=0)] = None
    blood_pressure_systolic: Optional[conint(ge=0)] = None
    blood_pressure_diastolic: Optional[conint(ge=0)] = None
    heart_rate: Optional[conint(ge=0)] = None
    temperature: Optional[confloat(gt=0)] = None
    blood_oxygen: Optional[confloat(ge=0, le=100)] = None
    blood_glucose: Optional[confloat(ge=0)] = None
    blood_type: Optional[BloodTypeEnum] = None
    allergies: Optional[str] = None
    chronic_diseases: Optional[str] = None
    current_medications: Optional[str] = None
    family_medical_history: Optional[str] = None
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    lab_results: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    appointment_id: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)

# Response Schema
class HealthRecordResponse(HealthRecordBase, TimestampSchema):
    id: int
    lab_results: Optional[Dict[str, Any]] = None
    appointment_id: Optional[int] = None
    anomaly_score: Optional[float] = None
    risk_factors: Optional[Dict[str, Any]] = None
    health_insights: Optional[Dict[str, Any]] = None
    predictive_indicators: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)