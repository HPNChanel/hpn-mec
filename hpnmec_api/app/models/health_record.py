from sqlalchemy import Column, Integer, Float, String, Text, Date, DateTime, ForeignKey, JSON
from sqlalchemy.types import Enum
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import BloodTypeEnum

class HealthRecord(Base, TimestampMixin):
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    record_date = Column(Date, nullable=False)
    
    # Basic health vitals
    height = Column(Float, nullable=True)  # in cm
    weight = Column(Float, nullable=True)  # in kg
    bmi = Column(Float, nullable=True)
    blood_pressure_systolic = Column(Integer, nullable=True)  # in mmHg
    blood_pressure_diastolic = Column(Integer, nullable=True)  # in mmHg
    heart_rate = Column(Integer, nullable=True)  # in bpm
    temperature = Column(Float, nullable=True)  # in Celsius
    blood_oxygen = Column(Float, nullable=True)  # SpO2 in percentage
    blood_glucose = Column(Float, nullable=True)  # in mg/dL
    blood_type = Column(Enum(BloodTypeEnum), nullable=True)
    
    # Medical history
    allergies = Column(Text, nullable=True)
    chronic_diseases = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    family_medical_history = Column(Text, nullable=True)
    
    # Additional data
    symptoms = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=True)
    treatment_plan = Column(Text, nullable=True)
    lab_results = Column(JSON, nullable=True)  # Store complex lab results as JSON
    notes = Column(Text, nullable=True)
    
    # AI analysis results
    anomaly_score = Column(Float, nullable=True)  # For anomaly detection
    risk_factors = Column(JSON, nullable=True)  # Store risk factors identified by AI
    health_insights = Column(JSON, nullable=True)  # Store AI-generated insights
    predictive_indicators = Column(JSON, nullable=True)  # Store predictive health indicators
    
    # Related appointment (if any)
    appointment_id = Column(Integer, ForeignKey("appointments.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="health_records")
    appointment = relationship("Appointment", back_populates="health_record")