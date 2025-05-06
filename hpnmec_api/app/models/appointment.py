from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.types import Enum
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import AppointmentStatusEnum

class Appointment(Base, TimestampMixin):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctor_profiles.id", ondelete="CASCADE"), nullable=False)
    appointment_datetime = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, default=30, nullable=False)
    status = Column(Enum(AppointmentStatusEnum), default=AppointmentStatusEnum.PENDING, nullable=False)
    reason = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    is_virtual = Column(Boolean, default=False)
    meeting_link = Column(String(255), nullable=True)  # For virtual appointments
    
    # Relationships
    patient = relationship("User", foreign_keys=[patient_id], back_populates="patient_appointments")
    doctor = relationship("DoctorProfile", foreign_keys=[doctor_id], back_populates="doctor_appointments")
    health_record = relationship("HealthRecord", back_populates="appointment", uselist=False)