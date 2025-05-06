from sqlalchemy import Column, Integer, String, Text, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin

class DoctorProfile(Base, TimestampMixin):
    __tablename__ = "doctor_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    specialization = Column(String(100), nullable=False)
    qualification = Column(String(255), nullable=False)
    experience_years = Column(Integer, nullable=False)
    license_number = Column(String(50), unique=True, nullable=False)
    hospital_affiliation = Column(String(255), nullable=True)
    consultation_fee = Column(Float, nullable=True)
    bio = Column(Text, nullable=True)
    consultation_hours_start = Column(String(5), nullable=True)  # Format: "HH:MM"
    consultation_hours_end = Column(String(5), nullable=True)    # Format: "HH:MM"
    is_available = Column(Boolean, default=True)
    rating = Column(Float, default=0.0)
    total_ratings = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="doctor_profile")
    doctor_appointments = relationship("Appointment", foreign_keys="[Appointment.doctor_id]", back_populates="doctor")
    bookmarked_by = relationship("UserDoctorBookmark", back_populates="doctor")