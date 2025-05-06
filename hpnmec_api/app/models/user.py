from sqlalchemy import Column, Integer, String, Boolean, Date, ForeignKey
from sqlalchemy.types import Enum
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import GenderEnum, UserRoleEnum

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    phone_number = Column(String(20), nullable=True)
    hashed_password = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(Enum(GenderEnum), nullable=True)
    role = Column(Enum(UserRoleEnum), default=UserRoleEnum.PATIENT, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    address = Column(String(255), nullable=True)
    profile_picture = Column(String(255), nullable=True)
    
    # Relationships
    doctor_profile = relationship("DoctorProfile", back_populates="user", uselist=False)
    health_records = relationship("HealthRecord", back_populates="user")
    patient_appointments = relationship("Appointment", foreign_keys="[Appointment.patient_id]", back_populates="patient")
    bookmarked_doctors = relationship("UserDoctorBookmark", back_populates="user")