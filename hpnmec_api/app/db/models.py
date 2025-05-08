import enum
from sqlalchemy import Column, Integer, String, Boolean, Date, ForeignKey, Float
from sqlalchemy.types import Enum as SQLAlchemyEnum
from sqlalchemy.orm import relationship
from datetime import datetime

# Enums
class GenderEnum(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class UserRoleEnum(enum.Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"

class AppointmentStatusEnum(enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class BloodTypeEnum(enum.Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

# Base and TimestampMixin
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class TimestampMixin:
    created_at = Column(Date, default=datetime.utcnow)
    updated_at = Column(Date, default=datetime.utcnow, onupdate=datetime.utcnow)

# User model
class User(Base, TimestampMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    phone_number = Column(String(20), nullable=True)
    hashed_password = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(SQLAlchemyEnum(GenderEnum), nullable=True)
    role = Column(SQLAlchemyEnum(UserRoleEnum), default=UserRoleEnum.PATIENT, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    address = Column(String(255), nullable=True)
    profile_picture = Column(String(255), nullable=True)
    doctor_profile = relationship("DoctorProfile", back_populates="user", uselist=False)
    health_records = relationship("HealthRecord", back_populates="user")
    patient_appointments = relationship("Appointment", foreign_keys="[Appointment.patient_id]", back_populates="patient")
    bookmarked_doctors = relationship("UserDoctorBookmark", back_populates="user")

# DoctorProfile model
class DoctorProfile(Base, TimestampMixin):
    __tablename__ = "doctor_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    specialization = Column(String(100), nullable=False)
    qualification = Column(String(255), nullable=False)
    experience_years = Column(Integer, nullable=False)
    license_number = Column(String(50), nullable=False)
    hospital_affiliation = Column(String(255), nullable=True)
    consultation_fee = Column(Float, nullable=True)
    bio = Column(String, nullable=True)
    consultation_hours_start = Column(String(5), nullable=True)
    consultation_hours_end = Column(String(5), nullable=True)
    is_available = Column(Boolean, default=True)
    user = relationship("User", back_populates="doctor_profile")

# Appointment model
class Appointment(Base, TimestampMixin):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    appointment_datetime = Column(Date, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    reason = Column(String, nullable=True)
    is_virtual = Column(Boolean, default=False)
    status = Column(SQLAlchemyEnum(AppointmentStatusEnum), default=AppointmentStatusEnum.PENDING, nullable=False)
    notes = Column(String, nullable=True)
    meeting_link = Column(String(255), nullable=True)
    patient = relationship("User", foreign_keys=[patient_id], back_populates="patient_appointments")
    doctor = relationship("User", foreign_keys=[doctor_id])

# HealthRecord model
class HealthRecord(Base, TimestampMixin):
    __tablename__ = "health_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    record_date = Column(Date, nullable=False)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    blood_pressure_systolic = Column(Integer, nullable=True)
    blood_pressure_diastolic = Column(Integer, nullable=True)
    heart_rate = Column(Integer, nullable=True)
    temperature = Column(Float, nullable=True)
    blood_oxygen = Column(Float, nullable=True)
    blood_glucose = Column(Float, nullable=True)
    blood_type = Column(SQLAlchemyEnum(BloodTypeEnum), nullable=True)
    allergies = Column(String, nullable=True)
    chronic_diseases = Column(String, nullable=True)
    current_medications = Column(String, nullable=True)
    family_medical_history = Column(String, nullable=True)
    symptoms = Column(String, nullable=True)
    diagnosis = Column(String, nullable=True)
    treatment_plan = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    lab_results = Column(String, nullable=True)
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True)
    anomaly_score = Column(Float, nullable=True)
    risk_factors = Column(String, nullable=True)
    health_insights = Column(String, nullable=True)
    predictive_indicators = Column(String, nullable=True)
    user = relationship("User", back_populates="health_records")

# UserDoctorBookmark model
class UserDoctorBookmark(Base, TimestampMixin):
    __tablename__ = "user_doctor_bookmarks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="bookmarked_doctors")
