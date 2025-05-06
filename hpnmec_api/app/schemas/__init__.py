# User schemas
from app.schemas.user import (
    UserBase, UserCreate, UserUpdate, 
    UserPasswordUpdate, UserResponse, UserLogin  # Added UserLogin
)

# Doctor profile schemas
from app.schemas.doctor_profile import (
    DoctorProfileBase, DoctorProfileCreate, 
    DoctorProfileUpdate, DoctorProfileResponse
)

# Health record schemas
from app.schemas.health_record import (
    HealthRecordBase, HealthRecordCreate, 
    HealthRecordUpdate, HealthRecordResponse
)

# Appointment schemas
from app.schemas.appointment import (
    AppointmentBase, AppointmentCreate, 
    AppointmentUpdate, AppointmentResponse
)

# Bookmark schemas
from app.schemas.user_doctor_bookmark import (
    UserDoctorBookmarkBase, UserDoctorBookmarkCreate, 
    UserDoctorBookmarkUpdate, UserDoctorBookmarkResponse
)