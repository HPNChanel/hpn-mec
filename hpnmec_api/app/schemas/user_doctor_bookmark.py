from pydantic import BaseModel, ConfigDict
from app.schemas.base import TimestampSchema

# Base Bookmark Schema
class UserDoctorBookmarkBase(BaseModel):
    user_id: int
    doctor_id: int

# Creation Schema
class UserDoctorBookmarkCreate(UserDoctorBookmarkBase):
    pass

# Update Schema
class UserDoctorBookmarkUpdate(BaseModel):
    pass

# Response Schema
class UserDoctorBookmarkResponse(UserDoctorBookmarkBase, TimestampSchema):
    id: int
    
    model_config = ConfigDict(from_attributes=True)