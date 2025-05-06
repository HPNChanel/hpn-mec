from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin

class UserDoctorBookmark(Base, TimestampMixin):
    __tablename__ = "user_doctor_bookmarks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctor_profiles.id", ondelete="CASCADE"), nullable=False)
    
    # Ensure a user can bookmark a doctor only once
    __table_args__ = (
        UniqueConstraint('user_id', 'doctor_id', name='uix_user_doctor_bookmark'),
    )
    
    # Relationships
    user = relationship("User", back_populates="bookmarked_doctors")
    doctor = relationship("DoctorProfile", back_populates="bookmarked_by")