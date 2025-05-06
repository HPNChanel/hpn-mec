from typing import List, Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.doctor_profile import DoctorProfile
from app.schemas.doctor_profile import DoctorProfileCreate, DoctorProfileUpdate

class CRUDDoctorProfile(CRUDBase[DoctorProfile, DoctorProfileCreate, DoctorProfileUpdate]):
    def get_by_user_id(self, db: Session, *, user_id: int) -> Optional[DoctorProfile]:
        return db.query(DoctorProfile).filter(DoctorProfile.user_id == user_id).first()
    
    def get_by_license_number(self, db: Session, *, license_number: str) -> Optional[DoctorProfile]:
        return db.query(DoctorProfile).filter(DoctorProfile.license_number == license_number).first()
    
    def get_available_doctors(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[DoctorProfile]:
        return db.query(DoctorProfile).filter(DoctorProfile.is_available == True).offset(skip).limit(limit).all()
    
    def get_by_specialization(self, db: Session, *, specialization: str, skip: int = 0, limit: int = 100) -> List[DoctorProfile]:
        return db.query(DoctorProfile).filter(DoctorProfile.specialization == specialization).offset(skip).limit(limit).all()
    
    def update_rating(self, db: Session, *, doctor_id: int, new_rating: float) -> DoctorProfile:
        doctor = self.get(db, id=doctor_id)
        if doctor:
            total = doctor.total_ratings + 1
            avg_rating = ((doctor.rating * doctor.total_ratings) + new_rating) / total
            doctor.rating = avg_rating
            doctor.total_ratings = total
            db.add(doctor)
            db.commit()
            db.refresh(doctor)
        return doctor


doctor_profile = CRUDDoctorProfile(DoctorProfile)