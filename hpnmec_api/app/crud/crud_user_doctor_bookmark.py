from typing import List, Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.user_doctor_bookmark import UserDoctorBookmark
from app.schemas.user_doctor_bookmark import UserDoctorBookmarkCreate, UserDoctorBookmarkUpdate

class CRUDUserDoctorBookmark(CRUDBase[UserDoctorBookmark, UserDoctorBookmarkCreate, UserDoctorBookmarkUpdate]):
    def get_by_user_doctor(self, db: Session, *, user_id: int, doctor_id: int) -> Optional[UserDoctorBookmark]:
        return db.query(UserDoctorBookmark).filter(
            UserDoctorBookmark.user_id == user_id,
            UserDoctorBookmark.doctor_id == doctor_id
        ).first()
    
    def get_by_user(self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100) -> List[UserDoctorBookmark]:
        return db.query(UserDoctorBookmark).filter(
            UserDoctorBookmark.user_id == user_id
        ).offset(skip).limit(limit).all()
    
    def get_by_doctor(self, db: Session, *, doctor_id: int, skip: int = 0, limit: int = 100) -> List[UserDoctorBookmark]:
        return db.query(UserDoctorBookmark).filter(
            UserDoctorBookmark.doctor_id == doctor_id
        ).offset(skip).limit(limit).all()
    
    def remove_by_user_doctor(self, db: Session, *, user_id: int, doctor_id: int) -> Optional[UserDoctorBookmark]:
        bookmark = self.get_by_user_doctor(db, user_id=user_id, doctor_id=doctor_id)
        if bookmark:
            db.delete(bookmark)
            db.commit()
        return bookmark


user_doctor_bookmark = CRUDUserDoctorBookmark(UserDoctorBookmark)