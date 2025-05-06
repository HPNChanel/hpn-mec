from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import date

from app.crud.base import CRUDBase
from app.models.health_record import HealthRecord
from app.schemas.health_record import HealthRecordCreate, HealthRecordUpdate

class CRUDHealthRecord(CRUDBase[HealthRecord, HealthRecordCreate, HealthRecordUpdate]):
    def get_by_user_id(self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100) -> List[HealthRecord]:
        return db.query(HealthRecord).filter(HealthRecord.user_id == user_id).order_by(HealthRecord.record_date.desc()).offset(skip).limit(limit).all()
    
    def get_by_user_date(self, db: Session, *, user_id: int, record_date: date) -> Optional[HealthRecord]:
        return db.query(HealthRecord).filter(
            HealthRecord.user_id == user_id,
            HealthRecord.record_date == record_date
        ).first()
    
    def get_by_appointment_id(self, db: Session, *, appointment_id: int) -> Optional[HealthRecord]:
        return db.query(HealthRecord).filter(HealthRecord.appointment_id == appointment_id).first()
    
    def update_ai_analysis(
        self, 
        db: Session, 
        *, 
        record_id: int, 
        anomaly_score: float = None,
        risk_factors: Dict[str, Any] = None,
        health_insights: Dict[str, Any] = None,
        predictive_indicators: Dict[str, Any] = None
    ) -> HealthRecord:
        record = self.get(db, id=record_id)
        if record:
            if anomaly_score is not None:
                record.anomaly_score = anomaly_score
            if risk_factors is not None:
                record.risk_factors = risk_factors
            if health_insights is not None:
                record.health_insights = health_insights
            if predictive_indicators is not None:
                record.predictive_indicators = predictive_indicators
                
            db.add(record)
            db.commit()
            db.refresh(record)
        return record
    
    def get_latest_by_user(self, db: Session, *, user_id: int) -> Optional[HealthRecord]:
        return db.query(HealthRecord).filter(
            HealthRecord.user_id == user_id
        ).order_by(HealthRecord.record_date.desc()).first()


health_record = CRUDHealthRecord(HealthRecord)