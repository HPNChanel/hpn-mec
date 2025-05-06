from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.crud.base import CRUDBase
from app.models.appointment import Appointment
from app.models.enums import AppointmentStatusEnum
from app.schemas.appointment import AppointmentCreate, AppointmentUpdate

class CRUDAppointment(CRUDBase[Appointment, AppointmentCreate, AppointmentUpdate]):
    def get_by_patient(self, db: Session, *, patient_id: int, skip: int = 0, limit: int = 100) -> List[Appointment]:
        return db.query(Appointment).filter(
            Appointment.patient_id == patient_id
        ).order_by(Appointment.appointment_datetime.desc()).offset(skip).limit(limit).all()
    
    def get_by_doctor(self, db: Session, *, doctor_id: int, skip: int = 0, limit: int = 100) -> List[Appointment]:
        return db.query(Appointment).filter(
            Appointment.doctor_id == doctor_id
        ).order_by(Appointment.appointment_datetime.desc()).offset(skip).limit(limit).all()
    
    def get_upcoming_by_patient(self, db: Session, *, patient_id: int, skip: int = 0, limit: int = 100) -> List[Appointment]:
        return db.query(Appointment).filter(
            Appointment.patient_id == patient_id,
            Appointment.appointment_datetime > datetime.now(),
            Appointment.status != AppointmentStatusEnum.CANCELLED
        ).order_by(Appointment.appointment_datetime.asc()).offset(skip).limit(limit).all()
    
    def get_upcoming_by_doctor(self, db: Session, *, doctor_id: int, skip: int = 0, limit: int = 100) -> List[Appointment]:
        return db.query(Appointment).filter(
            Appointment.doctor_id == doctor_id,
            Appointment.appointment_datetime > datetime.now(),
            Appointment.status != AppointmentStatusEnum.CANCELLED
        ).order_by(Appointment.appointment_datetime.asc()).offset(skip).limit(limit).all()
    
    def update_status(self, db: Session, *, appointment_id: int, status: AppointmentStatusEnum) -> Optional[Appointment]:
        appointment = self.get(db, id=appointment_id)
        if appointment:
            appointment.status = status
            db.add(appointment)
            db.commit()
            db.refresh(appointment)
        return appointment
    
    def check_appointment_conflict(
    self, 
    db: Session, 
    *, 
    doctor_id: int, 
    start_time: datetime, 
    end_time: datetime,
    exclude_id: Optional[int] = None
) -> bool:
        """Check if there's an appointment conflict for a doctor"""
        from datetime import timedelta
        
        query = db.query(Appointment).filter(
            Appointment.doctor_id == doctor_id,
            Appointment.status != AppointmentStatusEnum.CANCELLED
        )
        
        if exclude_id:
            query = query.filter(Appointment.id != exclude_id)
        
        appointments = query.all()
        
        # For each existing appointment, calculate its end time and check overlap
        for appointment in appointments:
            existing_end_time = appointment.appointment_datetime + timedelta(minutes=appointment.duration_minutes)
            
            # Check for overlap
            if (
                (start_time <= appointment.appointment_datetime < end_time) or  # New appointment starts before existing and overlaps
                (start_time < existing_end_time <= end_time) or                # New appointment ends after existing starts but before it ends
                (appointment.appointment_datetime <= start_time < existing_end_time) or  # Existing appointment starts before new and overlaps
                (start_time <= appointment.appointment_datetime and end_time >= existing_end_time)  # New appointment completely covers existing
            ):
                return True
                
        return False


appointment = CRUDAppointment(Appointment)