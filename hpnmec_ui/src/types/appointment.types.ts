export interface Appointment {
  id: string;
  patient_id: string;
  doctor_id: string;
  date: string; // ISO format date
  time_slot: string; // HH:MM format
  duration: number; // In minutes
  status: AppointmentStatus;
  reason: string;
  notes?: string;
  attachments?: string[];
  created_at: string;
  updated_at: string;
  patient_name?: string; // For doctor's view
  doctor_name?: string; // For patient's view
}

export enum AppointmentStatus {
  SCHEDULED = 'SCHEDULED',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED',
  MISSED = 'MISSED',
  IN_PROGRESS = 'IN_PROGRESS',
  PENDING = 'PENDING'
}

export interface TimeSlot {
  id: string;
  start_time: string;
  end_time: string;
  is_available: boolean;
}
