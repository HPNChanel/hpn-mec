import { BloodType } from './enums.types';

// Base health record interface
export interface IHealthRecord {
  id: number;
  user_id: number;
  record_date: string; // ISO format YYYY-MM-DD
  height?: number;
  weight?: number;
  bmi?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  heart_rate?: number;
  temperature?: number;
  blood_oxygen?: number;
  blood_glucose?: number;
  blood_type?: BloodType;
  allergies?: string;
  chronic_diseases?: string;
  current_medications?: string;
  family_medical_history?: string;
  symptoms?: string;
  diagnosis?: string;
  treatment_plan?: string;
  lab_results?: Record<string, any>;
  notes?: string;
  appointment_id?: number;
  anomaly_score?: number;
  risk_factors?: Record<string, any>;
  health_insights?: Record<string, any>;
  predictive_indicators?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

// For creating health records
export interface IHealthRecordCreate {
  user_id: number;
  record_date: string; // ISO format YYYY-MM-DD
  height?: number;
  weight?: number;
  bmi?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  heart_rate?: number;
  temperature?: number;
  blood_oxygen?: number;
  blood_glucose?: number;
  blood_type?: BloodType;
  allergies?: string;
  chronic_diseases?: string;
  current_medications?: string;
  family_medical_history?: string;
  symptoms?: string;
  diagnosis?: string;
  treatment_plan?: string;
  lab_results?: Record<string, any>;
  notes?: string;
  appointment_id?: number;
}

// For updating health records
export interface IHealthRecordUpdate {
  record_date?: string;
  height?: number;
  weight?: number;
  bmi?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  heart_rate?: number;
  temperature?: number;
  blood_oxygen?: number;
  blood_glucose?: number;
  blood_type?: BloodType;
  allergies?: string;
  chronic_diseases?: string;
  current_medications?: string;
  family_medical_history?: string;
  symptoms?: string;
  diagnosis?: string;
  treatment_plan?: string;
  lab_results?: Record<string, any>;
  notes?: string;
  appointment_id?: number;
}
