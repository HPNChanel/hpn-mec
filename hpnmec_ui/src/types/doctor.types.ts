import { ITimestamp } from './base.types';
import { UserRole } from './user.types';

/**
 * Base interface for Doctor Profile properties.
 * Matches DoctorProfileBase Pydantic schema.
 */
export interface IDoctorBase {
  specialization: string;
  qualification: string;
  experience_years: number;
  license_number: string;
  hospital_affiliation?: string | null;
  consultation_fee?: number | null;
  bio?: string | null;
  consultation_hours_start?: string | null; // "HH:MM"
  consultation_hours_end?: string | null;   // "HH:MM"
  is_available: boolean;
}

/**
 * Interface for creating a new Doctor Profile.
 * Matches DoctorProfileCreate Pydantic schema.
 */
export interface IDoctorCreate extends IDoctorBase {
  user_id: number;
}

/**
 * Interface for updating an existing Doctor Profile.
 * Matches DoctorProfileUpdate Pydantic schema.
 * All fields are optional.
 */
export interface IDoctorUpdate {
  specialization?: string | null;
  qualification?: string | null;
  experience_years?: number | null;
  hospital_affiliation?: string | null;
  consultation_fee?: number | null;
  bio?: string | null;
  consultation_hours_start?: string | null;
  consultation_hours_end?: string | null;
  is_available?: boolean | null;
}

/**
 * Interface representing a Doctor Profile as returned by the API.
 * Matches DoctorProfileResponse Pydantic schema.
 * Includes base fields, ID, user ID, rating, and timestamps.
 */
export interface IDoctor extends IDoctorBase, ITimestamp {
  id: number;
  user_id: number;
  rating: number;
  total_ratings: number;
  // Optional: Include user details if the API response includes them
  // user?: IUser; 
}

export interface Doctor {
  id: string;
  user_id: string;
  full_name: string;
  specialization: string;
  medical_license: string;
  experience_years: number;
  education: string;
  hospital: string;
  bio: string;
  profile_picture: string | null;
  availability: {
    days: string[];
    hours: {
      start: string;
      end: string;
    };
  };
  rating: number;
  review_count: number;
  created_at: string;
  updated_at: string;
  status: 'ACTIVE' | 'INACTIVE' | 'PENDING' | 'SUSPENDED';
}

export interface DoctorListItem {
  id: string;
  full_name: string;
  specialization: string;
  hospital: string;
  profile_picture: string | null;
  rating: number;
  review_count: number;
}

export interface DoctorFilterOptions {
  specialization?: string;
  hospital?: string;
  rating?: number;
  availability?: string[];
}
