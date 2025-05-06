import { Gender, UserRole } from './enums.types';

// Base interface for User
export interface IUser {
  id: number;
  username: string;
  email: string;
  full_name: string;
  phone_number?: string;
  date_of_birth?: string; // ISO format YYYY-MM-DD
  gender?: Gender;
  role: UserRole;
  address?: string;
  profile_picture?: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string; // ISO format
  updated_at: string; // ISO format
}

// For creating a new user
export interface IUserCreate {
  username: string;
  email: string;
  password: string;
  full_name: string;
  phone_number?: string;
  date_of_birth?: string;
  gender?: Gender;
  role?: UserRole;
  address?: string;
  profile_picture?: string;
}

// For updating user details
export interface IUserUpdate {
  username?: string;
  email?: string;
  full_name?: string;
  phone_number?: string;
  date_of_birth?: string;
  gender?: Gender;
  address?: string;
  profile_picture?: string;
  is_active?: boolean;
  is_verified?: boolean;
}

// For login
export interface IUserLogin {
  email: string;
  password: string;
}

// For password update
export interface IPasswordUpdate {
  current_password: string;
  new_password: string;
}
