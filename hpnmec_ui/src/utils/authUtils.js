import { UserRole } from '../types';

/**
 * Check if the user has permission based on role
 * @param {object} user - The user object
 * @param {Array<UserRole>} allowedRoles - Array of roles that have permission
 * @returns {boolean} True if user has permission, false otherwise
 */
export const hasPermission = (user, allowedRoles) => {
  if (!user) return false;
  return allowedRoles.includes(user.role);
};

/**
 * Check if the user is an admin
 * @param {object} user - The user object
 * @returns {boolean} True if user is admin, false otherwise
 */
export const isAdmin = (user) => {
  if (!user) return false;
  return user.role === UserRole.ADMIN;
};

/**
 * Check if the user is a doctor
 * @param {object} user - The user object
 * @returns {boolean} True if user is doctor, false otherwise
 */
export const isDoctor = (user) => {
  if (!user) return false;
  return user.role === UserRole.DOCTOR;
};

/**
 * Check if the user is a patient
 * @param {object} user - The user object
 * @returns {boolean} True if user is patient, false otherwise
 */
export const isPatient = (user) => {
  if (!user) return false;
  return user.role === UserRole.PATIENT;
};
