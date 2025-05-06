import api from './api';
// Import withRetry from errorUtils
import { isRetriableError, withRetry } from '../utils/errorUtils'; 
import { getCacheItem, setCacheItem, createCacheKey, invalidateCacheItem } from '../utils/cacheUtils';

/**
 * @typedef {import('../types').AppointmentStatus} AppointmentStatus
 * @typedef {import('../types').Appointment} Appointment
 * @typedef {import('../types').AppointmentCreate} AppointmentCreate
 * @typedef {import('../types').AppointmentUpdate} AppointmentUpdate
 */

// Base API endpoint for appointments
const APPOINTMENTS_ENDPOINT = '/appointments';

// Configuration for cache TTL (time-to-live)
const CACHE_TTL = {
  LIST: 2 * 60 * 1000, // 2 minutes for appointment lists (shorter since they change frequently)
  DETAIL: 5 * 60 * 1000 // 5 minutes for appointment details
};

const appointmentService = {
  /**
   * Retrieves all appointments
   * @param {Object} [params] - Query parameters
   * @param {boolean} [useCache=true] - Whether to use cached data if available
   * @returns {Promise<Array<Appointment>>} List of appointments
   */
  getAllAppointments: async (params = {}, useCache = true) => {
    const cacheKey = createCacheKey(APPOINTMENTS_ENDPOINT, params);
    
    if (useCache) {
      const cachedData = getCacheItem(cacheKey);
      if (cachedData) return cachedData;
    }
    
    const response = await withRetry(() => api.get(APPOINTMENTS_ENDPOINT, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves an appointment by ID
   * @param {number} id - Appointment ID
   * @param {boolean} [useCache=true] - Whether to use cached data if available
   * @returns {Promise<Appointment>} Appointment details
   */
  getAppointmentById: async (id, useCache = true) => {
    const cacheKey = createCacheKey(`${APPOINTMENTS_ENDPOINT}/${id}`);
    
    if (useCache) {
      const cachedData = getCacheItem(cacheKey);
      if (cachedData) return cachedData;
    }
    
    const response = await withRetry(() => api.get(`${APPOINTMENTS_ENDPOINT}/${id}`));
    setCacheItem(cacheKey, response.data, CACHE_TTL.DETAIL);
    return response.data;
  },
  
  /**
   * Creates a new appointment
   * @param {AppointmentCreate} appointmentData - Appointment data
   * @returns {Promise<Appointment>} Created appointment
   */
  createAppointment: async (appointmentData) => {
    const response = await withRetry(() => api.post(APPOINTMENTS_ENDPOINT, appointmentData));
    
    // Invalidate list cache since we've added a new appointment
    invalidateCacheItem(createCacheKey(APPOINTMENTS_ENDPOINT));
    
    return response.data;
  },
  
  /**
   * Updates an appointment
   * @param {number} id - Appointment ID
   * @param {AppointmentUpdate} appointmentData - Updated appointment data
   * @returns {Promise<Appointment>} Updated appointment
   */
  updateAppointment: async (id, appointmentData) => {
    const response = await withRetry(() => 
      api.put(`${APPOINTMENTS_ENDPOINT}/${id}`, appointmentData)
    );
    
    // Invalidate caches for this appointment
    invalidateCacheItem(createCacheKey(`${APPOINTMENTS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(APPOINTMENTS_ENDPOINT));
    
    return response.data;
  },
  
  /**
   * Deletes an appointment
   * @param {number} id - Appointment ID
   * @returns {Promise<void>}
   */
  deleteAppointment: async (id) => {
    await withRetry(() => api.delete(`${APPOINTMENTS_ENDPOINT}/${id}`));
    
    // Invalidate caches for this appointment
    invalidateCacheItem(createCacheKey(`${APPOINTMENTS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(APPOINTMENTS_ENDPOINT));
  },
  
  /**
   * Retrieves appointments for a patient
   * @param {number} patientId - Patient ID
   * @param {Object} [params] - Query parameters
   * @returns {Promise<Array<Appointment>>} List of patient's appointments
   */
  getPatientAppointments: async (patientId, params = {}) => {
    const endpoint = `${APPOINTMENTS_ENDPOINT}/patient/${patientId}`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves appointments for a doctor
   * @param {number} doctorId - Doctor ID
   * @param {Object} [params] - Query parameters
   * @returns {Promise<Array<Appointment>>} List of doctor's appointments
   */
  getDoctorAppointments: async (doctorId, params = {}) => {
    const endpoint = `${APPOINTMENTS_ENDPOINT}/doctor/${doctorId}`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves upcoming appointments for a patient
   * @param {number} patientId - Patient ID
   * @param {Object} [params] - Query parameters
   * @returns {Promise<Array<Appointment>>} List of upcoming appointments
   */
  getUpcomingPatientAppointments: async (patientId, params = {}) => {
    const endpoint = `${APPOINTMENTS_ENDPOINT}/patient/${patientId}/upcoming`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves upcoming appointments for a doctor
   * @param {number} doctorId - Doctor ID
   * @param {Object} [params] - Query parameters
   * @returns {Promise<Array<Appointment>>} List of upcoming appointments
   */
  getUpcomingDoctorAppointments: async (doctorId, params = {}) => {
    const endpoint = `${APPOINTMENTS_ENDPOINT}/doctor/${doctorId}/upcoming`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Updates the status of an appointment
   * @param {number} id - Appointment ID
   * @param {AppointmentStatus} status - New status (pending, confirmed, completed, cancelled)
   * @returns {Promise<Appointment>} Updated appointment
   */
  updateAppointmentStatus: async (id, status) => {
    const response = await withRetry(() => 
      api.put(`${APPOINTMENTS_ENDPOINT}/${id}/status`, { status })
    );
    
    // Invalidate caches for this appointment
    invalidateCacheItem(createCacheKey(`${APPOINTMENTS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(APPOINTMENTS_ENDPOINT));
    
    return response.data;
  },
  
  /**
   * Checks if there's a conflict with an existing appointment
   * @param {Object} appointmentData - Appointment data to check
   * @returns {Promise<boolean>} Whether there's a conflict
   */
  checkAppointmentConflict: async (appointmentData) => {
    const response = await withRetry(() => 
      api.post(`${APPOINTMENTS_ENDPOINT}/check-conflict`, appointmentData)
    );
    return response.data;
  }
};

export default appointmentService;
