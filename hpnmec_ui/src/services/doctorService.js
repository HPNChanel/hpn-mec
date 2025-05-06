import api from './api';
// Import withRetry from errorUtils
import { isRetriableError, withRetry } from '../utils/errorUtils';
import { getCacheItem, setCacheItem, createCacheKey, invalidateCacheItem } from '../utils/cacheUtils';

/**
 * @typedef {import('../types/doctor.types').IDoctor} IDoctor
 * @typedef {import('../types/doctor.types').IDoctorCreate} IDoctorCreate
 * @typedef {import('../types/doctor.types').IDoctorUpdate} IDoctorUpdate
 */

// Configuration for cache TTL (time-to-live)
const CACHE_TTL = {
  LIST: 5 * 60 * 1000, // 5 minutes for doctor lists
  DETAIL: 10 * 60 * 1000 // 10 minutes for doctor details
};

// Base API endpoint for doctors
const DOCTORS_ENDPOINT = '/doctors';

const doctorService = {
  /**
   * Retrieves all doctors
   * @param {Object} [params] - Query parameters
   * @param {number} [params.skip=0] - Number of records to skip
   * @param {number} [params.limit=100] - Maximum number of records to return
   * @param {boolean} [useCache=true] - Whether to use cached data if available
   * @returns {Promise<Array<IDoctor>>} List of doctors
   */
  getAllDoctors: async (params = { skip: 0, limit: 100 }, useCache = true) => {
    const cacheKey = createCacheKey(DOCTORS_ENDPOINT, params);
    
    if (useCache) {
      const cachedData = getCacheItem(cacheKey);
      if (cachedData) return cachedData;
    }
    
    const response = await withRetry(() => api.get(DOCTORS_ENDPOINT, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves a doctor by ID
   * @param {number} id - Doctor ID
   * @param {boolean} [useCache=true] - Whether to use cached data if available
   * @returns {Promise<IDoctor>} Doctor details
   */
  getDoctorById: async (id, useCache = true) => {
    const cacheKey = createCacheKey(`${DOCTORS_ENDPOINT}/${id}`);
    
    if (useCache) {
      const cachedData = getCacheItem(cacheKey);
      if (cachedData) return cachedData;
    }
    
    const response = await withRetry(() => api.get(`${DOCTORS_ENDPOINT}/${id}`));
    setCacheItem(cacheKey, response.data, CACHE_TTL.DETAIL);
    return response.data;
  },
  
  /**
   * Creates a new doctor profile
   * @param {IDoctorCreate} doctorData - Doctor data
   * @returns {Promise<IDoctor>} Created doctor
   */
  createDoctor: async (doctorData) => {
    const response = await withRetry(() => api.post(DOCTORS_ENDPOINT, doctorData));
    // Invalidate list cache since we've added a new doctor
    invalidateCacheItem(createCacheKey(DOCTORS_ENDPOINT));
    return response.data;
  },
  
  /**
   * Updates a doctor's information
   * @param {number} id - Doctor ID
   * @param {IDoctorUpdate} doctorData - Updated doctor data
   * @returns {Promise<IDoctor>} Updated doctor
   */
  updateDoctor: async (id, doctorData) => {
    const response = await withRetry(() => api.put(`${DOCTORS_ENDPOINT}/${id}`, doctorData));
    
    // Invalidate caches for this doctor
    invalidateCacheItem(createCacheKey(`${DOCTORS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(DOCTORS_ENDPOINT));
    
    return response.data;
  },
  
  /**
   * Deletes a doctor
   * @param {number} id - Doctor ID
   * @returns {Promise<void>}
   */
  deleteDoctor: async (id) => {
    await withRetry(() => api.delete(`${DOCTORS_ENDPOINT}/${id}`));
    
    // Invalidate caches for this doctor
    invalidateCacheItem(createCacheKey(`${DOCTORS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(DOCTORS_ENDPOINT));
  },
  
  /**
   * Retrieves a doctor by user ID
   * @param {number} userId - User ID
   * @returns {Promise<IDoctor>} Doctor details
   */
  getDoctorByUserId: async (userId) => {
    const response = await withRetry(() => 
      api.get(`${DOCTORS_ENDPOINT}/by-user/${userId}`)
    );
    return response.data;
  },
  
  /**
   * Retrieves doctors by specialization
   * @param {string} specialization - Medical specialization
   * @param {Object} [params] - Query parameters
   * @param {number} [params.skip=0] - Number of records to skip
   * @param {number} [params.limit=100] - Maximum number of records to return
   * @returns {Promise<Array<IDoctor>>} List of doctors with the specified specialization
   */
  getDoctorsBySpecialization: async (specialization, params = { skip: 0, limit: 100 }) => {
    const cacheKey = createCacheKey(`${DOCTORS_ENDPOINT}/specialization/${specialization}`, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => 
      api.get(`${DOCTORS_ENDPOINT}/specialization/${specialization}`, { params })
    );
    
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves all available doctors
   * @param {Object} [params] - Query parameters
   * @param {number} [params.skip=0] - Number of records to skip
   * @param {number} [params.limit=100] - Maximum number of records to return
   * @returns {Promise<Array<IDoctor>>} List of available doctors
   */
  getAvailableDoctors: async (params = { skip: 0, limit: 100 }) => {
    const cacheKey = createCacheKey(`${DOCTORS_ENDPOINT}/available`, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => 
      api.get(`${DOCTORS_ENDPOINT}/available/`, { params })
    );
    
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Rates a doctor
   * @param {number} doctorId - Doctor ID
   * @param {number} rating - Rating value (0-5)
   * @returns {Promise<IDoctor>} Updated doctor with new rating
   */
  rateDoctor: async (doctorId, rating) => {
    if (rating < 0 || rating > 5) {
      throw new Error('Rating must be between 0 and 5');
    }
    
    const response = await withRetry(() => 
      api.post(`${DOCTORS_ENDPOINT}/${doctorId}/rating/${rating}`)
    );
    
    // Invalidate caches for this doctor
    invalidateCacheItem(createCacheKey(`${DOCTORS_ENDPOINT}/${doctorId}`));
    invalidateCacheItem(createCacheKey(DOCTORS_ENDPOINT));
    
    return response.data;
  }
};

export default doctorService;
