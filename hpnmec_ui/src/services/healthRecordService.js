import api from './api';
import { isRetriableError, withRetry } from '../utils/errorUtils';
import { 
  getCacheItem, 
  setCacheItem, 
  createCacheKey, 
  invalidateCacheItem 
} from '../utils/cacheUtils';
import { 
  prepareDataForApi, 
  processApiResponse, 
  processApiResponseArray 
} from '../utils/apiUtils';
// Removed TypeScript interface imports

/**
 * @typedef {Object} HealthRecord
 * @property {number} id - Health record ID
 * @property {number} user_id - User ID
 * @property {string} record_date - Date of the health record
 * @property {number} [height] - Height in cm
 * @property {number} [weight] - Weight in kg
 * @property {number} [blood_pressure_systolic] - Systolic blood pressure
 * @property {number} [blood_pressure_diastolic] - Diastolic blood pressure
 * @property {number} [heart_rate] - Heart rate in bpm
 * @property {number} [blood_sugar] - Blood sugar level
 * @property {number} [body_temperature] - Body temperature in Celsius
 * @property {string} [diagnosis] - Diagnosis information
 * @property {string} [medications] - Prescribed medications
 * @property {string} [allergies] - Allergy information
 * @property {string} [notes] - Additional notes
 * @property {string} created_at - Creation timestamp
 * @property {string} updated_at - Last update timestamp
 */

/**
 * @typedef {Object} HealthRecordCreate
 * @property {number} user_id - User ID
 * @property {string} record_date - Date of the health record
 * @property {number} [height] - Height in cm
 * @property {number} [weight] - Weight in kg
 * @property {number} [blood_pressure_systolic] - Systolic blood pressure
 * @property {number} [blood_pressure_diastolic] - Diastolic blood pressure
 * @property {number} [heart_rate] - Heart rate in bpm
 * @property {number} [blood_sugar] - Blood sugar level
 * @property {number} [body_temperature] - Body temperature in Celsius
 * @property {string} [diagnosis] - Diagnosis information
 * @property {string} [medications] - Prescribed medications
 * @property {string} [allergies] - Allergy information
 * @property {string} [notes] - Additional notes
 */

/**
 * @typedef {Object} HealthRecordUpdate
 * @property {string} [record_date] - Date of the health record
 * @property {number} [height] - Height in cm
 * @property {number} [weight] - Weight in kg
 * @property {number} [blood_pressure_systolic] - Systolic blood pressure
 * @property {number} [blood_pressure_diastolic] - Diastolic blood pressure
 * @property {number} [heart_rate] - Heart rate in bpm
 * @property {number} [blood_sugar] - Blood sugar level
 * @property {number} [body_temperature] - Body temperature in Celsius
 * @property {string} [diagnosis] - Diagnosis information
 * @property {string} [medications] - Prescribed medications
 * @property {string} [allergies] - Allergy information
 * @property {string} [notes] - Additional notes
 */

// Base API endpoint for health records
const HEALTH_RECORDS_ENDPOINT = '/health-records';

// Configuration for cache TTL (time-to-live)
const CACHE_TTL = {
  LIST: 5 * 60 * 1000, // 5 minutes for health record lists
  DETAIL: 10 * 60 * 1000 // 10 minutes for health record details
};

// Date field mapping for transformations
const DATE_FIELDS = ['record_date'];
const DATETIME_FIELDS = ['created_at', 'updated_at'];

const healthRecordService = {
  /**
   * Retrieves all health records
   * @param {Object} [params] - Query parameters
   * @param {boolean} [useCache=true] - Whether to use cached data if available
   * @returns {Promise<Array<HealthRecord>>} List of health records
   */
  getAllHealthRecords: async (params = {}, useCache = true) => {
    const cacheKey = createCacheKey(HEALTH_RECORDS_ENDPOINT, params);
    
    if (useCache) {
      const cachedData = getCacheItem(cacheKey);
      if (cachedData) return cachedData;
    }
    
    const response = await withRetry(() => api.get(HEALTH_RECORDS_ENDPOINT, { params }));
    const processedData = processApiResponseArray(response.data, DATE_FIELDS, DATETIME_FIELDS);
    setCacheItem(cacheKey, processedData, CACHE_TTL.LIST);
    return processedData;
  },
  
  /**
   * Retrieves a health record by ID
   * @param {number} id - Health record ID
   * @param {boolean} [useCache=true] - Whether to use cached data if available
   * @returns {Promise<HealthRecord>} Health record details
   */
  getHealthRecordById: async (id, useCache = true) => {
    const cacheKey = createCacheKey(`${HEALTH_RECORDS_ENDPOINT}/${id}`);
    
    if (useCache) {
      const cachedData = getCacheItem(cacheKey);
      if (cachedData) return cachedData;
    }
    
    const response = await withRetry(() => api.get(`${HEALTH_RECORDS_ENDPOINT}/${id}`));
    const processedData = processApiResponse(response.data, DATE_FIELDS, DATETIME_FIELDS);
    setCacheItem(cacheKey, processedData, CACHE_TTL.DETAIL);
    return processedData;
  },
  
  /**
   * Creates a new health record
   * @param {HealthRecordCreate} healthRecordData - Health record data
   * @returns {Promise<HealthRecord>} Created health record
   */
  createHealthRecord: async (healthRecordData) => {
    const preparedData = prepareDataForApi(healthRecordData, DATE_FIELDS, []);
    const response = await withRetry(() => api.post(HEALTH_RECORDS_ENDPOINT, preparedData));
    
    // Invalidate list cache since we've added a new health record
    invalidateCacheItem(createCacheKey(HEALTH_RECORDS_ENDPOINT));
    
    return processApiResponse(response.data, DATE_FIELDS, DATETIME_FIELDS);
  },
  
  /**
   * Updates a health record
   * @param {number} id - Health record ID
   * @param {HealthRecordUpdate} healthRecordData - Updated health record data
   * @returns {Promise<HealthRecord>} Updated health record
   */
  updateHealthRecord: async (id, healthRecordData) => {
    const preparedData = prepareDataForApi(healthRecordData, DATE_FIELDS, []);
    const response = await withRetry(() => 
      api.put(`${HEALTH_RECORDS_ENDPOINT}/${id}`, preparedData)
    );
    
    // Invalidate caches for this health record
    invalidateCacheItem(createCacheKey(`${HEALTH_RECORDS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(HEALTH_RECORDS_ENDPOINT));
    
    return processApiResponse(response.data, DATE_FIELDS, DATETIME_FIELDS);
  },
  
  /**
   * Deletes a health record
   * @param {number} id - Health record ID
   * @returns {Promise<void>}
   */
  deleteHealthRecord: async (id) => {
    await withRetry(() => api.delete(`${HEALTH_RECORDS_ENDPOINT}/${id}`));
    
    // Invalidate caches for this health record
    invalidateCacheItem(createCacheKey(`${HEALTH_RECORDS_ENDPOINT}/${id}`));
    invalidateCacheItem(createCacheKey(HEALTH_RECORDS_ENDPOINT));
  },
  
  /**
   * Retrieves health records for a user
   * @param {number} userId - User ID
   * @param {Object} [params] - Query parameters
   * @returns {Promise<Array<HealthRecord>>} List of user's health records
   */
  getUserHealthRecords: async (userId, params = {}) => {
    const endpoint = `${HEALTH_RECORDS_ENDPOINT}/user/${userId}`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves the latest health record for a user
   * @param {number} userId - User ID
   * @returns {Promise<HealthRecord>} Latest health record
   */
  getLatestUserHealthRecord: async (userId) => {
    const endpoint = `${HEALTH_RECORDS_ENDPOINT}/user/${userId}/latest`;
    const cacheKey = createCacheKey(endpoint);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint));
    setCacheItem(cacheKey, response.data, CACHE_TTL.DETAIL);
    return response.data;
  },
  
  /**
   * Retrieves health records for a date range
   * @param {number} userId - User ID
   * @param {string} startDate - Start date (YYYY-MM-DD)
   * @param {string} endDate - End date (YYYY-MM-DD)
   * @returns {Promise<Array<HealthRecord>>} List of health records in the date range
   */
  getHealthRecordsByDateRange: async (userId, startDate, endDate) => {
    const params = { start_date: startDate, end_date: endDate };
    const endpoint = `${HEALTH_RECORDS_ENDPOINT}/user/${userId}/date-range`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  },
  
  /**
   * Retrieves a summary of health metrics over time
   * @param {number} userId - User ID
   * @param {string} metric - Metric type (weight, blood_pressure, heart_rate, etc.)
   * @param {Object} [params] - Query parameters
   * @returns {Promise<Object>} Health metric summary data
   */
  getHealthMetricSummary: async (userId, metric, params = {}) => {
    const endpoint = `${HEALTH_RECORDS_ENDPOINT}/user/${userId}/metrics/${metric}`;
    const cacheKey = createCacheKey(endpoint, params);
    const cachedData = getCacheItem(cacheKey);
    
    if (cachedData) return cachedData;
    
    const response = await withRetry(() => api.get(endpoint, { params }));
    setCacheItem(cacheKey, response.data, CACHE_TTL.LIST);
    return response.data;
  }
};

export default healthRecordService;
