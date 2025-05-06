import { 
  formatDateToISO, 
  formatDateTimeToISO, 
  parseISODate, 
  parseISODateTime 
} from './dateUtils';

/**
 * Transforms frontend data to backend format for API requests
 * @param data Frontend data
 * @param dateFields Array of fields that should be formatted as ISO date strings
 * @param dateTimeFields Array of fields that should be formatted as ISO datetime strings
 * @returns Data formatted for backend
 */
export const prepareDataForApi = <T extends Record<string, any>>(
  data: T, 
  dateFields: string[] = [], 
  dateTimeFields: string[] = []
): T => {
  const result = { ...data };

  dateFields.forEach(field => {
    if (result[field] instanceof Date) {
      result[field] = formatDateToISO(result[field]);
    }
  });

  dateTimeFields.forEach(field => {
    if (result[field] instanceof Date) {
      result[field] = formatDateTimeToISO(result[field]);
    }
  });

  return result;
};

/**
 * Transforms backend response data to frontend format
 * @param data Backend data
 * @param dateFields Array of fields that should be parsed as Date objects from ISO date strings
 * @param dateTimeFields Array of fields that should be parsed as Date objects from ISO datetime strings
 * @returns Data formatted for frontend
 */
export const processApiResponse = <T extends Record<string, any>>(
  data: T, 
  dateFields: string[] = [], 
  dateTimeFields: string[] = []
): T => {
  if (!data) return data;
  
  const result = { ...data };

  dateFields.forEach(field => {
    if (result[field] && typeof result[field] === 'string') {
      result[field] = parseISODate(result[field]);
    }
  });

  dateTimeFields.forEach(field => {
    if (result[field] && typeof result[field] === 'string') {
      result[field] = parseISODateTime(result[field]);
    }
  });

  return result;
};

/**
 * Process an array of items from the API
 * @param items Array of items from the API
 * @param dateFields Array of fields that should be parsed as Date objects from ISO date strings
 * @param dateTimeFields Array of fields that should be parsed as Date objects from ISO datetime strings
 * @returns Processed array
 */
export const processApiResponseArray = <T extends Record<string, any>>(
  items: T[], 
  dateFields: string[] = [], 
  dateTimeFields: string[] = []
): T[] => {
  if (!items) return items;
  return items.map(item => processApiResponse(item, dateFields, dateTimeFields));
};
