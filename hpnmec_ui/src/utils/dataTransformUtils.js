/**
 * Utility functions for transforming and normalizing data
 */

/**
 * Normalizes a collection of objects by ID
 * @param {Array<Object>} items - Array of objects with an id property
 * @param {string} [idKey='id'] - The key to use as the identifier
 * @returns {Object} An object where keys are ids and values are the items
 */
export const normalizeById = (items, idKey = 'id') => {
  if (!Array.isArray(items)) return {};
  
  return items.reduce((acc, item) => {
    if (item && item[idKey] !== undefined) {
      acc[item[idKey]] = item;
    }
    return acc;
  }, {});
};

/**
 * Strips HTML tags from a string
 * @param {string} html - String potentially containing HTML
 * @returns {string} String with HTML tags removed
 */
export const stripHtml = (html) => {
  if (!html || typeof html !== 'string') return '';
  return html.replace(/<\/?[^>]+(>|$)/g, '');
};

/**
 * Truncates a string to a specified length and adds an ellipsis
 * @param {string} str - String to truncate
 * @param {number} [length=100] - Maximum length
 * @returns {string} Truncated string
 */
export const truncateText = (str, length = 100) => {
  if (!str || typeof str !== 'string') return '';
  if (str.length <= length) return str;
  return str.substring(0, length).trim() + '...';
};

/**
 * Formats a monetary value
 * @param {number} value - The monetary value
 * @param {string} [currency='VND'] - Currency code
 * @returns {string} Formatted monetary value
 */
export const formatCurrency = (value, currency = 'VND') => {
  if (value === null || value === undefined || isNaN(value)) return '';
  
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(value);
};

/**
 * Converts snake_case to camelCase
 * @param {string} str - Snake case string
 * @returns {string} Camel case string
 */
export const snakeToCamel = (str) => {
  if (!str || typeof str !== 'string') return '';
  return str.replace(/_([a-z])/g, (match, group) => group.toUpperCase());
};

/**
 * Recursively transforms keys in an object from snake_case to camelCase
 * @param {Object|Array} data - The data to transform
 * @returns {Object|Array} Transformed data
 */
export const transformKeysToCamelCase = (data) => {
  if (Array.isArray(data)) {
    return data.map(transformKeysToCamelCase);
  }
  
  if (data && typeof data === 'object' && data !== null) {
    return Object.keys(data).reduce((acc, key) => {
      const camelKey = snakeToCamel(key);
      acc[camelKey] = transformKeysToCamelCase(data[key]);
      return acc;
    }, {});
  }
  
  return data;
};
