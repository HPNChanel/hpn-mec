/**
 * Utility functions for date/time formatting and manipulation
 * This file re-exports functionality from dateUtils.ts and adds JS-specific methods
 */

// Import core functionality from TypeScript version
import {
  formatDateToISO,
  formatDateTimeToISO,
  parseISODate,
  parseISODateTime,
  formatDisplayDate,
  formatDisplayDateTime
} from './dateUtils.ts';

// Re-export the TypeScript functions
export {
  formatDateToISO,
  formatDateTimeToISO,
  parseISODate,
  parseISODateTime,
  formatDisplayDate,
  formatDisplayDateTime
};

/**
 * Formats a date string or Date object to a localized date string
 * @param {string|Date} date - Date to format
 * @param {string} [format='short'] - Format option ('short', 'full', 'numeric')
 * @returns {string} Formatted date string
 */
export const formatDate = (date, format = 'short') => {
  if (!date) return '';
  
  const dateObj = date instanceof Date ? date : new Date(date);
  
  // Check if date is valid
  if (isNaN(dateObj.getTime())) return 'Ngày không hợp lệ';
  
  const options = { 
    short: { day: '2-digit', month: '2-digit', year: 'numeric' },
    full: { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' },
    numeric: { day: 'numeric', month: 'numeric', year: 'numeric' }
  };
  
  return dateObj.toLocaleDateString('vi-VN', options[format] || options.short);
};

/**
 * Formats a date string or Date object to a localized time string
 * @param {string|Date} date - Date to format
 * @param {boolean} [includeSeconds=false] - Whether to include seconds
 * @returns {string} Formatted time string
 */
export const formatTime = (date, includeSeconds = false) => {
  if (!date) return '';
  
  const dateObj = date instanceof Date ? date : new Date(date);
  
  // Check if date is valid
  if (isNaN(dateObj.getTime())) return 'Thời gian không hợp lệ';
  
  const options = { 
    hour: '2-digit', 
    minute: '2-digit',
    ...(includeSeconds ? { second: '2-digit' } : {})
  };
  
  return dateObj.toLocaleTimeString('vi-VN', options);
};

/**
 * Formats a date string or Date object to a full datetime string
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted datetime string
 */
export const formatDateTime = (date) => {
  if (!date) return '';
  
  const dateObj = date instanceof Date ? date : new Date(date);
  
  // Check if date is valid
  if (isNaN(dateObj.getTime())) return 'Ngày giờ không hợp lệ';
  
  return `${formatDate(dateObj)} ${formatTime(dateObj)}`;
};

/**
 * Converts a time string in HH:MM format to a display format
 * @param {string} timeString - Time string in HH:MM format
 * @returns {string} Formatted time string
 */
export const formatTimeHHMM = (timeString) => {
  if (!timeString || !/^\d{2}:\d{2}$/.test(timeString)) return timeString || '';
  
  const [hours, minutes] = timeString.split(':').map(Number);
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
};

/**
 * Checks if a date is in the past
 * @param {string|Date} date - Date to check
 * @returns {boolean} Whether the date is in the past
 */
export const isPastDate = (date) => {
  if (!date) return false;
  
  const dateObj = date instanceof Date ? date : new Date(date);
  return dateObj < new Date();
};

/**
 * Formats duration in minutes to a human-readable format
 * @param {number} minutes - Duration in minutes
 * @returns {string} Formatted duration
 */
export const formatDuration = (minutes) => {
  if (!minutes || isNaN(minutes)) return '';
  
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  
  if (hours === 0) return `${mins} phút`;
  if (mins === 0) return `${hours} giờ`;
  return `${hours} giờ ${mins} phút`;
};
