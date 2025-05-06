/**
 * Utilities for date/time handling between frontend and backend
 */

/**
 * Formats a date object to ISO date string (YYYY-MM-DD)
 * @param date JavaScript Date object
 * @returns ISO date string
 */
export const formatDateToISO = (date: Date): string => {
  return date.toISOString().split('T')[0];
};

/**
 * Formats a date object to ISO datetime string
 * @param date JavaScript Date object
 * @returns ISO datetime string
 */
export const formatDateTimeToISO = (date: Date): string => {
  return date.toISOString();
};

/**
 * Parses an ISO date string to JavaScript Date
 * @param isoDate ISO date string (YYYY-MM-DD)
 * @returns JavaScript Date object
 */
export const parseISODate = (isoDate: string): Date => {
  return new Date(isoDate);
};

/**
 * Parses an ISO datetime string to JavaScript Date
 * @param isoDateTime ISO datetime string
 * @returns JavaScript Date object
 */
export const parseISODateTime = (isoDateTime: string): Date => {
  return new Date(isoDateTime);
};

/**
 * Formats a date for display in Vietnamese format
 * @param date JavaScript Date object or ISO date string
 * @returns Formatted date string (DD/MM/YYYY)
 */
export const formatDisplayDate = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleDateString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
};

/**
 * Formats a datetime for display in Vietnamese format
 * @param date JavaScript Date object or ISO datetime string
 * @returns Formatted datetime string (DD/MM/YYYY HH:MM)
 */
export const formatDisplayDateTime = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};
