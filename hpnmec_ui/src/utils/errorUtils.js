/**
 * Utility for handling API errors consistently
 */

/**
 * Formats error messages for display
 * @param {Error} error - The error object
 * @returns {string} Formatted error message
 */
export const formatErrorMessage = (error) => {
  if (!error) return 'Đã xảy ra lỗi không xác định';
  
  // Handle Axios error response
  if (error.response) {
    const { status, data } = error.response;
    
    // Handle validation errors from backend
    if (status === 422 && data.detail) {
      return Array.isArray(data.detail) 
        ? data.detail.map(err => `${err.loc.join('.')} - ${err.msg}`).join(', ')
        : data.detail;
    }
    
    // Handle other error responses with detail
    if (data.detail) return data.detail;
    
    // Default message based on status code
    switch (status) {
      case 400: return 'Yêu cầu không hợp lệ';
      case 401: return 'Chưa đăng nhập hoặc đăng nhập đã hết hạn';
      case 403: return 'Bạn không có quyền truy cập tài nguyên này';
      case 404: return 'Không tìm thấy tài nguyên';
      case 500: return 'Lỗi máy chủ nội bộ';
      default: return `Lỗi (${status}): ${data.message || 'Không có thông tin chi tiết'}`;
    }
  }
  
  // Handle network errors
  if (error.request && !error.response) {
    return 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng của bạn.';
  }
  
  // Handle other errors
  return error.message || 'Đã xảy ra lỗi không xác định';
};

/**
 * Determines if an error is a network error that can be retried
 * @param {Error} error - The error object
 * @returns {boolean} Whether the error is retriable
 */
export const isRetriableError = (error) => {
  // Network errors can be retried
  if (error.request && !error.response) return true;
  
  // Some server errors can be retried
  if (error.response && [502, 503, 504].includes(error.response.status)) return true;
  
  return false;
};

/**
 * Maximum number of retry attempts for retriable errors
 */
const MAX_RETRIES = 3;

/**
 * Makes an API request with retry capability using exponential backoff.
 * @param {Function} apiCall - The async function making the API call.
 * @param {number} [retries=MAX_RETRIES] - Number of retry attempts remaining.
 * @returns {Promise<*>} API response data.
 * @throws {Error} Throws the error if retries are exhausted or the error is not retriable.
 */
export const withRetry = async (apiCall, retries = MAX_RETRIES) => {
  try {
    return await apiCall();
  } catch (error) {
    if (retries > 0 && isRetriableError(error)) {
      // Calculate delay using exponential backoff: 1s, 2s, 4s, ...
      const delay = 2 ** (MAX_RETRIES - retries) * 1000; 
      console.warn(`Retrying API call after ${delay}ms due to error: ${error.message}`);
      // Wait for the calculated delay
      await new Promise(resolve => setTimeout(resolve, delay));
      // Recursively call with one less retry attempt
      return withRetry(apiCall, retries - 1);
    }
    // If no retries left or error is not retriable, throw the error
    throw error;
  }
};
