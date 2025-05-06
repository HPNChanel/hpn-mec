import api from './api';

// Authentication service functions
const authService = {
  login: async (credentials) => {
    // Ensure proper payload format matching backend expectations
    let payload;
    
    if (typeof credentials === 'object') {
      // Already an object, use as is
      payload = credentials;
    } else if (typeof credentials === 'string') {
      // Only email was passed, but we need both email and password
      console.error('Invalid credentials format: expected object with email and password');
      throw new Error('Invalid credentials format');
    } else {
      // Neither object nor string - invalid
      console.error('Invalid credentials format');
      throw new Error('Invalid credentials format');
    }
    
    // Log the payload for debugging
    console.log('Sending login payload:', payload);
    
    const response = await api.post('/auth/login', payload);
    return {
      access_token: response.data.token,
      user: response.data.user
    };
  },
  
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },
  
  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },
  
  getCurrentUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },
  
  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  },
};

export default authService;