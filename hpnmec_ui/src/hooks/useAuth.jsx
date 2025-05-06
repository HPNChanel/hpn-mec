import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../services';
import { setAuthToken, clearAuthToken } from '../services/api';
import { getCacheItem, setCacheItem, removeCacheItem } from '../utils/cacheUtils';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initializeAuth = async () => {
      setLoading(true);
      setError(null);
      const cachedToken = getCacheItem('authToken');
      const cachedUser = getCacheItem('user');

      if (cachedToken && cachedUser) {
        setAuthToken(cachedToken);
        setUser(cachedUser);
        // Optionally verify token validity with the backend here
        try {
          // Example: await authService.verifyToken(); // Implement this if needed
          console.log("User loaded from cache.");
        } catch (verifyError) {
          console.error("Token verification failed, clearing auth state.", verifyError);
          // Use the logout function defined within this provider
          clearAuthToken();
          setUser(null);
          removeCacheItem('authToken');
          removeCacheItem('user');
        }
      }
      setLoading(false);
    };
    initializeAuth();
  }, []);

  const login = async (credentials) => {
    setLoading(true);
    setError(null);
    try {
      const { access_token, user: userData } = await authService.login(credentials);
      setAuthToken(access_token);
      setUser(userData);
      setCacheItem('authToken', access_token); // Cache token
      setCacheItem('user', userData); // Cache user data
      setLoading(false);
      return userData;
    } catch (err) {
      console.error("Login failed:", err);
      setError(err.response?.data?.detail || 'Đăng nhập thất bại.'); // Vietnamese error
      setLoading(false);
      throw err; // Re-throw error for component handling
    }
  };

  const register = async (userData) => {
    setLoading(true);
    setError(null);
    try {
      const registeredUser = await authService.register(userData);
      setLoading(false);
      // Optionally log the user in automatically after registration
      // await login({ username: userData.username, password: userData.password });
      return registeredUser;
    } catch (err) {
      console.error("Registration failed:", err);
      setError(err.response?.data?.detail || 'Đăng ký thất bại.'); // Vietnamese error
      setLoading(false);
      throw err; // Re-throw error for component handling
    }
  };

  const logout = async () => {
    setLoading(true);
    setError(null);
    try {
      // Optional: Call backend logout endpoint if it exists
      // await authService.logout();
      clearAuthToken();
      setUser(null);
      removeCacheItem('authToken'); // Clear cached token
      removeCacheItem('user'); // Clear cached user data
      console.log("User logged out.");
    } catch (err) {
      console.error("Logout failed:", err);
      // Even if backend logout fails, clear frontend state
      clearAuthToken();
      setUser(null);
      removeCacheItem('authToken');
      removeCacheItem('user');
      setError('Đăng xuất thất bại.'); // Vietnamese error
    } finally {
      setLoading(false);
    }
  };

  const value = {
    user,
    loading,
    error,
    login,
    register,
    logout,
    isAuthenticated: !!user,
  };

  // The use of JSX requires this file to be .jsx
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
