import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { CircularProgress, Box, Typography, Paper, Alert } from '@mui/material';
import { useAuth } from '../hooks/useAuth.jsx';
import { hasPermission, isAdmin } from '../utils/authUtils';

/**
 * A specialized wrapper for routes that require specific role permissions
 * Provides enhanced security and better user feedback for role-based access
 * 
 * @param {object} props - Component props
 * @param {Array} props.allowedRoles - Array of roles that can access this route (required)
 * @param {React.ReactNode} props.children - Child components to render if authenticated and authorized
 * @param {string} [props.redirectTo='/unauthorized'] - Redirect path for unauthorized users
 * @param {boolean} [props.strictAdmin=false] - If true, only admins can access regardless of allowedRoles
 * @returns {React.ReactNode} The protected component or redirect with appropriate message
 */
const RoleProtectedRoute = ({ 
  allowedRoles, 
  children, 
  redirectTo = '/unauthorized',
  strictAdmin = false
}) => {
  const { user, loading, isAuthenticated } = useAuth();
  const location = useLocation();
  
  // Enhanced loading state with text
  if (loading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center', 
          alignItems: 'center', 
          height: 'calc(100vh - 64px)', // Adjust height as needed
          gap: 2
        }}
      >
        <CircularProgress size={40} thickness={4} color="primary" />
        <Typography variant="h6" color="text.secondary">
          Đang kiểm tra quyền truy cập...
        </Typography>
      </Box>
    );
  }
  
  // Handle unauthenticated users
  if (!isAuthenticated) {
    return <Navigate 
      to="/login" 
      state={{ 
        from: location,
        message: "Vui lòng đăng nhập để tiếp tục." // Vietnamese message
      }} 
      replace 
    />;
  }
  
  // For strict admin routes, only allow admins regardless of allowedRoles
  if (strictAdmin && !isAdmin(user)) {
    return <Navigate 
      to={redirectTo}
      state={{ 
        message: "Trang này chỉ dành cho quản trị viên." // Vietnamese message
      }}
      replace 
    />;
  }
  
  // For non-strict routes, check role permissions
  // Ensure allowedRoles is provided before checking permissions
  if (allowedRoles && Array.isArray(allowedRoles) && !hasPermission(user, allowedRoles)) {
    return <Navigate 
      to={redirectTo}
      state={{ 
        message: "Bạn không có quyền truy cập vào trang này." // Vietnamese message
      }}
      replace 
    />;
  }
  
  // User is authenticated and authorized
  return children;
};

export default RoleProtectedRoute;
