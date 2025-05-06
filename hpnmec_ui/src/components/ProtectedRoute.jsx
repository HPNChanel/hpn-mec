import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { CircularProgress, Box, Typography, Paper } from '@mui/material';
import { useAuth } from '../hooks/useAuth.jsx';
import { hasPermission } from '../utils/authUtils';

/**
 * A wrapper around routes that need authentication and specific role permissions
 * @param {object} props - Component props
 * @param {Array} props.allowedRoles - Array of roles that can access this route
 * @param {React.ReactNode} props.children - Child components to render if authenticated
 * @param {string} [props.redirectTo='/login'] - Redirect path if not authenticated
 * @returns {React.ReactNode} The protected component or redirect
 */
const ProtectedRoute = ({ 
  allowedRoles, 
  children, 
  redirectTo = '/login' 
}) => {
  const { user, loading, isAuthenticated } = useAuth();
  const location = useLocation();
  
  if (loading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center', 
          alignItems: 'center', 
          height: 'calc(100vh - 64px)', 
          gap: 2 
        }}
      >
        <CircularProgress size={40} thickness={4} />
        <Typography variant="h6" color="text.secondary">
          Đang xác thực...
        </Typography>
      </Box>
    );
  }
  
  if (!isAuthenticated) {
    // Save the current location they were trying to go to
    return <Navigate to={redirectTo} state={{ from: location }} replace />;
  }
  
  if (allowedRoles && !hasPermission(user, allowedRoles)) {
    // User is authenticated but doesn't have permission
    return <Navigate to="/unauthorized" replace />;
  }
  
  return children;
};

export default ProtectedRoute;
