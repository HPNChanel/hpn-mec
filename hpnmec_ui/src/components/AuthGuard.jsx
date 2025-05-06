import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { CircularProgress, Box, Typography } from '@mui/material';
import { useAuth } from '../hooks/useAuth.jsx';

/**
 * AuthGuard component to prevent authenticated users from accessing certain pages
 * or to prevent unauthenticated users from accessing protected pages
 * 
 * @param {object} props Component props
 * @param {boolean} props.requireAuth If true, requires authentication to proceed
 * @param {string} props.redirectTo The path to redirect to if conditions are not met
 * @returns React component
 */
function AuthGuard({ requireAuth = true, redirectTo = requireAuth ? '/login' : '/dashboard' }) {
  const { isAuthenticated, loading } = useAuth();
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
          Đang kiểm tra...
        </Typography>
      </Box>
    );
  }
  
  // If requireAuth is true, we need to be authenticated to proceed
  // If requireAuth is false, we need to be NOT authenticated to proceed
  const canAccess = requireAuth ? isAuthenticated : !isAuthenticated;
  
  if (!canAccess) {
    // When redirecting from protected routes, save the location
    return <Navigate 
      to={redirectTo} 
      state={requireAuth ? { from: location } : undefined} 
      replace 
    />;
  }
  
  return <Outlet />;
}

export default AuthGuard;
