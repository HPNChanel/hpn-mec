import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Navbar from './Navbar.jsx';
import Footer from './Footer.jsx';
import AdminSidebar from './admin/AdminSidebar';
import { useAuth } from '../hooks/useAuth';
import { isAdmin } from '../utils/authUtils';

function Layout() {
  const location = useLocation();
  const { user, isAuthenticated } = useAuth();
  
  // Show admin sidebar for all /admin/* routes
  const isAdminRoute = location.pathname.startsWith('/admin');
  const showAdminSidebar = isAuthenticated && isAdmin(user) && isAdminRoute;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Navbar />
      
      <Box sx={{ display: 'flex', flexGrow: 1 }}>
        {/* Render admin sidebar for admin routes */}
        {showAdminSidebar && (
          <Box sx={{ width: 280, flexShrink: 0 }}>
            <AdminSidebar />
          </Box>
        )}
        
        {/* Main content */}
        <Box 
          component="main" 
          sx={{ 
            flexGrow: 1,
            p: 3,
            ml: showAdminSidebar ? '280px' : 0,
            transition: 'margin 0.2s',
          }}
        >
          <Outlet />
        </Box>
      </Box>
      
      <Footer />
    </Box>
  );
}

export default Layout;
