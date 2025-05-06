import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Box, useMediaQuery, useTheme } from '@mui/material';
import Navbar from './Navbar.jsx';
import Sidebar from './Sidebar.jsx';

function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const toggleSidebar = () => {
    setSidebarOpen((open) => !open);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar open={sidebarOpen} onClose={toggleSidebar} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, sm: 3, md: 4 },
          mt: { xs: '56px', sm: '64px' },
          width: '100%',
          minHeight: 'calc(100vh - 56px)',
          transition: 'margin 0.2s cubic-bezier(.4,0,.2,1)',
          ...(isMobile
            ? {}
            : {
                ml: sidebarOpen ? '240px' : 0,
                width: sidebarOpen ? 'calc(100% - 240px)' : '100%',
              }),
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}

export default Layout;