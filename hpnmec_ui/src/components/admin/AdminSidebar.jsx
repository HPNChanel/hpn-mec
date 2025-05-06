import React from 'react';
import { NavLink as RouterLink } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Paper,
  Typography
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import PeopleIcon from '@mui/icons-material/People';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import EventNoteIcon from '@mui/icons-material/EventNote';
import SettingsIcon from '@mui/icons-material/Settings';
import BarChartIcon from '@mui/icons-material/BarChart';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import { styled } from '@mui/material/styles';

// Custom styled component for the active nav link
const StyledNavLink = styled(RouterLink)(({ theme }) => ({
  textDecoration: 'none',
  color: theme.palette.text.primary,
  display: 'block',
  width: '100%',
  borderRadius: theme.shape.borderRadius,
  '&.active': {
    backgroundColor: theme.palette.action.selected,
    '& .MuiListItemIcon-root': {
      color: theme.palette.primary.main,
    },
    '& .MuiListItemText-primary': {
      fontWeight: 600,
      color: theme.palette.primary.main,
    },
  },
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

/**
 * Sidebar component for admin dashboard
 */
function AdminSidebar() {
  const menuItems = [
    { text: 'Tổng quan', icon: <DashboardIcon />, path: '/admin' },
    { text: 'Quản lý người dùng', icon: <PeopleIcon />, path: '/admin/users' },
    { text: 'Quản lý bác sĩ', icon: <MedicalServicesIcon />, path: '/admin/doctors' },
    { text: 'Quản lý hồ sơ y tế', icon: <HealthAndSafetyIcon />, path: '/admin/health-records' },
    { text: 'Báo cáo thống kê', icon: <BarChartIcon />, path: '/admin/reports' },
    { text: 'Cài đặt hệ thống', icon: <SettingsIcon />, path: '/admin/settings' },
  ];

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        width: 280, 
        borderRight: 1, 
        borderColor: 'divider',
        height: '100%',
        position: 'fixed',
        overflowY: 'auto'
      }}
    >
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
        <HealthAndSafetyIcon color="primary" sx={{ fontSize: 32, mr: 1 }} />
        <Typography variant="h6" component="h1" fontWeight={600}>
          HPN Medicare Admin
        </Typography>
      </Box>
      <Divider />
      <List component="nav" sx={{ px: 2 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} sx={{ px: 1, py: 0.5 }}>
            <StyledNavLink to={item.path} className={({ isActive }) => isActive ? 'active' : ''}>
              <ListItem>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            </StyledNavLink>
          </ListItem>
        ))}
      </List>
    </Paper>
  );
}

export default AdminSidebar;
