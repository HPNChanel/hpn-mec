import React, { useState } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Box,
  Menu,
  MenuItem,
  Avatar,
  Chip,
  Divider,
  ListItemIcon,
  useTheme,
  useMediaQuery,
  Drawer,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import DashboardIcon from '@mui/icons-material/Dashboard';
import EventNoteIcon from '@mui/icons-material/EventNote';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';
import LogoutIcon from '@mui/icons-material/Logout';
import SettingsIcon from '@mui/icons-material/Settings';
import PersonIcon from '@mui/icons-material/Person';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { useThemeMode } from '../theme/ThemeProvider';
import { useAuth } from '../hooks/useAuth';
import { isAdmin } from '../utils/authUtils';

function Navbar() {
  const theme = useTheme();
  const { mode, toggleColorMode } = useThemeMode();
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleMobileMenuToggle = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const handleLogout = () => {
    handleProfileMenuClose();
    logout();
    navigate('/login');
  };

  const handleNavigate = (path) => {
    navigate(path);
    handleProfileMenuClose();
    setMobileMenuOpen(false);
  };

  // Navigation items for the app
  const navItems = [
    { text: 'Trang chủ', path: '/' },
    { text: 'Bác sĩ', path: '/doctors' },
    { text: 'Dịch vụ', path: '/services' },
    { text: 'Giới thiệu', path: '/about' },
    { text: 'Liên hệ', path: '/contact' },
  ];

  // Navigation items for authenticated users
  const authNavItems = [
    { text: 'Bảng điều khiển', path: '/dashboard', icon: <DashboardIcon /> },
    { text: 'Lịch hẹn', path: '/appointments', icon: <EventNoteIcon /> },
    { text: 'Bác sĩ', path: '/doctors', icon: <MedicalServicesIcon /> },
    { text: 'Hồ sơ sức khỏe', path: '/health-records', icon: <HealthAndSafetyIcon /> },
  ];

  // Admin menu item - only shown to admins
  const adminMenuItem = {
    text: 'Quản trị hệ thống',
    path: '/admin',
    icon: <AdminPanelSettingsIcon />,
  };

  return (
    <>
      <AppBar position="static" color="default" elevation={1} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar>
          <Box display="flex" alignItems="center" component={RouterLink} to="/" sx={{ textDecoration: 'none', color: 'inherit' }}>
            <HealthAndSafetyIcon color="primary" sx={{ mr: 1 }} />
            <Typography variant="h6" component="div" sx={{ fontWeight: 700, letterSpacing: 0.5 }}>
              HPN Medicare
            </Typography>
          </Box>
          <Box sx={{ display: { xs: 'none', md: 'flex' }, ml: 4 }}>
            {navItems.map((item) => (
              <Button
                key={item.text}
                component={RouterLink}
                to={item.path}
                sx={{ color: 'inherit', mx: 1, fontWeight: 500 }}
              >
                {item.text}
              </Button>
            ))}
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          <IconButton onClick={toggleColorMode} color="inherit" sx={{ ml: 1 }}>
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={handleMobileMenuToggle}
              sx={{ ml: 1 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          {isAuthenticated ? (
            <>
              <Chip
                avatar={
                  <Avatar
                    src={user.profile_picture}
                    alt={user.full_name}
                    sx={{
                      bgcolor: isAdmin(user) ? 'error.main' : 'primary.main',
                      color: '#fff',
                      fontWeight: 700,
                    }}
                  >
                    {user.full_name?.charAt(0) || user.username?.charAt(0) || 'U'}
                  </Avatar>
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <span>{user.full_name || user.username}</span>
                    <Chip
                      label={isAdmin(user) ? 'Quản trị viên' : 'Người dùng'}
                      color={isAdmin(user) ? 'error' : 'default'}
                      size="small"
                      sx={{
                        ml: 1,
                        fontWeight: 600,
                        fontSize: '0.8rem',
                        height: 22,
                        ...(isAdmin(user) && {
                          bgcolor: 'error.main',
                          color: '#fff',
                        }),
                      }}
                    />
                  </Box>
                }
                onClick={handleProfileMenuOpen}
                clickable
                sx={{
                  height: 44,
                  pl: 0.5,
                  ml: 2,
                  display: { xs: 'none', sm: 'flex' },
                  fontWeight: 600,
                  fontSize: '1rem',
                  bgcolor: isAdmin(user) ? 'error.light' : 'grey.100',
                  color: 'text.primary',
                  border: isAdmin(user) ? '2px solid' : undefined,
                  borderColor: isAdmin(user) ? 'error.main' : undefined,
                }}
              />
              <IconButton
                color="inherit"
                onClick={handleProfileMenuOpen}
                sx={{ display: { xs: 'flex', sm: 'none' }, ml: 1 }}
              >
                <AccountCircleIcon />
              </IconButton>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleProfileMenuClose}
                transformOrigin={{ horizontal: 'right', vertical: 'top' }}
                anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
                PaperProps={{
                  elevation: 3,
                  sx: { minWidth: 240, mt: 1, borderRadius: 2 },
                }}
              >
                <Box sx={{ px: 2, py: 1 }}>
                  <Typography variant="subtitle1" fontWeight={700}>
                    {user.full_name || user.username}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {user.email}
                  </Typography>
                  {isAdmin(user) && (
                    <Chip
                      label="Quản trị viên"
                      color="error"
                      size="small"
                      sx={{ mt: 1, fontWeight: 600, fontSize: '0.8rem', height: 22 }}
                    />
                  )}
                </Box>
                <Divider />
                {authNavItems.map((item) => (
                  <MenuItem key={item.text} onClick={() => handleNavigate(item.path)}>
                    <ListItemIcon>{item.icon}</ListItemIcon>
                    {item.text}
                  </MenuItem>
                ))}
                {isAdmin(user) && (
                  <MenuItem onClick={() => handleNavigate(adminMenuItem.path)}>
                    <ListItemIcon>{adminMenuItem.icon}</ListItemIcon>
                    {adminMenuItem.text}
                  </MenuItem>
                )}
                <Divider />
                <MenuItem onClick={() => handleNavigate('/profile')}>
                  <ListItemIcon><PersonIcon /></ListItemIcon>
                  Tài khoản
                </MenuItem>
                <MenuItem onClick={() => handleNavigate('/settings')}>
                  <ListItemIcon><SettingsIcon /></ListItemIcon>
                  Cài đặt
                </MenuItem>
                <MenuItem onClick={handleLogout}>
                  <ListItemIcon><LogoutIcon /></ListItemIcon>
                  Đăng xuất
                </MenuItem>
              </Menu>
            </>
          ) : (
            <Box sx={{ display: 'flex' }}>
              <Button 
                color="inherit" 
                component={RouterLink} 
                to="/login"
                sx={{ display: { xs: 'none', sm: 'block' }, fontWeight: 600 }}
              >
                Đăng nhập
              </Button>
              <Button 
                variant="contained" 
                color="primary"
                component={RouterLink} 
                to="/register"
                sx={{ ml: 1, fontWeight: 600 }}
              >
                Đăng ký
              </Button>
            </Box>
          )}
        </Toolbar>
      </AppBar>
      <Drawer
        anchor="right"
        open={mobileMenuOpen}
        onClose={handleMobileMenuToggle}
        PaperProps={{
          sx: { width: 280, borderRadius: 0 },
        }}
      >
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6" component="div" sx={{ fontWeight: 700 }}>
            Menu
          </Typography>
          <IconButton onClick={handleMobileMenuToggle}>
            <MenuIcon />
          </IconButton>
        </Box>
        <Divider />
        <List>
          {navItems.map((item) => (
            <ListItem button key={item.text} onClick={() => handleNavigate(item.path)}>
              <ListItemText primary={item.text} />
            </ListItem>
          ))}
          <Divider />
          {isAuthenticated ? (
            <>
              {authNavItems.map((item) => (
                <ListItem button key={item.text} onClick={() => handleNavigate(item.path)}>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItem>
              ))}
              {isAdmin(user) && (
                <ListItem button onClick={() => handleNavigate(adminMenuItem.path)}>
                  <ListItemIcon>{adminMenuItem.icon}</ListItemIcon>
                  <ListItemText primary={adminMenuItem.text} />
                </ListItem>
              )}
              <Divider />
              <ListItem button onClick={handleLogout}>
                <ListItemIcon><LogoutIcon /></ListItemIcon>
                <ListItemText primary="Đăng xuất" />
              </ListItem>
            </>
          ) : (
            <>
              <ListItem button onClick={() => handleNavigate('/login')}>
                <ListItemText primary="Đăng nhập" />
              </ListItem>
              <ListItem button onClick={() => handleNavigate('/register')}>
                <ListItemText primary="Đăng ký" />
              </ListItem>
            </>
          )}
        </List>
      </Drawer>
    </>
  );
}

export default Navbar;
