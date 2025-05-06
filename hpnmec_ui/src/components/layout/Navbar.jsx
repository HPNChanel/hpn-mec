import { useState } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Button,
  Avatar,
  Menu,
  MenuItem,
  Box,
  useMediaQuery,
  Fade,
  Chip,
  Divider,
  ListItemIcon,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import NotificationsIcon from '@mui/icons-material/Notifications';
import PersonIcon from '@mui/icons-material/Person';
import LogoutIcon from '@mui/icons-material/Logout';
import SettingsIcon from '@mui/icons-material/Settings';
import { useThemeMode } from '../../theme/ThemeProvider';
import { useAuth } from '../../hooks/useAuth';
import { isAdmin } from '../../utils/authUtils';

function Navbar({ toggleSidebar }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { mode, toggleColorMode } = useThemeMode();
  const { user, isAuthenticated, logout } = useAuth();
  const [anchorEl, setAnchorEl] = useState(null);

  const handleMenu = (event) => setAnchorEl(event.currentTarget);
  const handleClose = () => setAnchorEl(null);

  return (
    <AppBar
      position="fixed"
      elevation={2}
      color="default"
      sx={{
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        zIndex: (theme) => theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar sx={{ minHeight: { xs: 56, sm: 64 }, px: { xs: 1, sm: 3 } }}>
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={toggleSidebar}
          sx={{ mr: 2, display: { sm: 'none' } }}
        >
          <MenuIcon />
        </IconButton>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            flexGrow: 1,
            textDecoration: 'none',
            color: 'inherit',
            fontWeight: 700,
            letterSpacing: 0.5,
            fontSize: { xs: '1.1rem', sm: '1.3rem' },
            transition: 'color 0.2s',
            '&:hover': { color: 'primary.main' },
          }}
        >
          HPN Medicare
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton sx={{ ml: 1 }} onClick={toggleColorMode} color="inherit">
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          <IconButton color="inherit" sx={{ ml: 1 }}>
            <NotificationsIcon />
          </IconButton>
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
                      width: 32,
                      height: 32,
                      fontSize: '1.1rem',
                    }}
                  >
                    {user.full_name?.charAt(0) || user.username?.charAt(0) || 'U'}
                  </Avatar>
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <span style={{ fontWeight: 600 }}>{user.full_name || user.username}</span>
                    <Chip
                      label={isAdmin(user) ? 'Quản trị viên' : 'Người dùng'}
                      color={isAdmin(user) ? 'error' : 'default'}
                      size="small"
                      sx={{
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
                onClick={handleMenu}
                clickable
                sx={{
                  height: 44,
                  pl: 0.5,
                  ml: 2,
                  fontWeight: 600,
                  fontSize: '1rem',
                  bgcolor: isAdmin(user) ? 'error.light' : 'grey.100',
                  color: 'text.primary',
                  border: isAdmin(user) ? '2px solid' : undefined,
                  borderColor: isAdmin(user) ? 'error.main' : undefined,
                  boxShadow: 1,
                  transition: 'box-shadow 0.2s',
                  '&:hover': { boxShadow: 3 },
                  display: { xs: 'none', sm: 'flex' },
                }}
              />
              <IconButton
                color="inherit"
                onClick={handleMenu}
                sx={{ display: { xs: 'flex', sm: 'none' }, ml: 1 }}
              >
                <PersonIcon />
              </IconButton>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleClose}
                TransitionComponent={Fade}
                transformOrigin={{ horizontal: 'right', vertical: 'top' }}
                anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
                PaperProps={{
                  elevation: 3,
                  sx: {
                    minWidth: 240,
                    mt: 1,
                    borderRadius: 2,
                    boxShadow: 8,
                    p: 0,
                  },
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
                <MenuItem onClick={handleClose} component={RouterLink} to="/profile">
                  <ListItemIcon><PersonIcon /></ListItemIcon>
                  Tài khoản
                </MenuItem>
                <MenuItem onClick={handleClose} component={RouterLink} to="/settings">
                  <ListItemIcon><SettingsIcon /></ListItemIcon>
                  Cài đặt
                </MenuItem>
                <MenuItem onClick={() => { handleClose(); logout(); }}>
                  <ListItemIcon><LogoutIcon /></ListItemIcon>
                  Đăng xuất
                </MenuItem>
              </Menu>
            </>
          ) : (
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                color="inherit"
                component={RouterLink}
                to="/login"
                sx={{
                  fontWeight: 600,
                  borderRadius: 2,
                  px: 2,
                  display: { xs: 'none', sm: 'block' },
                }}
              >
                Đăng nhập
              </Button>
              <Button
                variant="contained"
                color="primary"
                component={RouterLink}
                to="/register"
                sx={{
                  fontWeight: 600,
                  borderRadius: 2,
                  px: 2,
                  boxShadow: 2,
                  display: { xs: 'none', sm: 'block' },
                }}
              >
                Đăng ký
              </Button>
            </Box>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;