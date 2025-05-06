import React from 'react';
import { Container, Typography, Paper, Button, Box, Alert } from '@mui/material';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import SecurityIcon from '@mui/icons-material/Security';
import { useAuth } from '../hooks/useAuth.jsx';

function UnauthorizedPage() {
  const location = useLocation();
  const { user } = useAuth();
  
  // Get custom message from navigation state or use default
  const message = location.state?.message || 
    "Bạn không có quyền truy cập vào trang này. Điều này có thể là do bạn không có quyền cần thiết hoặc tài khoản của bạn không được cấp phép cho tính năng này.";

  return (
    <Container maxWidth="md" sx={{ py: 8 }}>
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center', 
          textAlign: 'center',
          borderRadius: 3,
          boxShadow: theme => (theme.palette.mode === 'dark' ? '0 4px 20px rgba(0,0,0,0.3)' : '0 4px 20px rgba(0,0,0,0.1)')
        }}
      >
        <SecurityIcon sx={{ fontSize: 72, color: 'error.main', mb: 2 }} />
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
          Truy cập bị từ chối
        </Typography>
        
        <Alert severity="error" sx={{ width: '100%', mb: 4, fontWeight: 500 }}>
          {message}
        </Alert>
        
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
          <Button 
            component={RouterLink} 
            to="/" 
            variant="contained" 
            color="primary"
            sx={{ px: 3, py: 1.2, fontWeight: 600, minWidth: 130 }}
          >
            Trang chủ
          </Button>
          
          {user && (
            <Button 
              component={RouterLink} 
              to="/dashboard" 
              variant="outlined" 
              color="primary"
              sx={{ px: 3, py: 1.2, fontWeight: 600, minWidth: 130 }}
            >
              Bảng điều khiển
            </Button>
          )}
          
          {!user && (
            <Button 
              component={RouterLink} 
              to="/login" 
              variant="outlined" 
              color="primary"
              sx={{ px: 3, py: 1.2, fontWeight: 600, minWidth: 130 }}
            >
              Đăng nhập
            </Button>
          )}
        </Box>
      </Paper>
    </Container>
  );
}

export default UnauthorizedPage;
