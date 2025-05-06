import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, Paper, CircularProgress } from '@mui/material';
// import DataGrid from MUI X if available, otherwise use Table as fallback
// import { userService } from '../../services';

function AdminUserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    // userService.getAllUsers()
    //   .then(setUsers)
    //   .catch(() => setError('Lỗi tải danh sách người dùng.'))
    //   .finally(() => setLoading(false));
    setLoading(false); // Remove when API is ready
  }, []);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" fontWeight={700}>Quản lý người dùng</Typography>
        <Button variant="contained" color="primary">Thêm người dùng</Button>
      </Box>
      <Paper sx={{ p: 3 }}>
        {loading && <CircularProgress />}
        {error && <Typography color="error">{error}</Typography>}
        {/* Replace with DataGrid or Table for users */}
        {/* ... */}
      </Paper>
    </Box>
  );
}

export default AdminUserManagement;
