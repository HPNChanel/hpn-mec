import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, Paper, CircularProgress } from '@mui/material';
// import DataGrid from MUI X if available, otherwise use Table as fallback
import { doctorService } from '../../services';

function AdminDoctorManagement() {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    doctorService.getAllDoctors()
      .then(setDoctors)
      .catch(() => setError('Lỗi tải danh sách bác sĩ.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" fontWeight={700}>Quản lý bác sĩ</Typography>
        <Button variant="contained" color="primary">Thêm bác sĩ</Button>
      </Box>
      <Paper sx={{ p: 3 }}>
        {loading && <CircularProgress />}
        {error && <Typography color="error">{error}</Typography>}
        {/* Replace with DataGrid or Table for doctors */}
        {/* ... */}
      </Paper>
    </Box>
  );
}

export default AdminDoctorManagement;
