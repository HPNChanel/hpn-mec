import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, Paper, CircularProgress } from '@mui/material';
// import DataGrid from MUI X if available, otherwise use Table as fallback
import { healthRecordService } from '../../services';

function AdminHealthRecordManagement() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    healthRecordService.getAllHealthRecords()
      .then(setRecords)
      .catch(() => setError('Lỗi tải hồ sơ y tế.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" fontWeight={700}>Quản lý hồ sơ y tế</Typography>
        <Button variant="contained" color="primary">Thêm hồ sơ</Button>
      </Box>
      <Paper sx={{ p: 3 }}>
        {loading && <CircularProgress />}
        {error && <Typography color="error">{error}</Typography>}
        {/* Replace with DataGrid or Table for health records */}
        {/* ... */}
      </Paper>
    </Box>
  );
}

export default AdminHealthRecordManagement;
