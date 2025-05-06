import React from 'react';
import { Grid, Paper, Typography, Box, Button } from '@mui/material';

function AdminDashboard() {
  // Replace with real data fetching
  const stats = {
    users: 1200,
    doctors: 80,
    healthRecords: 3500,
    appointments: 900,
  };

  return (
    <Box>
      <Typography variant="h4" fontWeight={700} mb={3}>Tổng quan hệ thống</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h6">Người dùng</Typography>
            <Typography variant="h4" color="primary">{stats.users}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h6">Bác sĩ</Typography>
            <Typography variant="h4" color="primary">{stats.doctors}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h6">Hồ sơ y tế</Typography>
            <Typography variant="h4" color="primary">{stats.healthRecords}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h6">Lịch hẹn</Typography>
            <Typography variant="h4" color="primary">{stats.appointments}</Typography>
          </Paper>
        </Grid>
      </Grid>
      {/* Add charts and quick action buttons here */}
    </Box>
  );
}

export default AdminDashboard;
