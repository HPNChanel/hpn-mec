import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  Box,
  Card, 
  CardContent,
  CardHeader,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  CircularProgress
} from '@mui/material';
import PeopleIcon from '@mui/icons-material/People';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import EventNoteIcon from '@mui/icons-material/EventNote';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { useAuth } from '../../hooks/useAuth.jsx';
import AdminStatCard from '../../components/admin/AdminStatCard';

function AdminDashboardPage() {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    users: 0,
    doctors: 0,
    appointments: 0,
    appointmentsToday: 0
  });
  const [recentActivities, setRecentActivities] = useState([]);

  useEffect(() => {
    // This would be replaced with actual API calls in a real implementation
    setTimeout(() => {
      setStats({
        users: 1245,
        doctors: 87,
        appointments: 5693,
        appointmentsToday: 124
      });
      
      setRecentActivities([
        { id: 1, type: 'user_registered', user: 'Nguyen Van A', timestamp: new Date(Date.now() - 25 * 60000) },
        { id: 2, type: 'appointment_created', user: 'Tran Thi B', doctor: 'Dr. Le Van C', timestamp: new Date(Date.now() - 45 * 60000) },
        { id: 3, type: 'doctor_approved', doctor: 'Dr. Pham Van D', timestamp: new Date(Date.now() - 120 * 60000) },
        { id: 4, type: 'user_registered', user: 'Hoang Van E', timestamp: new Date(Date.now() - 150 * 60000) }
      ]);
      
      setLoading(false);
    }, 1500);
  }, []);

  // Function to format timestamp
  const formatTime = (date) => {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.round(diffMs / 60000);
    
    if (diffMins < 60) {
      return `${diffMins} phút trước`;
    } else if (diffMins < 1440) {
      return `${Math.floor(diffMins / 60)} giờ trước`;
    } else {
      return `${Math.floor(diffMins / 1440)} ngày trước`;
    }
  };

  // Function to get activity text
  const getActivityText = (activity) => {
    switch (activity.type) {
      case 'user_registered':
        return `${activity.user} đã đăng ký tài khoản mới`;
      case 'appointment_created':
        return `${activity.user} đã đặt lịch hẹn với ${activity.doctor}`;
      case 'doctor_approved':
        return `${activity.doctor} đã được phê duyệt tài khoản`;
      default:
        return 'Hoạt động không xác định';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        {/* Welcome message */}
        <Grid item xs={12}>
          <Typography variant="h4" component="h1" gutterBottom fontWeight={600}>
            Quản trị hệ thống
          </Typography>
          <Typography variant="subtitle1" color="text.secondary" paragraph>
            Chào mừng, {user?.full_name || 'Admin'}! Đây là tổng quan về hệ thống HPN Medicare.
          </Typography>
          <Divider sx={{ my: 2 }} />
        </Grid>

        {/* Statistics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <AdminStatCard 
            title="Người dùng"
            value={stats.users}
            icon={<PeopleIcon fontSize="large" />}
            color="#1976d2"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <AdminStatCard 
            title="Bác sĩ"
            value={stats.doctors}
            icon={<MedicalServicesIcon fontSize="large" />}
            color="#2e7d32"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <AdminStatCard 
            title="Lịch hẹn"
            value={stats.appointments}
            icon={<EventNoteIcon fontSize="large" />}
            color="#ed6c02"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <AdminStatCard 
            title="Lịch hẹn hôm nay"
            value={stats.appointmentsToday}
            icon={<TrendingUpIcon fontSize="large" />}
            color="#9c27b0"
          />
        </Grid>

        {/* Recent Activities */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardHeader title="Hoạt động gần đây" />
            <Divider />
            <CardContent sx={{ p: 0 }}>
              <List>
                {recentActivities.map((activity) => (
                  <React.Fragment key={activity.id}>
                    <ListItem>
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: 'primary.main' }}>
                          {activity.type === 'user_registered' && <PeopleIcon />}
                          {activity.type === 'appointment_created' && <EventNoteIcon />}
                          {activity.type === 'doctor_approved' && <MedicalServicesIcon />}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText 
                        primary={getActivityText(activity)} 
                        secondary={formatTime(activity.timestamp)} 
                      />
                    </ListItem>
                    <Divider variant="inset" component="li" />
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardHeader title="Trạng thái hệ thống" />
            <Divider />
            <CardContent>
              <Typography variant="body1" paragraph>
                Hệ thống đang hoạt động bình thường.
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">CPU Usage</Typography>
                  <Box display="flex" alignItems="center">
                    <Box width="100%" mr={1}>
                      <LinearProgress variant="determinate" value={32} sx={{ height: 10, borderRadius: 5 }} />
                    </Box>
                    <Box minWidth={35}>
                      <Typography variant="body2" color="text.secondary">32%</Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Memory Usage</Typography>
                  <Box display="flex" alignItems="center">
                    <Box width="100%" mr={1}>
                      <LinearProgress variant="determinate" value={64} sx={{ height: 10, borderRadius: 5 }} />
                    </Box>
                    <Box minWidth={35}>
                      <Typography variant="body2" color="text.secondary">64%</Typography>
                    </Box>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

// Helper component for LinearProgress
function LinearProgress(props) {
  return (
    <Box
      sx={{
        height: props.sx?.height || 10,
        borderRadius: props.sx?.borderRadius || 5,
        bgcolor: 'background.paper',
        boxShadow: 'inset 0 1px 3px rgba(0,0,0,.2)',
        position: 'relative',
      }}
    >
      <Box
        sx={{
          width: `${props.value}%`,
          height: '100%',
          borderRadius: 'inherit',
          bgcolor: props.color || 'primary.main',
          transition: 'width .4s ease-in-out',
        }}
      />
    </Box>
  );
}

export default AdminDashboardPage;
