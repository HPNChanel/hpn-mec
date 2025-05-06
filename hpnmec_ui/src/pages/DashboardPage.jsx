import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Container,
  Card,
  CardContent,
  CardHeader,
  Divider,
  List,
  ListItem,
  ListItemAvatar,
  Avatar,
  ListItemText,
  CircularProgress,
} from '@mui/material';
import PeopleIcon from '@mui/icons-material/People';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import EventNoteIcon from '@mui/icons-material/EventNote';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

function DashboardPage() {
  // Example stats and activities
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    users: 1245,
    doctors: 87,
    appointments: 5693,
    healthRecords: 3120,
  });
  const [recentActivities, setRecentActivities] = useState([]);

  useEffect(() => {
    setTimeout(() => {
      setStats({
        users: 1245,
        doctors: 87,
        appointments: 5693,
        healthRecords: 3120,
      });
      setRecentActivities([
        { id: 1, type: 'appointment', user: 'Nguyen Van A', doctor: 'Dr. Tran Thi B', time: '10 phút trước' },
        { id: 2, type: 'record', user: 'Le Van C', time: '30 phút trước' },
        { id: 3, type: 'register', user: 'Pham Thi D', time: '1 giờ trước' },
      ]);
      setLoading(false);
    }, 1200);
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4, fontWeight: 700, letterSpacing: 0.5 }}>
        Tổng quan
      </Typography>
      <Grid container spacing={3}>
        {/* Status Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={3} sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2, borderRadius: 3 }}>
            <PeopleIcon color="primary" sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h6" fontWeight={600}>Người dùng</Typography>
              <Typography variant="h4" color="primary" fontWeight={700}>{stats.users}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={3} sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2, borderRadius: 3 }}>
            <MedicalServicesIcon color="primary" sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h6" fontWeight={600}>Bác sĩ</Typography>
              <Typography variant="h4" color="primary" fontWeight={700}>{stats.doctors}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={3} sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2, borderRadius: 3 }}>
            <EventNoteIcon color="primary" sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h6" fontWeight={600}>Lịch hẹn</Typography>
              <Typography variant="h4" color="primary" fontWeight={700}>{stats.appointments}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={3} sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2, borderRadius: 3 }}>
            <MonitorHeartIcon color="primary" sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h6" fontWeight={600}>Hồ sơ sức khỏe</Typography>
              <Typography variant="h4" color="primary" fontWeight={700}>{stats.healthRecords}</Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Visualization and Timeline */}
        <Grid item xs={12} md={8}>
          <Card elevation={2} sx={{ borderRadius: 3 }}>
            <CardHeader title="Biểu đồ sức khỏe tổng quan" />
            <Divider />
            <CardContent>
              {/* Replace with real chart */}
              <Box sx={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'text.secondary' }}>
                <TrendingUpIcon sx={{ fontSize: 64, mr: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  (Biểu đồ sẽ hiển thị tại đây)
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card elevation={2} sx={{ borderRadius: 3 }}>
            <CardHeader title="Hoạt động gần đây" />
            <Divider />
            <CardContent sx={{ p: 0 }}>
              <List>
                {recentActivities.map((activity) => (
                  <ListItem key={activity.id} alignItems="flex-start">
                    <ListItemAvatar>
                      <Avatar color="primary">
                        {activity.type === 'appointment' && <EventNoteIcon />}
                        {activity.type === 'record' && <MonitorHeartIcon />}
                        {activity.type === 'register' && <PeopleIcon />}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        activity.type === 'appointment'
                          ? `${activity.user} đã đặt lịch hẹn với ${activity.doctor}`
                          : activity.type === 'record'
                          ? `${activity.user} đã cập nhật hồ sơ sức khỏe`
                          : `${activity.user} vừa đăng ký tài khoản`
                      }
                      secondary={activity.time}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

export default DashboardPage;