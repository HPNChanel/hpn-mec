import { Box, Typography, Container, Paper, Grid, Button, useTheme } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import PeopleIcon from '@mui/icons-material/People';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';

function HomePage() {
  const theme = useTheme();
  
  return (
    <Box sx={{ minHeight: 'calc(100vh - 64px)' }}>
      <Box
        sx={{
          background: theme.palette.mode === 'dark'
            ? 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)'
            : 'linear-gradient(45deg, #2196f3 30%, #64b5f6 90%)',
          color: '#fff',
          py: { xs: 6, md: 10 },
          px: 2,
          borderRadius: { xs: 0, md: 2 },
          mb: 6,
          boxShadow: 3,
        }}
      >
        <Container maxWidth="lg">
          <Typography 
            variant="h3" 
            component="h1" 
            gutterBottom
            sx={{ 
              fontWeight: 700,
              fontSize: { xs: '2rem', md: '3rem' }
            }}
          >
            HPN Medicare
          </Typography>
          <Typography 
            variant="h5" 
            component="h2" 
            gutterBottom
            sx={{
              mb: 4,
              maxWidth: '800px',
              fontSize: { xs: '1.2rem', md: '1.5rem' }
            }}
          >
            Hệ thống quản lý y tế thông minh với công nghệ AI
          </Typography>
          <Box sx={{ mt: 4 }}>
            <Button
              variant="contained"
              color="secondary"
              size="large"
              component={RouterLink}
              to="/login"
              sx={{ 
                mr: 2, 
                px: 4, 
                py: 1.5,
                fontSize: '1rem',
                boxShadow: 2,
              }}
            >
              Đăng nhập ngay
            </Button>
            <Button
              variant="outlined"
              size="large"
              sx={{ 
                backgroundColor: 'rgba(255,255,255,0.1)',
                borderColor: '#fff',
                color: '#fff',
                px: 4,
                py: 1.5,
                fontSize: '1rem',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.2)',
                  borderColor: '#fff',
                }
              }}
              component={RouterLink}
              to="/dashboard"
            >
              Khám phá
            </Button>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="lg">
        <Typography 
          variant="h4" 
          component="h2" 
          align="center" 
          gutterBottom
          sx={{ mb: 6, fontWeight: 700, letterSpacing: 0.5 }}
        >
          Tính năng chính
        </Typography>
        
        <Grid container spacing={4}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper 
              elevation={theme.palette.mode === 'dark' ? 2 : 4}
              sx={{
                p: 3,
                height: '100%',
                borderRadius: 3,
                boxShadow: theme.palette.mode === 'dark'
                  ? '0 2px 12px 0 rgba(25, 118, 210, 0.08)'
                  : '0 2px 12px 0 rgba(25, 118, 210, 0.12)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-6px) scale(1.03)',
                  boxShadow: 8,
                }
              }}
            >
              <MonitorHeartIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                Phân tích AI
              </Typography>
              <Typography variant="body1">
                Sử dụng trí tuệ nhân tạo để phân tích dữ liệu sức khỏe và đưa ra cảnh báo sớm.
              </Typography>
            </Paper>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Paper 
              elevation={theme.palette.mode === 'dark' ? 2 : 4}
              sx={{
                p: 3,
                height: '100%',
                borderRadius: 3,
                boxShadow: theme.palette.mode === 'dark'
                  ? '0 2px 12px 0 rgba(25, 118, 210, 0.08)'
                  : '0 2px 12px 0 rgba(25, 118, 210, 0.12)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-6px) scale(1.03)',
                  boxShadow: 8,
                }
              }}
            >
              <HealthAndSafetyIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                Hồ sơ sức khỏe
              </Typography>
              <Typography variant="body1">
                Quản lý và theo dõi dữ liệu sức khỏe cá nhân một cách dễ dàng và bảo mật.
              </Typography>
            </Paper>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Paper 
              elevation={theme.palette.mode === 'dark' ? 2 : 4}
              sx={{
                p: 3,
                height: '100%',
                borderRadius: 3,
                boxShadow: theme.palette.mode === 'dark'
                  ? '0 2px 12px 0 rgba(25, 118, 210, 0.08)'
                  : '0 2px 12px 0 rgba(25, 118, 210, 0.12)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-6px) scale(1.03)',
                  boxShadow: 8,
                }
              }}
            >
              <MedicalServicesIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                Đặt lịch khám
              </Typography>
              <Typography variant="body1">
                Đặt lịch khám với bác sĩ yêu thích của bạn bất cứ lúc nào, bất cứ nơi đâu.
              </Typography>
            </Paper>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Paper 
              elevation={theme.palette.mode === 'dark' ? 2 : 4}
              sx={{
                p: 3,
                height: '100%',
                borderRadius: 3,
                boxShadow: theme.palette.mode === 'dark'
                  ? '0 2px 12px 0 rgba(25, 118, 210, 0.08)'
                  : '0 2px 12px 0 rgba(25, 118, 210, 0.12)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-6px) scale(1.03)',
                  boxShadow: 8,
                }
              }}
            >
              <PeopleIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                Tìm bác sĩ
              </Typography>
              <Typography variant="body1">
                Tìm kiếm bác sĩ theo chuyên khoa, đánh giá và lưu trữ các bác sĩ yêu thích.
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default HomePage;