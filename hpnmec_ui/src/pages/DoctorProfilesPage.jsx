import { useEffect, useState } from 'react';
import { Container, Typography, Paper } from '@mui/material';
import { doctorService } from '../services';

function DoctorProfilesPage() {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    doctorService.getAllDoctors()
      .then(data => {
        if (mounted) setDoctors(data);
      })
      .catch(err => {
        if (mounted) setError('Không thể tải danh sách bác sĩ.');
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4, fontWeight: 600 }}>
        Danh sách Bác sĩ
      </Typography>
      <Paper sx={{ p: 3 }}>
        {loading && <Typography variant="body1">Đang tải...</Typography>}
        {error && <Typography variant="body1" color="error">{error}</Typography>}
        {!loading && !error && doctors.length === 0 && (
          <Typography variant="body1">
            Không có bác sĩ nào được tìm thấy.
          </Typography>
        )}
        {/* ...render doctors list here if needed... */}
      </Paper>
    </Container>
  );
}

export default DoctorProfilesPage;
