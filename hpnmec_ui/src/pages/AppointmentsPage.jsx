import { useEffect, useState } from 'react';
import { Container, Typography, Paper } from '@mui/material';
import { appointmentService } from '../services';

function AppointmentsPage() {
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    appointmentService.getAllAppointments()
      .then(data => {
        if (mounted) setAppointments(data);
      })
      .catch(err => {
        if (mounted) setError('Không thể tải lịch hẹn.');
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4, fontWeight: 600 }}>
        Lịch hẹn
      </Typography>
      <Paper sx={{ p: 3 }}>
        {loading && <Typography variant="body1">Đang tải...</Typography>}
        {error && <Typography variant="body1" color="error">{error}</Typography>}
        {!loading && !error && appointments.length === 0 && (
          <Typography variant="body1">
            Không có lịch hẹn nào.
          </Typography>
        )}
        {/* ...render appointments list here if needed... */}
      </Paper>
    </Container>
  );
}

export default AppointmentsPage;
