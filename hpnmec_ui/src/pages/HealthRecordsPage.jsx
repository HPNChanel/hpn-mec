import { useEffect, useState } from 'react';
import { Container, Typography, Paper } from '@mui/material';
import { healthRecordService } from '../services';

function HealthRecordsPage() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    healthRecordService.getAllHealthRecords()
      .then(data => {
        if (mounted) setRecords(data);
      })
      .catch(err => {
        if (mounted) setError('Không thể tải hồ sơ sức khỏe.');
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4, fontWeight: 600 }}>
        Hồ sơ sức khỏe
      </Typography>
      <Paper sx={{ p: 3 }}>
        {loading && <Typography variant="body1">Đang tải...</Typography>}
        {error && <Typography variant="body1" color="error">{error}</Typography>}
        {!loading && !error && records.length === 0 && (
          <Typography variant="body1">
            Không có hồ sơ sức khỏe nào.
          </Typography>
        )}
        {/* ...render health records list here if needed... */}
      </Paper>
    </Container>
  );
}

export default HealthRecordsPage;
