import React from 'react';
import { Box, Container, Typography, Link, Divider } from '@mui/material';

function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: (theme) =>
          theme.palette.mode === 'light' ? theme.palette.grey[100] : theme.palette.grey[900],
      }}
    >
      <Divider />
      <Container maxWidth="lg">
        <Box sx={{ pt: 3, pb: 2, display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            {'© '}
            {new Date().getFullYear()}
            {' '}
            <Link color="inherit" href="/">
              HPN Medicare
            </Link>
            {' - Hệ thống quản lý y tế trực tuyến'}
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}

export default Footer;
