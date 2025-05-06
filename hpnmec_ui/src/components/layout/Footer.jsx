import React from 'react';
import { Box, Typography, Container, Link } from '@mui/material';

// Get current year and app version (replace with actual version if available)
const currentYear = new Date().getFullYear();
const appVersion = process.env.REACT_APP_VERSION || '1.0.0'; // Example: Get version from env

function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto', // Push footer to the bottom
        backgroundColor: (theme) =>
          theme.palette.mode === 'light'
            ? theme.palette.grey[200]
            : theme.palette.grey[800],
        borderTop: (theme) => `1px solid ${theme.palette.divider}`,
      }}
    >
      <Container maxWidth="lg">
        <Typography variant="body2" color="text.secondary" align="center">
          {'Bản quyền © '}
          <Link color="inherit" href="https://your-website.com/"> {/* Replace with your link */}
            HPN Medicare
          </Link>{' '}
          {currentYear}
          {'.'}
        </Typography>
        <Typography variant="caption" color="text.secondary" align="center" display="block" sx={{ mt: 0.5 }}>
          Phiên bản {appVersion}
        </Typography>
      </Container>
    </Box>
  );
}

export default Footer;
