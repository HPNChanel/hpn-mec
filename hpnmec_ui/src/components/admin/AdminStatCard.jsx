import React from 'react';
import { Box, Card, CardContent, Typography } from '@mui/material';

/**
 * A card component for displaying statistics in the admin dashboard
 * 
 * @param {Object} props
 * @param {string} props.title - The title of the statistic
 * @param {number|string} props.value - The value to display
 * @param {React.ReactNode} props.icon - Icon element to display
 * @param {string} props.color - Color for the icon
 * @returns {React.ReactElement}
 */
function AdminStatCard({ title, value, icon, color = 'primary.main' }) {
  return (
    <Card elevation={2} sx={{ height: '100%' }}>
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h5" component="div" fontWeight={600}>
              {value.toLocaleString()}
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              {title}
            </Typography>
          </Box>
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              bgcolor: `${color}22`, // Using hex code with 22 opacity
              p: 1.5,
              borderRadius: '50%'
            }}
          >
            {React.cloneElement(icon, { sx: { color: color } })}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

export default AdminStatCard;
