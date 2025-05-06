import { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  Link,
  IconButton,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  FormHelperText, // Add FormHelperText for validation messages
} from '@mui/material';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useThemeMode } from '../theme/ThemeProvider';
import { authService } from '../services'; // Import authService
import { formatErrorMessage } from '../utils/errorUtils'; // Assuming errorUtils exists

function RegisterPage() {
  const [formData, setFormData] = useState({
    username: '', // Add username
    full_name: '',
    email: '',
    password: '',
    confirmPassword: '',
    gender: '',
    date_of_birth: '', // Format YYYY-MM-DD
    phone_number: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validationErrors, setValidationErrors] = useState({}); // State for validation errors
  const { mode } = useThemeMode(); // Only need mode for styling consistency if necessary
  const navigate = useNavigate();

  const validateField = (name, value) => {
    let fieldError = '';
    switch (name) {
      case 'username':
        if (!value) {
          fieldError = 'Tên đăng nhập là bắt buộc.';
        } else if (value.length < 3) {
          fieldError = 'Tên đăng nhập phải có ít nhất 3 ký tự.';
        } else if (/\s/.test(value)) {
          fieldError = 'Tên đăng nhập không được chứa khoảng trắng.';
        }
        break;
      case 'full_name':
        if (!value) fieldError = 'Họ và tên là bắt buộc.';
        break;
      case 'email':
        if (!value) {
          fieldError = 'Email là bắt buộc.';
        } else if (!/\S+@\S+\.\S+/.test(value)) {
          fieldError = 'Địa chỉ email không hợp lệ.';
        }
        break;
      case 'password':
        if (!value) {
          fieldError = 'Mật khẩu là bắt buộc.';
        } else if (value.length < 6) { // Example: Minimum password length
          fieldError = 'Mật khẩu phải có ít nhất 6 ký tự.';
        }
        break;
      case 'confirmPassword':
        if (!value) {
          fieldError = 'Xác nhận mật khẩu là bắt buộc.';
        } else if (value !== formData.password) {
          fieldError = 'Mật khẩu xác nhận không khớp.';
        }
        break;
      case 'gender':
        if (!value) fieldError = 'Giới tính là bắt buộc.';
        break;
      case 'date_of_birth':
        if (!value) fieldError = 'Ngày sinh là bắt buộc.';
        break;
      case 'phone_number':
        if (!value) {
          fieldError = 'Số điện thoại là bắt buộc.';
        } else if (!/^\d{10,}$/.test(value)) { // Example: Basic phone number validation (10+ digits)
          fieldError = 'Số điện thoại không hợp lệ.';
        }
        break;
      default:
        break;
    }
    return fieldError;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));

    // Validate on change and clear error if valid
    const fieldError = validateField(name, value);
    setValidationErrors((prev) => ({ ...prev, [name]: fieldError }));

    // Also re-validate confirmPassword if password changes
    if (name === 'password') {
      const confirmPasswordError = validateField('confirmPassword', formData.confirmPassword);
      setValidationErrors((prev) => ({ ...prev, confirmPassword: confirmPasswordError }));
    }
  };

  const validateForm = () => {
    const errors = {};
    let isValid = true;
    Object.keys(formData).forEach((key) => {
      // Skip confirmPassword validation here as it's frontend only
      if (key !== 'confirmPassword') {
        const error = validateField(key, formData[key]);
        if (error) {
          errors[key] = error;
          isValid = false;
        }
      }
    });
    // Explicitly validate confirmPassword against password
    const confirmPasswordError = validateField('confirmPassword', formData.confirmPassword);
    if (confirmPasswordError) {
        errors.confirmPassword = confirmPasswordError;
        isValid = false;
    }

    setValidationErrors(errors);
    return isValid;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null); // Clear previous API errors
    setValidationErrors({}); // Clear previous validation errors

    if (!validateForm()) {
      return; // Stop submission if validation fails
    }

    setLoading(true);
    try {
      // Prepare data for API (exclude confirmPassword)
      const { confirmPassword, ...apiData } = formData;
      await authService.register(apiData);
      // Optionally show a success message before redirecting
      navigate('/login'); // Redirect to login page on successful registration
    } catch (err) {
      setError(formatErrorMessage(err)); // Use error formatting utility
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        bgcolor: (theme) => theme.palette.background.default,
        py: 4,
      }}
    >
      <Container component="main" maxWidth="sm"> {/* Increased maxWidth for more fields */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mb: 4,
          }}
        >
          <Button
            component={RouterLink}
            to="/login"
            startIcon={<ArrowBackIcon />}
            sx={{ textTransform: 'none' }}
          >
            Về trang đăng nhập
          </Button>
          {/* Theme toggle removed as it's likely in Navbar */}
        </Box>

        <Paper
          elevation={3}
          sx={{
            p: { xs: 3, sm: 4 }, // Responsive padding
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            borderRadius: 3,
            boxShadow: 3,
            width: '100%',
            maxWidth: 520,
            mx: 'auto',
          }}
        >
          <Typography
            component="h1"
            variant="h5"
            sx={{ mb: 3, fontWeight: 700, letterSpacing: 0.5 }}
          >
            Đăng ký tài khoản
          </Typography>

          {error && <Alert severity="error" sx={{ width: '100%', mb: 2 }}>{error}</Alert>}

          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1, width: '100%' }}>
            <Grid container spacing={2}>
              {/* Line 1: Username + Email */}
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  id="username"
                  label="Tên đăng nhập"
                  name="username"
                  autoComplete="username"
                  value={formData.username}
                  onChange={handleChange}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.username}
                  helperText={validationErrors.username}
                  autoFocus // Focus on username first
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  id="email"
                  label="Email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  value={formData.email}
                  onChange={handleChange}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.email}
                  helperText={validationErrors.email}
                />
              </Grid>

              {/* Line 2: Full Name + Phone */}
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  id="full_name"
                  label="Họ và tên"
                  name="full_name"
                  autoComplete="name"
                  value={formData.full_name}
                  onChange={handleChange}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.full_name}
                  helperText={validationErrors.full_name}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  id="phone_number"
                  label="Số điện thoại"
                  name="phone_number"
                  type="tel"
                  autoComplete="tel"
                  value={formData.phone_number}
                  onChange={handleChange}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.phone_number}
                  helperText={validationErrors.phone_number}
                />
              </Grid>

              {/* Line 3: Password + Confirm Password */}
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  name="password"
                  label="Mật khẩu"
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  autoComplete="new-password"
                  value={formData.password}
                  onChange={handleChange}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.password}
                  helperText={validationErrors.password}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          aria-label="toggle password visibility"
                          onClick={() => setShowPassword(!showPassword)}
                          edge="end"
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  name="confirmPassword"
                  label="Xác nhận mật khẩu"
                  type={showConfirmPassword ? 'text' : 'password'}
                  id="confirmPassword"
                  autoComplete="new-password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.confirmPassword}
                  helperText={validationErrors.confirmPassword}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          aria-label="toggle confirm password visibility"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          edge="end"
                        >
                          {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>

              {/* Line 4: Gender + Date of Birth */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth required variant="outlined" disabled={loading} error={!!validationErrors.gender}>
                  <InputLabel id="gender-label">Giới tính</InputLabel>
                  <Select
                    labelId="gender-label"
                    id="gender"
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    label="Giới tính"
                  >
                    <MenuItem value="">
                      <em>Chọn giới tính</em>
                    </MenuItem>
                    <MenuItem value={'male'}>Nam</MenuItem>
                    <MenuItem value={'female'}>Nữ</MenuItem>
                    <MenuItem value={'other'}>Khác</MenuItem>
                  </Select>
                  {validationErrors.gender && <FormHelperText>{validationErrors.gender}</FormHelperText>}
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  id="date_of_birth"
                  label="Ngày sinh"
                  name="date_of_birth"
                  type="date"
                  value={formData.date_of_birth}
                  onChange={handleChange}
                  InputLabelProps={{ shrink: true }}
                  variant="outlined"
                  disabled={loading}
                  error={!!validationErrors.date_of_birth}
                  helperText={validationErrors.date_of_birth}
                />
              </Grid>
            </Grid>
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2, py: 1.5, fontWeight: 600, fontSize: '1.1rem', borderRadius: 2 }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Đăng ký'}
            </Button>
            <Grid container justifyContent="center">
              <Grid item>
                <Typography variant="body2">
                  Đã có tài khoản?{' '}
                  <Link component={RouterLink} to="/login" variant="body2" fontWeight="500">
                    Đăng nhập
                  </Link>
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}

export default RegisterPage;
