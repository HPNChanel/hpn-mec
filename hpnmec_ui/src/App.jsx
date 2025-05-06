import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { UserPreferencesProvider } from './contexts/UserPreferencesContext';
import { UIStateProvider } from './contexts/UIStateContext';
import { CachedDataProvider } from './contexts/CachedDataContext';
import { ThemeModeProvider } from './theme/ThemeProvider';
import { AuthProvider } from './hooks/useAuth.jsx'; // Ensure .jsx extension
import { Layout } from './components';
import ProtectedRoute from './components/ProtectedRoute';
import RoleProtectedRoute from './components/RoleProtectedRoute';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import AppointmentsPage from './pages/AppointmentsPage';
import DoctorProfilesPage from './pages/DoctorProfilesPage';
import HealthRecordsPage from './pages/HealthRecordsPage';
import RegisterPage from './pages/RegisterPage';
import UnauthorizedPage from './pages/UnauthorizedPage';
import AdminDashboardPage from './pages/admin/AdminDashboardPage';
import AdminUserManagementPage from './pages/admin/AdminUserManagementPage';
import AdminDoctorManagementPage from './pages/admin/AdminDoctorManagementPage';
import { UserRole } from './types';
import AuthGuard from './components/AuthGuard';
import AdminLayout from './components/admin/AdminLayout';
import AdminHealthRecordManagement from './pages/admin/AdminHealthRecordManagement';

function App() {
  return (
    <UserPreferencesProvider>
      <UIStateProvider>
        <CachedDataProvider>
          <AuthProvider>
            <ThemeModeProvider>
              <Routes>
                {/* Public routes with guards to prevent authenticated users from accessing */}
                <Route element={<AuthGuard requireAuth={false} />}>
                  <Route path="/login" element={<LoginPage />} />
                  <Route path="/register" element={<RegisterPage />} />
                </Route>
                
                {/* Always public routes */}
                <Route path="/unauthorized" element={<UnauthorizedPage />} />
                
                {/* Main layout - all routes within the main app layout */}
                <Route path="/" element={<Layout />}>
                  {/* Public routes within main layout */}
                  <Route index element={<HomePage />} />
                  <Route path="doctors" element={<DoctorProfilesPage />} />
                  
                  {/* Protected routes for authenticated users (any role) */}
                  <Route element={<ProtectedRoute />}>
                    <Route path="dashboard" element={<DashboardPage />} />
                    <Route path="appointments" element={<AppointmentsPage />} />
                    <Route path="health-records" element={<HealthRecordsPage />} />
                  </Route>
                  
                  {/* Admin routes - strict role protection */}
                  {/* Using RoleProtectedRoute with strictAdmin=true ensures only ADMIN can access */}
                  <Route path="/admin" element={<AdminLayout />}>
                    <Route index element={<AdminDashboardPage />} />
                    <Route path="doctors" element={<AdminDoctorManagementPage />} />
                    <Route path="users" element={<AdminUserManagementPage />} />
                    <Route path="health-records" element={<AdminHealthRecordManagement />} />
                    <Route path="appointments" element={<div>Admin Appointments Page</div>} />
                    <Route path="reports" element={<div>Admin Reports Page</div>} />
                    <Route path="settings" element={<div>Admin Settings Page</div>} />
                    {/* Redirect from /admin to /admin/dashboard */}
                    <Route index element={<Navigate to="/admin/dashboard" replace />} />
                  </Route>
                  
                  {/* Doctor specific routes - Allows DOCTOR and ADMIN */}
                  <Route element={<RoleProtectedRoute allowedRoles={[UserRole.DOCTOR, UserRole.ADMIN]} />}>
                    <Route path="doctor">
                      <Route path="patients" element={<div>Doctor's Patients Page</div>} />
                      <Route path="schedule" element={<div>Doctor's Schedule Page</div>} />
                    </Route>
                  </Route>
                  
                  {/* Catch all route - 404 */}
                  <Route path="*" element={<div>404 - Page Not Found</div>} />
                </Route>
              </Routes>
            </ThemeModeProvider>
          </AuthProvider>
        </CachedDataProvider>
      </UIStateProvider>
    </UserPreferencesProvider>
  );
}

export default App;