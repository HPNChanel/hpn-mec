import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Box,
  Button,
  IconButton,
  TextField,
  InputAdornment,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  Alert
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import LockIcon from '@mui/icons-material/Lock';
import LockOpenIcon from '@mui/icons-material/LockOpen';
import AddIcon from '@mui/icons-material/Add';
import { UserRole } from '../../types';

// This would come from your API in a real app
const mockUsers = [
  { id: 1, username: 'user1', email: 'user1@example.com', full_name: 'Nguyen Van A', role: UserRole.PATIENT, is_active: true, date_joined: '2023-01-15' },
  { id: 2, username: 'user2', email: 'user2@example.com', full_name: 'Tran Thi B', role: UserRole.DOCTOR, is_active: true, date_joined: '2023-02-20' },
  { id: 3, username: 'user3', email: 'user3@example.com', full_name: 'Le Van C', role: UserRole.ADMIN, is_active: true, date_joined: '2023-03-10' },
  { id: 4, username: 'user4', email: 'user4@example.com', full_name: 'Pham Thi D', role: UserRole.PATIENT, is_active: false, date_joined: '2023-04-05' },
  { id: 5, username: 'user5', email: 'user5@example.com', full_name: 'Hoang Van E', role: UserRole.PATIENT, is_active: true, date_joined: '2023-05-12' },
  { id: 6, username: 'user6', email: 'user6@example.com', full_name: 'Vo Thi F', role: UserRole.DOCTOR, is_active: true, date_joined: '2023-06-18' },
  { id: 7, username: 'user7', email: 'user7@example.com', full_name: 'Nguyen Van G', role: UserRole.PATIENT, is_active: true, date_joined: '2023-07-22' },
  { id: 8, username: 'user8', email: 'user8@example.com', full_name: 'Tran Van H', role: UserRole.PATIENT, is_active: false, date_joined: '2023-08-30' },
];

function AdminUserManagementPage() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [searchQuery, setSearchQuery] = useState('');
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [userToDelete, setUserToDelete] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  useEffect(() => {
    // This would be an API call in a real app
    setTimeout(() => {
      setUsers(mockUsers);
      setLoading(false);
    }, 1000);
  }, []);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
    setPage(0);
  };

  const handleDeleteClick = (user) => {
    setUserToDelete(user);
    setOpenDeleteDialog(true);
  };

  const handleDeleteConfirm = () => {
    // In a real app, this would call your API to delete the user
    setUsers(users.filter(user => user.id !== userToDelete.id));
    setOpenDeleteDialog(false);
    setSuccessMessage(`Người dùng ${userToDelete.full_name} đã được xóa thành công.`);
    
    // Clear success message after 3 seconds
    setTimeout(() => {
      setSuccessMessage(null);
    }, 3000);
  };

  const handleToggleActive = (userId, currentStatus) => {
    // In a real app, this would call your API to toggle the user's active status
    setUsers(users.map(user => {
      if (user.id === userId) {
        return { ...user, is_active: !currentStatus };
      }
      return user;
    }));
    
    const user = users.find(u => u.id === userId);
    const newStatus = !currentStatus;
    setSuccessMessage(`Người dùng ${user.full_name} đã được ${newStatus ? 'kích hoạt' : 'vô hiệu hóa'} thành công.`);
    
    // Clear success message after 3 seconds
    setTimeout(() => {
      setSuccessMessage(null);
    }, 3000);
  };

  // Filter users based on search query
  const filteredUsers = users.filter(user => {
    const searchLower = searchQuery.toLowerCase();
    return (
      user.username.toLowerCase().includes(searchLower) ||
      user.email.toLowerCase().includes(searchLower) ||
      user.full_name.toLowerCase().includes(searchLower) ||
      user.role.toLowerCase().includes(searchLower)
    );
  });

  // Get users for current page
  const paginatedUsers = filteredUsers.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  const getRoleColor = (role) => {
    switch (role) {
      case UserRole.ADMIN:
        return 'error';
      case UserRole.DOCTOR:
        return 'info';
      case UserRole.PATIENT:
        return 'success';
      default:
        return 'default';
    }
  };

  const getRoleLabel = (role) => {
    switch (role) {
      case UserRole.ADMIN:
        return 'Quản trị viên';
      case UserRole.DOCTOR:
        return 'Bác sĩ';
      case UserRole.PATIENT:
        return 'Bệnh nhân';
      default:
        return role;
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" fontWeight={600}>
          Quản lý người dùng
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          sx={{ borderRadius: 2 }}
        >
          Thêm người dùng
        </Button>
      </Box>

      {successMessage && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {successMessage}
        </Alert>
      )}

      <Paper elevation={2} sx={{ p: 3, mb: 4, borderRadius: 3, boxShadow: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <TextField
            label="Tìm kiếm"
            variant="outlined"
            size="small"
            value={searchQuery}
            onChange={handleSearchChange}
            sx={{ width: 300 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />
        </Box>
        <TableContainer component={Paper} elevation={0} sx={{ borderRadius: 2 }}>
          {loading ? (
            <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Box sx={{ p: 3 }}>
              <Alert severity="error">{error}</Alert>
            </Box>
          ) : (
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Tên đăng nhập</TableCell>
                  <TableCell>Họ tên</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Vai trò</TableCell>
                  <TableCell>Trạng thái</TableCell>
                  <TableCell>Ngày tham gia</TableCell>
                  <TableCell align="right">Thao tác</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {paginatedUsers.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      Không tìm thấy người dùng nào.
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedUsers.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell>{user.id}</TableCell>
                      <TableCell>{user.username}</TableCell>
                      <TableCell>{user.full_name}</TableCell>
                      <TableCell>{user.email}</TableCell>
                      <TableCell>
                        <Chip
                          label={getRoleLabel(user.role)}
                          color={getRoleColor(user.role)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={user.is_active ? 'Hoạt động' : 'Đã khóa'}
                          color={user.is_active ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{new Date(user.date_joined).toLocaleDateString('vi-VN')}</TableCell>
                      <TableCell align="right">
                        <IconButton color="primary" size="small" title="Chỉnh sửa">
                          <EditIcon fontSize="small" />
                        </IconButton>
                        <IconButton 
                          color={user.is_active ? 'warning' : 'success'} 
                          size="small"
                          onClick={() => handleToggleActive(user.id, user.is_active)}
                          title={user.is_active ? 'Khóa tài khoản' : 'Mở khóa tài khoản'}
                        >
                          {user.is_active ? <LockIcon fontSize="small" /> : <LockOpenIcon fontSize="small" />}
                        </IconButton>
                        <IconButton 
                          color="error" 
                          size="small"
                          onClick={() => handleDeleteClick(user)}
                          title="Xóa tài khoản"
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          )}
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={filteredUsers.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
            labelRowsPerPage="Hiển thị:"
            labelDisplayedRows={({ from, to, count }) => `${from}-${to} của ${count}`}
            sx={{ borderTop: 1, borderColor: 'divider' }}
          />
        </TableContainer>
      </Paper>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={openDeleteDialog}
        onClose={() => setOpenDeleteDialog(false)}
        PaperProps={{ sx: { borderRadius: 3 } }}
      >
        <DialogTitle sx={{ fontWeight: 700 }}>Xác nhận xóa người dùng</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Bạn có chắc chắn muốn xóa người dùng "{userToDelete?.full_name}"? Hành động này không thể hoàn tác.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDeleteDialog(false)} color="primary" variant="outlined">
            Hủy bỏ
          </Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Xóa
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default AdminUserManagementPage;
