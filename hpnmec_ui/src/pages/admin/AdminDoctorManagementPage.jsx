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
  Chip, // Import Chip
  CircularProgress,
  Alert
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete'; // Fixed import path here
import { useAuth } from '../../hooks/useAuth'; // Import useAuth
import { isAdmin } from '../../utils/authUtils'; // Import isAdmin
import doctorService from '../../services/doctorService'; // Assuming doctorService exists

// Mock data - replace with actual data fetching
const mockDoctors = [
  { id: 1, user_id: 2, full_name: 'Dr. Tran Thi B', specialization: 'Cardiology', license_number: 'DOC002', is_available: true, rating: 4.5, total_ratings: 10 },
  { id: 2, user_id: 6, full_name: 'Dr. Vo Thi F', specialization: 'Neurology', license_number: 'DOC006', is_available: false, rating: 4.8, total_ratings: 15 },
  // Add more mock doctors as needed
];

function AdminDoctorManagementPage() {
  const { user } = useAuth(); // Get user from auth context
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    const fetchDoctors = async () => {
      try {
        setLoading(true);
        setError(null);
        // Replace with actual API call using doctorService
        // const data = await doctorService.getAllDoctors({ page: page + 1, limit: rowsPerPage });
        // setDoctors(data.items || data); // Adjust based on API response structure
        // setTotalCount(data.total || data.length); // Adjust based on API response structure
        
        // Using mock data for now
        setTimeout(() => {
          setDoctors(mockDoctors);
          setLoading(false);
        }, 1000);
      } catch (err) {
        setError('Lỗi tải danh sách bác sĩ.'); // Vietnamese error
        console.error(err);
        setLoading(false);
      }
    };
    fetchDoctors();
  }, [page, rowsPerPage]); // Refetch when pagination changes

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleEditClick = (doctorId) => {
      // Logic to open edit modal/form
      console.log("Edit doctor:", doctorId);
      // Example: navigate(`/admin/doctors/edit/${doctorId}`); or open a modal
  };

  const handleDeleteClick = (doctor) => {
      // Logic to open delete confirmation dialog
      console.log("Delete doctor:", doctor);
      // Example: setDoctorToDelete(doctor); setOpenDeleteDialog(true);
  };

  const handleAddClick = () => {
      // Logic to open add modal/form
      console.log("Add new doctor");
      // Example: navigate('/admin/doctors/add'); or open a modal
  };

  // Calculate empty rows for pagination
  const emptyRows = page > 0 ? Math.max(0, (1 + page) * rowsPerPage - doctors.length) : 0;

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" fontWeight={600}>
          Quản lý Bác sĩ
        </Typography>
        {/* Conditionally render Add button only for admins */}
        {isAdmin(user) && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            sx={{ borderRadius: 2 }}
            onClick={handleAddClick} // Add onClick handler
          >
            Thêm Bác sĩ
          </Button>
        )}
      </Box>

      {/* Alert for success/error messages */}
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      {/* Add success alert if needed */}

      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        {/* Add Search/Filter UI here if needed */}

        <TableContainer component={Paper} elevation={0}>
          {loading ? (
            <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress />
            </Box>
          ) : (
            <Table sx={{ minWidth: 650 }} aria-label="doctor management table">
              <TableHead>
                <TableRow sx={{ '& th': { fontWeight: 'bold' } }}>
                  <TableCell>ID</TableCell>
                  <TableCell>Họ tên</TableCell>
                  <TableCell>Chuyên khoa</TableCell>
                  <TableCell>Số giấy phép</TableCell>
                  <TableCell>Trạng thái</TableCell>
                  <TableCell>Đánh giá</TableCell>
                  {/* Conditionally render Actions column header only for admins */}
                  {isAdmin(user) && <TableCell align="right">Hành động</TableCell>}
                </TableRow>
              </TableHead>
              <TableBody>
                {doctors.length === 0 && !loading ? (
                  <TableRow>
                    <TableCell colSpan={isAdmin(user) ? 7 : 6} align="center">
                      Không tìm thấy bác sĩ nào.
                    </TableCell>
                  </TableRow>
                ) : (
                  doctors.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((doc) => (
                    <TableRow key={doc.id} hover sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                      <TableCell component="th" scope="row">{doc.id}</TableCell>
                      <TableCell>{doc.full_name}</TableCell>
                      <TableCell>{doc.specialization}</TableCell>
                      <TableCell>{doc.license_number}</TableCell>
                      <TableCell>
                        <Chip
                          label={doc.is_available ? 'Sẵn sàng' : 'Không sẵn sàng'}
                          color={doc.is_available ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{doc.rating?.toFixed(1) || 'N/A'} ({doc.total_ratings || 0})</TableCell>
                      {/* Conditionally render Actions column cells only for admins */}
                      {isAdmin(user) && (
                        <TableCell align="right">
                          <IconButton 
                            color="primary" 
                            size="small" 
                            title="Chỉnh sửa"
                            onClick={() => handleEditClick(doc.id)} // Add onClick handler
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                          <IconButton 
                            color="error" 
                            size="small" 
                            title="Xóa"
                            onClick={() => handleDeleteClick(doc)} // Add onClick handler
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </TableCell>
                      )}
                    </TableRow>
                  ))
                )}
                {emptyRows > 0 && (
                  <TableRow style={{ height: 53 * emptyRows }}>
                    <TableCell colSpan={isAdmin(user) ? 7 : 6} />
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
          {/* TablePagination */}
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            // count={totalCount} // Use total count from API
            count={doctors.length} // Using mock data length for now
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
            labelRowsPerPage="Số dòng mỗi trang:" // Vietnamese label
            labelDisplayedRows={({ from, to, count }) => `${from}-${to} của ${count}`} // Vietnamese label
          />
        </TableContainer>
      </Paper>
      
      {/* Add Dialogs for Add/Edit/Delete here */}
      
    </Container>
  );
}

export default AdminDoctorManagementPage;
