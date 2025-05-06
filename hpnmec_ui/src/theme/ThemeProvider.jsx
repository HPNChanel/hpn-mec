import { createContext, useMemo, useContext } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useUserPreferences } from '../contexts/UserPreferencesContext';

// Create theme context for toggling
export const ThemeModeContext = createContext({
  toggleColorMode: () => {},
  mode: 'light',
});

// Hook to use theme context
export const useThemeMode = () => useContext(ThemeModeContext);

export function ThemeModeProvider({ children }) {
  const { theme: mode, setTheme } = useUserPreferences();

  // Mode toggle function uses user preferences context
  const toggleColorMode = () => {
    setTheme(prevMode => (prevMode === 'light' ? 'dark' : 'light'));
  };

  // Theme configuration
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: '#1976d2',
            light: '#42a5f5',
            dark: '#1565c0',
            contrastText: '#fff',
          },
          secondary: {
            main: '#00bcd4',
            light: '#4dd0e1',
            dark: '#008394',
            contrastText: '#fff',
          },
          background: {
            default: mode === 'light' ? '#f5f7fa' : '#121212',
            paper: mode === 'light' ? '#fff' : '#1e1e1e',
          },
          success: { main: '#2e7d32' },
          warning: { main: '#ed6c02' },
          error: { main: '#d32f2f' },
          info: { main: '#0288d1' },
          divider: mode === 'light' ? '#e0e0e0' : '#333',
        },
        spacing: 8, // base spacing unit
        typography: {
          fontFamily: [
            'Be Vietnam Pro',
            'Roboto',
            'Noto Sans',
            'Arial',
            'sans-serif',
          ].join(','),
          h1: { fontWeight: 700, fontSize: '2.5rem', lineHeight: 1.2, letterSpacing: 0.5 },
          h2: { fontWeight: 600, fontSize: '2rem', lineHeight: 1.25, letterSpacing: 0.3 },
          h3: { fontWeight: 600, fontSize: '1.5rem', lineHeight: 1.3 },
          h4: { fontWeight: 500, fontSize: '1.25rem', lineHeight: 1.35 },
          h5: { fontWeight: 500, fontSize: '1.1rem', lineHeight: 1.4 },
          h6: { fontWeight: 500, fontSize: '1rem', lineHeight: 1.4 },
          subtitle1: { fontWeight: 500, fontSize: '1rem' },
          subtitle2: { fontWeight: 500, fontSize: '0.95rem' },
          body1: { fontWeight: 400, fontSize: '1rem', lineHeight: 1.5 },
          body2: { fontWeight: 400, fontSize: '0.95rem', lineHeight: 1.43 },
          button: {
            fontWeight: 600,
            textTransform: 'none',
            letterSpacing: 0.2,
          },
          caption: { fontSize: '0.85rem', color: '#888' },
        },
        shape: {
          borderRadius: 12,
        },
        components: {
          MuiCssBaseline: {
            styleOverrides: {
              body: {
                scrollbarWidth: 'thin',
                '&::-webkit-scrollbar': {
                  width: '8px',
                  height: '8px',
                },
                '&::-webkit-scrollbar-thumb': {
                  backgroundColor: mode === 'light' ? '#bbbbbb' : '#666666',
                  borderRadius: '4px',
                },
              },
            },
          },
          MuiButton: {
            styleOverrides: {
              root: {
                borderRadius: 12,
                boxShadow: 'none',
                fontWeight: 600,
                fontSize: '1.05rem',
                letterSpacing: 0.2,
                transition: 'box-shadow 0.2s',
                '&:hover': {
                  boxShadow: '0 2px 8px 0 rgba(25, 118, 210, 0.10)',
                },
              },
              contained: {
                boxShadow: '0 2px 8px 0 rgba(25, 118, 210, 0.10)',
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                borderRadius: 14,
                boxShadow: mode === 'light'
                  ? '0 2px 12px 0 rgba(25, 118, 210, 0.06)'
                  : '0 2px 12px 0 rgba(0,0,0,0.25)',
              },
            },
          },
          MuiCard: {
            styleOverrides: {
              root: {
                borderRadius: 16,
                boxShadow: mode === 'light'
                  ? '0 2px 12px 0 rgba(25, 118, 210, 0.08)'
                  : '0 2px 12px 0 rgba(0,0,0,0.25)',
              },
            },
          },
          MuiTableCell: {
            styleOverrides: {
              head: {
                fontWeight: 600,
                background: mode === 'light' ? '#f5f7fa' : '#23272b',
                fontSize: '1rem',
              },
              body: {
                fontSize: '0.98rem',
              },
            },
          },
          MuiChip: {
            styleOverrides: {
              root: {
                fontWeight: 500,
                borderRadius: 8,
              },
            },
          },
          MuiAlert: {
            styleOverrides: {
              root: {
                borderRadius: 12,
                fontSize: '1rem',
              },
            },
          },
        },
      }),
    [mode]
  );

  const contextValue = useMemo(
    () => ({
      toggleColorMode,
      mode,
    }),
    [mode]
  );

  return (
    <ThemeModeContext.Provider value={contextValue}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
}