import React, { createContext, useState, useContext, useMemo } from 'react';

const UserPreferencesContext = createContext();

export const UserPreferencesProvider = ({ children }) => {
  const [theme, setTheme] = useState('light'); // User's preferred theme
  const [language, setLanguage] = useState('en'); // Default language

  // Only store and provide theme preference, not theme logic
  const value = useMemo(() => ({
    theme,
    setTheme,
    language,
    setLanguage,
  }), [theme, language]);

  return (
    <UserPreferencesContext.Provider value={value}>
      {children}
    </UserPreferencesContext.Provider>
  );
};

export const useUserPreferences = () => {
  const context = useContext(UserPreferencesContext);
  if (context === undefined) {
    throw new Error('useUserPreferences must be used within a UserPreferencesProvider');
  }
  return context;
};
