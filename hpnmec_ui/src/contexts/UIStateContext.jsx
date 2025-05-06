import React, { createContext, useState, useContext, useMemo, useCallback } from 'react';

const UIStateContext = createContext();

export const UIStateProvider = ({ children }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [openModals, setOpenModals] = useState({}); // e.g., { profileModal: true, settingsModal: false }

  const toggleSidebar = useCallback(() => {
    setIsSidebarOpen(prev => !prev);
  }, []);

  const openModal = useCallback((modalId) => {
    setOpenModals(prev => ({ ...prev, [modalId]: true }));
  }, []);

  const closeModal = useCallback((modalId) => {
    setOpenModals(prev => ({ ...prev, [modalId]: false }));
  }, []);

  const value = useMemo(() => ({
    isSidebarOpen,
    toggleSidebar,
    openModals,
    openModal,
    closeModal,
  }), [isSidebarOpen, toggleSidebar, openModals, openModal, closeModal]);

  return (
    <UIStateContext.Provider value={value}>
      {children}
    </UIStateContext.Provider>
  );
};

export const useUIState = () => {
  const context = useContext(UIStateContext);
  if (context === undefined) {
    throw new Error('useUIState must be used within a UIStateProvider');
  }
  return context;
};
