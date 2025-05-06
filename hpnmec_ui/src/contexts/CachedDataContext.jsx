import React, { createContext, useState, useContext, useMemo, useCallback } from 'react';

const CachedDataContext = createContext();

export const CachedDataProvider = ({ children }) => {
  const [cache, setCache] = useState({}); // Simple key-value cache

  const getCachedData = useCallback((key) => {
    return cache[key]; // Add logic for expiration if needed
  }, [cache]);

  const setCachedData = useCallback((key, data, ttl) => { // ttl = time to live (optional)
    setCache(prev => ({ ...prev, [key]: { data, timestamp: Date.now(), ttl } }));
  }, []);

  const clearCache = useCallback((key) => {
    if (key) {
      setCache(prev => {
        const newCache = { ...prev };
        delete newCache[key];
        return newCache;
      });
    } else {
      setCache({}); // Clear all cache
    }
  }, []);

  // Add logic here to periodically clean up expired cache entries if using TTL

  const value = useMemo(() => ({
    getCachedData,
    setCachedData,
    clearCache,
  }), [getCachedData, setCachedData, clearCache]);

  return (
    <CachedDataContext.Provider value={value}>
      {children}
    </CachedDataContext.Provider>
  );
};

export const useCachedData = () => {
  const context = useContext(CachedDataContext);
  if (context === undefined) {
    throw new Error('useCachedData must be used within a CachedDataProvider');
  }
  return context;
};
