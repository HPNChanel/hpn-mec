import { useState, useEffect, useCallback } from 'react';
// Assuming you have an api utility, e.g., src/utils/api.js
// import api from '../utils/api';

// Placeholder for API calls - replace with your actual API fetching logic
const fakeApiCall = async (endpoint, options) => {
  console.log(`Fetching ${endpoint} with options:`, options);
  await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
  // Simulate different responses based on endpoint
  if (endpoint === '/api/dashboard') {
    return { summary: 'Dashboard data loaded', details: 'More details...' };
  }
  if (endpoint === '/api/health-records') {
      return [{ id: 1, record: 'Record A' }, { id: 2, record: 'Record B' }];
  }
  // Simulate an error for a specific endpoint for testing
  if (endpoint === '/api/error-endpoint') {
      throw new Error("Simulated API Error");
  }
  return { message: `Data for ${endpoint}` };
};


export const useFetchData = (endpoint, options = {}, initialData = null) => {
  const [data, setData] = useState(initialData);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Replace fakeApiCall with your actual API call function
      // const response = await api.get(endpoint, options);
      const response = await fakeApiCall(endpoint, options);
      setData(response);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message || 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, [endpoint, JSON.stringify(options)]); // Use JSON.stringify for dependency array if options is an object

  useEffect(() => {
    if (endpoint) { // Only fetch if endpoint is provided
        fetchData();
    }
  }, [fetchData, endpoint]); // Re-fetch when endpoint or options change

  return { data, isLoading, error, refetch: fetchData };
};
