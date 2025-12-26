import axios from 'axios';

const API_BASE_URL =  'http://localhost:5010/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response);
    return response;
  },
  (error) => {
    if (error.code === 'ERR_NETWORK') {
      console.error('Network error - is the Flask backend running on port 5000?');
    }
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const getDrugs = async () => {
  try {
    const response = await api.get('/drugs');
    console.log('getDrugs full response:', response);
    console.log('getDrugs data:', response.data);
    return response;
  } catch (error) {
    console.error('Error fetching drugs:', error);
    throw error;
  }
};

export const getCancerTypes = async () => {
  try {
    const response = await api.get('/cancer-types');
    console.log('getCancerTypes full response:', response);
    console.log('getCancerTypes data:', response.data);
    return response;
  } catch (error) {
    console.error('Error fetching cancer types:', error);
    throw error;
  }
};

export const getMoleculeImage = (smiles) => api.post('/molecule-image', { smiles });

export const predictSynergy = (data) => api.post('/predict', data);

export const batchPredict = (combinations) => api.post('/batch-predict', { combinations });

export const healthCheck = () => api.get('/health');

export default api;
