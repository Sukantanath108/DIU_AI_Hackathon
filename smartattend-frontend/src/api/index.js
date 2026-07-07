import axios from 'axios';

// The backend is hosted on port 8000
const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const attendApi = {
  // Create attendance session
  createSession: async (section, subject, teacherId) => {
    const response = await api.post('/attend/session', {
      section,
      subject,
      teacher_id: teacherId,
    });
    return response.data;
  },

  // Upload 3 classroom photos
  uploadPhotos: async (sessionId, leftBlob, centerBlob, rightBlob) => {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('front_left', leftBlob, 'left.jpg');
    formData.append('front_center', centerBlob, 'center.jpg');
    formData.append('front_right', rightBlob, 'right.jpg');

    const response = await api.post('/attend/photos', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Manual teacher correction
  overrideStatus: async (sessionId, studentId, status) => {
    const response = await api.patch('/attend/override', {
      session_id: sessionId,
      student_id: studentId,
      status: status,
    });
    return response.data;
  },

  // Export report URL helpers
  getExportUrl: (sessionId, format) => {
    return `${API_BASE_URL}/attend/export/${sessionId}?format=${format}`;
  },
};

export default api;
