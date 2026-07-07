import React, { useState } from 'react';
import { attendApi } from '../api';

const ClassSelector = ({ onSessionCreated }) => {
  const [subject, setSubject] = useState('CSE301');
  const [section, setSection] = useState('A');
  const [teacherId, setTeacherId] = useState('T101');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await attendApi.createSession(section, subject, teacherId);
      if (response.status === 'ok') {
        onSessionCreated({
          sessionId: response.data.session_id,
          subject: response.data.subject,
          section: response.data.section,
          teacherId: response.data.teacher_id,
        });
      } else {
        setError(response.message || 'Failed to create session');
      }
    } catch (err) {
      setError('Connection to backend failed. Make sure the FastAPI server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title" style={{ textAlign: 'center', marginBottom: '24px' }}>
        Start Attendance Session
      </div>
      
      {error && (
        <div style={{ 
          background: 'rgba(239, 68, 68, 0.12)', 
          border: '1px solid rgba(239, 68, 68, 0.25)', 
          color: '#EF4444', 
          borderRadius: '12px', 
          padding: '12px', 
          fontSize: '13px', 
          marginBottom: '16px' 
        }}>
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Subject / Course</label>
          <select 
            className="select-input" 
            value={subject} 
            onChange={(e) => setSubject(e.target.value)}
          >
            <option value="CSE301">CSE301 — Artificial Intelligence</option>
            <option value="CSE302">CSE302 — Computer Vision Lab</option>
            <option value="CSE403">CSE403 — Machine Learning</option>
            <option value="MAT201">MAT201 — Linear Algebra</option>
          </select>
        </div>

        <div className="form-group">
          <label>Class Section</label>
          <select 
            className="select-input" 
            value={section} 
            onChange={(e) => setSection(e.target.value)}
          >
            <option value="A">Section A (CSE)</option>
            <option value="B">Section B (CSE)</option>
            <option value="C">Section C (CIS)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Teacher ID Reference</label>
          <input 
            type="text" 
            className="text-input" 
            value={teacherId} 
            onChange={(e) => setTeacherId(e.target.value)}
            required
            placeholder="e.g. T101"
          />
        </div>

        <button 
          type="submit" 
          className="btn-primary" 
          disabled={loading}
          style={{ marginTop: '12px' }}
        >
          {loading ? 'Initializing...' : 'Initialize Session'}
        </button>
      </form>
    </div>
  );
};

export default ClassSelector;
