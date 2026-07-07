import React, { useState } from 'react';
import { attendApi } from '../api';

const ResultsScreen = ({ sessionData, initialRecords, onReset }) => {
  const [records, setRecords] = useState(initialRecords);
  const [updatingId, setUpdatingId] = useState(null);
  const [error, setError] = useState('');

  // Stats calculation
  const total = records.length;
  const presentCount = records.filter((r) => r.status === 'present').length;
  const lowConfCount = records.filter((r) => r.status === 'low_confidence').length;
  const absentCount = records.filter((r) => r.status === 'absent').length;

  const handleOverride = async (studentId, currentStatus) => {
    // Toggle status: if present/low_confidence -> make absent. If absent -> make present.
    const targetStatus = (currentStatus === 'present' || currentStatus === 'low_confidence') ? 'absent' : 'present';
    
    setUpdatingId(studentId);
    setError('');

    try {
      const response = await attendApi.overrideStatus(sessionData.sessionId, studentId, targetStatus);
      if (response.status === 'ok') {
        // Update local React state
        setRecords((prev) =>
          prev.map((r) =>
            r.student_id === studentId
              ? { ...r, status: targetStatus, overridden: 1, confidence: targetStatus === 'present' ? 1.0 : 0.0 }
              : r
          )
        );
      } else {
        setError(response.message || 'Failed to update attendance status');
      }
    } catch (err) {
      setError('Failed to reach backend to save the override.');
      console.error(err);
    } finally {
      setUpdatingId(null);
    }
  };

  const handleExport = (format) => {
    const url = attendApi.getExportUrl(sessionData.sessionId, format);
    window.open(url, '_blank');
  };

  return (
    <div className="card" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h2 style={{ fontSize: '16px', fontWeight: 600 }}>{sessionData.subject} Results</h2>
          <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Section {sessionData.section} | Session #{sessionData.sessionId}</span>
        </div>
        <button className="btn-secondary" onClick={onReset} style={{ width: 'auto', padding: '6px 12px', fontSize: '12px' }}>
          Done
        </button>
      </div>

      {error && (
        <div style={{ 
          background: 'rgba(239, 68, 68, 0.12)', 
          border: '1px solid rgba(239, 68, 68, 0.25)', 
          color: '#EF4444', 
          borderRadius: '12px', 
          padding: '10px 14px', 
          fontSize: '12px', 
          marginBottom: '16px' 
        }}>
          {error}
        </div>
      )}

      {/* Summary Statistics */}
      <div className="results-header-summary">
        <div className="stat-item">
          <div className="stat-val" style={{ color: 'var(--text-primary)' }}>{total}</div>
          <div className="stat-lbl">Enrolled</div>
        </div>
        <div className="stat-item">
          <div className="stat-val present">{presentCount}</div>
          <div className="stat-lbl">Present</div>
        </div>
        {lowConfCount > 0 && (
          <div className="stat-item">
            <div className="stat-val low-confidence">{lowConfCount}</div>
            <div className="stat-lbl">Amber</div>
          </div>
        )}
        <div className="stat-item">
          <div className="stat-val absent">{absentCount}</div>
          <div className="stat-lbl">Absent</div>
        </div>
      </div>

      {/* Amber Review Banner */}
      {lowConfCount > 0 && (
        <div className="amber-banner">
          <span>⚠️</span>
          <span style={{ flex: 1 }}>
            <b>{lowConfCount}</b> face(s) matched with low confidence (45%–60%). Please review these flagged entries below.
          </span>
        </div>
      )}

      {/* Exporters Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '20px' }}>
        <button className="btn-secondary" onClick={() => handleExport('csv')} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '13px' }}>
          <span>📥</span> Export CSV
        </button>
        <button className="btn-secondary" onClick={() => handleExport('pdf')} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '13px' }}>
          <span>📄</span> Export PDF
        </button>
      </div>

      {/* Student List */}
      <div className="student-list-container">
        {records.map((student) => {
          const isPresent = student.status === 'present';
          const isLowConf = student.status === 'low_confidence';
          const isAbsent = student.status === 'absent';
          const isUpdating = updatingId === student.student_id;
          
          let displayStatus = isPresent ? 'Present' : (isLowConf ? 'Low Confidence' : 'Absent');
          
          return (
            <div 
              key={student.student_id} 
              className={`student-row ${student.status}`}
              style={{ opacity: isUpdating ? 0.6 : 1 }}
            >
              <div className="student-info">
                <div className="student-name">
                  {student.name}
                  {student.confidence > 0 && (
                    <span className={`confidence-badge ${student.status}`}>
                      {(student.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                <div className="student-meta">
                  {student.student_id} 
                  {student.matched_photo && student.matched_photo !== 'none' && (
                    <span style={{ marginLeft: '8px', color: 'var(--text-muted)' }}>
                      Matched: {student.matched_photo} angle
                    </span>
                  )}
                  {student.overridden === 1 && (
                    <span style={{ marginLeft: '8px', color: 'var(--accent-blue)', fontWeight: 500 }}>
                      • Edited
                    </span>
                  )}
                </div>
              </div>

              <div className="status-actions">
                <div 
                  className="status-pill-toggle"
                  onClick={() => !isUpdating && handleOverride(student.student_id, student.status)}
                >
                  <div className={`status-pill-option ${isPresent || isLowConf ? 'active ' + student.status : ''}`}>
                    Present
                  </div>
                  <div className={`status-pill-option ${isAbsent ? 'active absent' : ''}`}>
                    Absent
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ResultsScreen;
