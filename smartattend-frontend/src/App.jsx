import React, { useState } from 'react';
import ClassSelector from './pages/ClassSelector';
import CaptureScreen from './pages/CaptureScreen';
import ResultsScreen from './pages/ResultsScreen';

const App = () => {
  const [step, setStep] = useState('select'); // 'select' | 'capture' | 'results'
  const [sessionData, setSessionData] = useState(null);
  const [records, setRecords] = useState([]);

  const handleSessionCreated = (data) => {
    setSessionData(data);
    setStep('capture');
  };

  const handleResultsReceived = (attendanceRecords) => {
    setRecords(attendanceRecords);
    setStep('results');
  };

  const handleReset = () => {
    setSessionData(null);
    setRecords([]);
    setStep('select');
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>SmartAttend AI</h1>
        <p>DIU AI Project Competition MVP • Team Mongolchari</p>
      </header>

      <main style={{ flex: 1 }}>
        {step === 'select' && (
          <ClassSelector onSessionCreated={handleSessionCreated} />
        )}
        
        {step === 'capture' && (
          <CaptureScreen 
            sessionData={sessionData} 
            onResultsReceived={handleResultsReceived} 
            onReset={handleReset} 
          />
        )}
        
        {step === 'results' && (
          <ResultsScreen 
            sessionData={sessionData} 
            initialRecords={records} 
            onReset={handleReset} 
          />
        )}
      </main>

      <footer style={{ 
        textAlign: 'center', 
        padding: '20px 0', 
        fontSize: '11px', 
        color: 'var(--text-muted)',
        fontWeight: 300 
      }}>
        © 2026 Team Mongolchari (CUET). All rights reserved.
      </footer>
    </div>
  );
};

export default App;
