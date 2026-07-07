import React, { useRef, useState, useEffect, useCallback } from 'react';
import { attendApi } from '../api';

const CaptureScreen = ({ sessionData, onResultsReceived, onReset }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const mountedRef = useRef(true); // Track if component is still mounted
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [capturedImages, setCapturedImages] = useState({
    left: null,
    center: null,
    right: null,
  });
  const [currentStep, setCurrentStep] = useState('left');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [laptopMode, setLaptopMode] = useState(() => {
    return localStorage.getItem('smartattend_laptop_mode') === 'true';
  });

  const guidePrompts = laptopMode ? {
    left: 'Position yourself on the LEFT side of frame — tap Capture',
    center: 'Center yourself in frame — tap Capture',
    right: 'Position yourself on the RIGHT side of frame — tap Capture',
    done: 'All 3 positions captured! Proceed to run AI attendance fusion.',
  } : {
    left: 'Aim at the FRONT-LEFT side of the classroom to capture students in those columns.',
    center: 'Aim at the FRONT-CENTER of the classroom. Ensure the middle rows are visible.',
    right: 'Aim at the FRONT-RIGHT side of the classroom to cover the remaining students.',
    done: 'Multi-angle photos captured successfully! Proceed to run AI attendance fusion.',
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      console.log('[SmartAttend] Stopping camera stream...');
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    // Do NOT null out videoRef.srcObject here — React owns that DOM node
    setCameraActive(false);
  }, []);

  const startCamera = useCallback(async (useLaptopMode) => {
    setCameraError('');
    stopCamera();

    const constraints = {
      video: useLaptopMode ? {
        facingMode: 'user',
        width: { ideal: 1280 },
        height: { ideal: 720 },
      } : {
        facingMode: 'environment',
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    };

    try {
      console.log('[SmartAttend] Requesting getUserMedia with constraints:', JSON.stringify(constraints.video));
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('[SmartAttend] getUserMedia succeeded. Tracks:', mediaStream.getTracks().map(t => t.label));

      // Check if component is still mounted after the async call
      if (!mountedRef.current) {
        console.warn('[SmartAttend] Component unmounted during getUserMedia. Stopping acquired stream.');
        mediaStream.getTracks().forEach(t => t.stop());
        return;
      }

      streamRef.current = mediaStream;

      // The <video> element is ALWAYS in the DOM now (hidden via CSS when not active).
      // So videoRef.current should always be valid here.
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        console.log('[SmartAttend] Attached stream to <video> element. Setting cameraActive=true');
        setCameraActive(true);
      } else {
        // This should never happen now, but log it just in case
        console.error('[SmartAttend] BUG: videoRef.current is null even though <video> is always rendered.');
        mediaStream.getTracks().forEach(t => t.stop());
        streamRef.current = null;
        setCameraError('Internal error: video element not found. Please refresh the page.');
      }
    } catch (err) {
      console.error('[SmartAttend] getUserMedia FAILED:', err.name, err.message);
      streamRef.current = null;
      setCameraActive(false);
      setCameraError(`Camera error: ${err.name} — ${err.message}. Check browser permissions.`);
    }
  }, [stopCamera]);

  // Mount: start camera once. Cleanup: stop camera + mark unmounted.
  useEffect(() => {
    mountedRef.current = true;
    startCamera(laptopMode);
    return () => {
      mountedRef.current = false;
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist laptop mode to localStorage (no camera restart here — that's in the toggle handler)
  useEffect(() => {
    localStorage.setItem('smartattend_laptop_mode', laptopMode ? 'true' : 'false');
  }, [laptopMode]);

  const handleLaptopModeToggle = useCallback(() => {
    const newMode = !laptopMode;
    setLaptopMode(newMode);
    if (currentStep !== 'done') {
      startCamera(newMode);
    }
  }, [laptopMode, currentStep, startCamera]);

  const capturePhoto = () => {
    if (cameraActive && videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (laptopMode) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      const imageUrl = canvas.toDataURL('image/jpeg', 0.85);

      canvas.toBlob((blob) => {
        saveCapturedImage(blob, imageUrl);
      }, 'image/jpeg', 0.85);
    } else {
      setError('Camera is not active. Please allow camera access in your browser and refresh.');
    }
  };

  const saveCapturedImage = (blob, url) => {
    setCapturedImages((prev) => {
      const updated = { ...prev, [currentStep]: { blob, url } };
      
      if (currentStep === 'left') {
        setCurrentStep('center');
      } else if (currentStep === 'center') {
        setCurrentStep('right');
      } else if (currentStep === 'right') {
        setCurrentStep('done');
        stopCamera();
      }
      
      return updated;
    });
  };

  const resetCaptures = () => {
    setCapturedImages({ left: null, center: null, right: null });
    setCurrentStep('left');
    startCamera(laptopMode);
  };

  const handleProcessAttendance = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await attendApi.uploadPhotos(
        sessionData.sessionId,
        capturedImages.left.blob,
        capturedImages.center.blob,
        capturedImages.right.blob
      );

      if (response.status === 'ok') {
        onResultsReceived(response.data.records);
      } else {
        setError(response.message || 'Failed to process photos');
      }
    } catch (err) {
      setError('AI backend processing failed. Please verify FastAPI is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h2 style={{ fontSize: '16px', fontWeight: 600 }}>{sessionData.subject} — Section {sessionData.section}</h2>
          <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Session ID: #{sessionData.sessionId}</span>
        </div>
        <button className="btn-secondary" onClick={onReset} style={{ width: 'auto', padding: '6px 12px', fontSize: '12px' }}>
          Change Class
        </button>
      </div>

      {/* Laptop Testing Mode Toggle */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        marginBottom: '14px', padding: '8px 14px',
        background: laptopMode ? 'rgba(59,130,246,0.08)' : 'rgba(255,255,255,0.03)',
        border: `1px solid ${laptopMode ? 'rgba(59,130,246,0.25)' : 'rgba(255,255,255,0.06)'}`,
        borderRadius: '10px', cursor: 'pointer', userSelect: 'none',
        transition: 'all 0.2s ease'
      }} onClick={handleLaptopModeToggle}>
        <div style={{
          width: '36px', height: '20px', borderRadius: '10px',
          background: laptopMode ? 'var(--accent-blue)' : 'rgba(100,116,139,0.3)',
          position: 'relative', transition: 'background 0.2s ease'
        }}>
          <div style={{
            width: '16px', height: '16px', borderRadius: '50%',
            background: '#fff', position: 'absolute', top: '2px',
            left: laptopMode ? '18px' : '2px',
            transition: 'left 0.2s ease', boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
          }} />
        </div>
        <span style={{ fontSize: '12px', fontWeight: 500, color: laptopMode ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>
          💻 Laptop testing mode {laptopMode ? 'ON' : 'OFF'}
        </span>
      </div>

      {(error || cameraError) && (
        <div style={{ 
          background: 'rgba(245, 158, 11, 0.08)', 
          border: '1px solid rgba(245, 158, 11, 0.25)', 
          color: 'var(--accent-amber)', 
          borderRadius: '12px', 
          padding: '10px 14px', 
          fontSize: '12px', 
          marginBottom: '16px' 
        }}>
          {error || cameraError}
        </div>
      )}

      {/* Guide Banner */}
      <div className="amber-banner" style={{ background: currentStep === 'done' ? 'rgba(16, 185, 129, 0.12)' : undefined, border: currentStep === 'done' ? '1px solid rgba(16, 185, 129, 0.25)' : undefined, color: currentStep === 'done' ? 'var(--accent-green)' : undefined }}>
        {currentStep !== 'done' && <span className="guide-step-tag">Step {currentStep === 'left' ? 1 : currentStep === 'center' ? 2 : 3} of 3</span>}
        <span style={{ flex: 1 }}>{guidePrompts[currentStep]}</span>
      </div>

      {/* Camera Stream / Live View
           CRITICAL FIX: The <video> element is ALWAYS rendered in the DOM.
           When camera is not active, it's hidden via CSS (display:none).
           This ensures videoRef.current is never null when getUserMedia resolves.
           The previous bug was: <video> was conditionally rendered only when
           cameraActive===true, but cameraActive was only set to true AFTER
           attaching the stream to videoRef — which required <video> to exist.
           Chicken-and-egg deadlock. */}
      {currentStep !== 'done' && (
        <div className="camera-preview-container">
          {/* Video element is ALWAYS in DOM — hidden when not active */}
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted
            className="camera-stream"
            style={{
              display: cameraActive ? 'block' : 'none',
              ...(laptopMode ? { transform: 'scaleX(-1)' } : {})
            }}
          />
          
          {/* Placeholder shown only when camera is not yet active */}
          {!cameraActive && (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              <span style={{ fontSize: '36px', marginBottom: '8px' }}>📸</span>
              <p style={{ fontSize: '13px' }}>
                {cameraError ? 'Camera access denied or unavailable.' : 'Requesting camera access...'}
              </p>
            </div>
          )}
          
          <div className="camera-guide-overlay">
            <span className="guide-step-tag" style={{ background: 'rgba(15,23,42,0.6)' }}>
              Angle: {currentStep.toUpperCase()}
            </span>
          </div>
        </div>
      )}

      {/* Grid status of captured items */}
      <div className="capture-status-grid">
        {['left', 'center', 'right'].map((angle) => (
          <div 
            key={angle}
            className={`thumbnail-slot ${capturedImages[angle] ? 'captured' : ''} ${currentStep === angle ? 'active' : ''}`}
          >
            {capturedImages[angle] ? (
              <img src={capturedImages[angle].url} alt={angle} className="thumbnail-img" />
            ) : (
              <span style={{ fontSize: '18px', opacity: 0.3 }}>📸</span>
            )}
            <span className="thumbnail-label">{angle}</span>
          </div>
        ))}
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Action buttons */}
      {currentStep !== 'done' ? (
        <button className="btn-primary" onClick={capturePhoto} disabled={!cameraActive}>
          {cameraActive ? `Capture ${currentStep.toUpperCase()} Photo` : 'Waiting for camera...'}
        </button>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <button 
            className="btn-primary" 
            onClick={handleProcessAttendance}
            disabled={loading}
            style={{ background: 'linear-gradient(135deg, var(--accent-green) 0%, #059669 100%)', boxShadow: '0 4px 15px rgba(16, 185, 129, 0.35)' }}
          >
            {loading ? 'AI Fusion Processing...' : 'Process Attendance with AI'}
          </button>
          <button className="btn-secondary" onClick={resetCaptures} disabled={loading}>
            Retake Photos
          </button>
        </div>
      )}
    </div>
  );
};

export default CaptureScreen;
