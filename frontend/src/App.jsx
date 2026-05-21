import { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, X, Activity, ShieldCheck, ShieldAlert, Image as ImageIcon } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [analysisMode, setAnalysisMode] = useState('media'); // 'media' or 'certificate'

  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (selectedFile) => {
    const isImage = selectedFile.type.startsWith('image/');
    const isVideo = selectedFile.type.startsWith('video/');

    setFile(selectedFile);
    setError(null);
    setResult(null);

    // Create preview
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const analyzeMedia = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    const isVideo = file.type.startsWith('video/');
    const endpoint = analysisMode === 'certificate'
      ? '/analyze-certificate/'
      : (isVideo ? '/analyze-video/' : '/analyze-image/');

    try {
      const response = await axios.post(`http://localhost:8001${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <div className="logo-container">
          <Activity size={32} color="#38bdf8" />
          <h1>Deepfake Detector</h1>
        </div>
        <p className="subtitle">AI-powered system to identify real vs AI-generated media</p>
      </div>

      <div className="main-content">
        <div className="upload-section">
          <div className="mode-toggle">
            <button
              className={`mode-btn ${analysisMode === 'media' ? 'active' : ''}`}
              onClick={() => { setAnalysisMode('media'); clearFile(); }}
            >
              Media (Face/Deepfake)
            </button>
            <button
              className={`mode-btn ${analysisMode === 'certificate' ? 'active' : ''}`}
              onClick={() => { setAnalysisMode('certificate'); clearFile(); }}
            >
              Certificate (OCR)
            </button>
          </div>

          {!preview ? (
            <div
              className={`upload-area ${dragActive ? "drag-active" : ""}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <UploadCloud size={64} className="upload-icon" />
              <div className="upload-text">Click or drag and drop to upload</div>
              <div className="upload-subtext">
                {analysisMode === 'certificate' ? 'Supports any file type (Images, PDFs, etc.)' : 'Supports any file type'}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                className="file-input"
                accept="*"
                onChange={handleChange}
              />
            </div>
          ) : (
            <div className="preview-container">
              <button className="remove-btn" onClick={clearFile}>
                <X size={16} />
              </button>

              {file.type.startsWith('video/') ? (
                <video
                  src={preview}
                  className="preview-media"
                  controls
                />
              ) : file.type.startsWith('image/') ? (
                <img
                  src={preview}
                  alt="Preview"
                  className="preview-media"
                />
              ) : (
                <div className="preview-media" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1e293b', flexDirection: 'column' }}>
                  <Activity size={64} color="#94a3b8" />
                  <span style={{ marginTop: '16px', color: '#cbd5e1', fontWeight: 'bold' }}>{file.name}</span>
                </div>
              )}
            </div>
          )}

          {error && (
            <div style={{ color: '#ef4444', marginTop: '16px', textAlign: 'center' }}>
              {error}
            </div>
          )}

          <button
            className="analyze-btn"
            onClick={analyzeMedia}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <Activity size={20} className="spinner" /> Analyzing...
              </>
            ) : (
              'Analyze Media'
            )}
          </button>
        </div>

        <div className="results-section">
          {!result && !loading && (
            <div className="empty-results">
              <ImageIcon size={48} />
              <p>Upload and analyze media to see results here.</p>
            </div>
          )}

          {loading && (
            <div className="empty-results">
              <Activity size={48} className="spinner" color="#38bdf8" />
              <p>Detecting faces and analyzing textures...</p>
            </div>
          )}

          {result && (
            <div className="results-content">
              <div className={`prediction-card ${result.prediction.includes('Real') ? 'real' : 'fake'}`}>
                <div className="prediction-title">Analysis Result</div>
                <h2 className="prediction-result">
                  {result.prediction.includes('Real') ? (
                    <><ShieldCheck size={32} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} /> {result.prediction}</>
                  ) : (
                    <><ShieldAlert size={32} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} /> {result.prediction}</>
                  )}
                </h2>
              </div>

              <div className="confidence-bar-container">
                <div className="confidence-label">
                  <span>{analysisMode === 'certificate' ? 'Fake Likelihood' : 'Deepfake Likelihood'}</span>
                  <span>{result.fake_probability}%</span>
                </div>
                <div className="bar-bg">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${result.fake_probability}%`,
                      background: result.prediction.includes('Real') ? '#4ade80' : '#f87171'
                    }}
                  ></div>
                </div>
              </div>

              {result.reason && (
                <div style={{ padding: '16px', background: 'rgba(56, 189, 248, 0.1)', borderRadius: '12px', borderLeft: '4px solid #38bdf8', marginTop: '-8px' }}>
                  <p style={{ margin: 0, fontSize: '0.95rem', color: '#e2e8f0', lineHeight: '1.5' }}>
                    <strong>Why?</strong> {result.reason}
                  </p>
                </div>
              )}

              <div className="details-box">
                <div className="detail-item">
                  <span className="detail-label">Media Type</span>
                  <span className="detail-value">{file?.type.split('/')[0].toUpperCase()}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">File Size</span>
                  <span className="detail-value">{(file?.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
                {result.frames_analyzed && (
                  <div className="detail-item">
                    <span className="detail-label">Frames Analyzed</span>
                    <span className="detail-value">{result.frames_analyzed}</span>
                  </div>
                )}
                {result.face_box && (
                  <div className="detail-item">
                    <span className="detail-label">Face Detected</span>
                    <span className="detail-value">Yes</span>
                  </div>
                )}
                {result.extracted_text && (
                  <div className="detail-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
                    <span className="detail-label">Extracted Text Preview</span>
                    <span className="detail-value" style={{ fontSize: '0.85rem', color: '#cbd5e1', fontStyle: 'italic', background: 'rgba(0,0,0,0.3)', padding: '8px', borderRadius: '6px', width: '100%', boxSizing: 'border-box' }}>
                      "{result.extracted_text}"
                    </span>
                  </div>
                )}
                {result.detected_document_type && (
                  <div className="detail-item">
                    <span className="detail-label">Document Type</span>
                    <span className="detail-value" style={{ color: '#38bdf8', fontWeight: 'bold' }}>{result.detected_document_type}</span>
                  </div>
                )}
                {result.reason && (
                  <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(56, 189, 248, 0.1)', borderRadius: '12px', borderLeft: '4px solid #38bdf8' }}>
                    <span style={{ fontSize: '0.8rem', color: '#38bdf8', fontWeight: 'bold', display: 'block', marginBottom: '4px' }}>FORENSIC EVIDENCE REPORT:</span>
                    <ul style={{ margin: 0, paddingLeft: '20px', color: '#cbd5e1', fontSize: '0.85rem' }}>
                      {result.reason.split('.').map((point, idx) => (
                        point.trim() && <li key={idx}>{point.trim()}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.structural_warnings && result.structural_warnings.length > 0 && (
                  <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(234, 179, 8, 0.1)', borderRadius: '12px', borderLeft: '4px solid #eab308' }}>
                    <span style={{ fontSize: '0.8rem', color: '#eab308', fontWeight: 'bold', display: 'block', marginBottom: '4px' }}>VERIFICATION WARNINGS:</span>
                    <ul style={{ margin: 0, paddingLeft: '20px', color: '#cbd5e1', fontSize: '0.85rem' }}>
                      {result.structural_warnings.map((warn, idx) => (
                        <li key={idx}>{warn}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
