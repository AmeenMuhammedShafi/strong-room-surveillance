import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  // Navigation tabs
  const [activeTab, setActiveTab] = useState('logs') // 'logs', 'enroll', 'users'
  
  // Real-time security metrics (derived from logs or events)
  const [currentStatus, setCurrentStatus] = useState('NORMAL') // 'NORMAL', 'WARNING', 'ALERT'
  const [detectedCount, setDetectedCount] = useState(0)
  const [authCount, setAuthCount] = useState(0)
  const [unknownCount, setUnknownCount] = useState(0)
  const [activeUsers, setActiveUsers] = useState([])
  const [systemLogs, setSystemLogs] = useState([])
  const [logFilter, setLogFilter] = useState('all') // 'all', 'info', 'warning', 'error'
  const [connectionStatus, setConnectionStatus] = useState('connecting') // 'connecting', 'connected', 'disconnected'

  // Enrollment fields
  const [enrollUserId, setEnrollUserId] = useState('')
  const [enrollName, setEnrollName] = useState('')
  const [enrollMode, setEnrollMode] = useState('camera') // 'camera', 'upload'
  const [enrollStatus, setEnrollStatus] = useState({ type: '', message: '' }) // type: 'success', 'error', 'loading'
  const [enrolling, setEnrolling] = useState(false)

  // Local browser camera enrollment state
  const [cameraActive, setCameraActive] = useState(false)
  const [capturedSnaps, setCapturedSnaps] = useState([]) // Array of Blob objects
  const [capturingCount, setCapturingCount] = useState(0)
  
  // File upload state
  const [uploadFiles, setUploadFiles] = useState([]) // Array of File objects
  const [dragActive, setDragActive] = useState(false)

  // Enrolled users list state
  const [enrolledUsers, setEnrolledUsers] = useState([])
  const [serverCameraActive, setServerCameraActive] = useState(true)

  // Refs
  const localVideoRef = useRef(null)
  const localStreamRef = useRef(null)
  const logsEndRef = useRef(null)
  const wsRef = useRef(null)

  // Establish WebSocket connection for real-time security events
  useEffect(() => {
    connectWebSocket()
    
    // Fetch initial server camera status
    fetch(`${API_BASE_URL}/api/camera/status`)
      .then(res => res.json())
      .then(data => setServerCameraActive(data.active))
      .catch(err => console.error('Failed to fetch camera status', err))

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Auto-scroll logs to bottom when new logs arrive
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [systemLogs])

  const connectWebSocket = () => {
    setConnectionStatus('connecting')
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//localhost:8000/ws/logs`
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setConnectionStatus('connected')
      console.log('✓ Security logs WebSocket connected')
    }

    ws.onmessage = (event) => {
      try {
        const logItem = JSON.parse(event.data)
        
        // Append log to list
        setSystemLogs((prev) => {
          // Avoid duplicate ID logs
          if (prev.some(l => l.id === logItem.id)) return prev;
          const updated = [...prev, logItem]
          return updated.slice(-100) // Keep last 100 logs
        })

        // Dynamic State Engine: parse logs to update the real-time HUD metrics
        const message = logItem.message || ''
        
        if (message.includes('SECURITY ALERT: Single person')) {
          setCurrentStatus('WARNING')
          setDetectedCount(1)
          setAuthCount(message.includes('authenticated') ? 1 : 0)
          setUnknownCount(message.includes('authenticated') ? 0 : 1)
        } else if (message.includes('SECURITY ALERT: Unauthorized access')) {
          setCurrentStatus('ALERT')
          // Parse unknown count or default
          setUnknownCount(2)
          setDetectedCount(2)
        } else if (message.includes('SECURITY ALERT: Excess people')) {
          setCurrentStatus('ALERT')
          setDetectedCount(3)
        } else if (message.includes('Authorized access:')) {
          setCurrentStatus('NORMAL')
          setAuthCount(2)
          setUnknownCount(0)
          setDetectedCount(2)
          
          // Try to extract user names
          const namesStr = message.split('Authorized access:')[1]
          if (namesStr) {
            setActiveUsers(namesStr.split(',').map(s => s.trim()))
          }
        } else if (message.includes('normal') || message.includes('stopped') || message.includes('NORMAL')) {
          setCurrentStatus('NORMAL')
          setDetectedCount(0)
          setAuthCount(0)
          setUnknownCount(0)
          setActiveUsers([])
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message', err)
      }
    }

    ws.onclose = () => {
      setConnectionStatus('disconnected')
      console.log('⚠ Security logs WebSocket disconnected. Reconnecting in 3s...')
      setTimeout(connectWebSocket, 3000)
    }

    ws.onerror = () => {
      setConnectionStatus('disconnected')
    }
  }

  // Fetch Enrolled Users list from Backend
  const fetchEnrolledUsers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users`)
      if (response.ok) {
        const data = await response.json()
        setEnrolledUsers(data)
      }
    } catch (err) {
      console.error('Failed to fetch enrolled users', err)
    }
  }

  // Fetch users when switching to the 'users' tab
  useEffect(() => {
    if (activeTab === 'users') {
      fetchEnrolledUsers()
    }
  }, [activeTab])

  // Delete an enrolled user
  const handleDeleteUser = async (userId) => {
    if (!window.confirm(`Are you sure you want to remove user ID: ${userId}?`)) return
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/${userId}`, { method: 'DELETE' })
      if (response.ok) {
        setEnrolledUsers(prev => prev.filter(u => u.id !== userId))
        setSystemLogs(prev => [
          ...prev, 
          {
            id: String(Date.now()),
            timestamp: new Date().toLocaleTimeString(),
            level: 'INFO',
            message: `[DATABASE] Successfully deleted user: ${userId}`
          }
        ])
      } else {
        alert('Failed to delete user.')
      }
    } catch (err) {
      console.error('Error deleting user', err)
    }
  }

  // Start Browser Camera for capture enrollment
  const startLocalCamera = async () => {
    setEnrollStatus({ type: '', message: '' })
    try {
      console.log('[DEBUG] Requesting browser camera access...')
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      console.log('[DEBUG] Camera stream obtained:', stream)
      console.log('[DEBUG] Video tracks:', stream.getVideoTracks())
      
      // Store stream in ref first
      localStreamRef.current = stream
      console.log('[DEBUG] Stream stored in ref')
      
      // Set camera active FIRST so video element renders
      setCameraActive(true)
      console.log('[DEBUG] Camera state set to active (waiting for render...)')
      
      // Wait for React to render the video element
      await new Promise(resolve => setTimeout(resolve, 50))
      
      // Now attach the stream
      if (localVideoRef.current) {
        console.log('[DEBUG] Video element found, setting srcObject...')
        localVideoRef.current.srcObject = stream
        console.log('[DEBUG] srcObject set successfully')
        
        // Add event listeners to debug
        localVideoRef.current.onloadedmetadata = () => {
          console.log('[DEBUG] Video metadata loaded, dimensions:', localVideoRef.current.videoWidth, 'x', localVideoRef.current.videoHeight)
        }
        
        localVideoRef.current.onplaying = () => {
          console.log('[DEBUG] Video is now playing!')
        }
        
        localVideoRef.current.onerror = (e) => {
          console.error('[DEBUG] Video element error:', e)
        }
        
        // Force play
        console.log('[DEBUG] Attempting to play video...')
        const playPromise = localVideoRef.current.play()
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              console.log('[DEBUG] Video play() succeeded')
            })
            .catch(err => {
              console.error('[DEBUG] Video play() failed:', err)
            })
        }
      } else {
        console.error('[ERROR] Video ref is still null after render!')
        setEnrollStatus({ type: 'error', message: 'Video element failed to initialize' })
      }
      
      setEnrollStatus({ type: 'success', message: 'Browser camera ready!' })
    } catch (err) {
      console.error('[ERROR] Failed to open local browser webcam:', err.name, err.message)
      setCameraActive(false)
      setEnrollStatus({ 
        type: 'error', 
        message: `Camera error: ${err.name} - ${err.message}. Try "Allow" camera permission or use file upload.` 
      })
    }
  }

  // Stop Browser Camera
  const stopLocalCamera = () => {
    console.log('[DEBUG] Stopping local camera...')
    if (localStreamRef.current) {
      console.log('[DEBUG] Stopping all video tracks...')
      localStreamRef.current.getTracks().forEach(track => {
        console.log('[DEBUG] Stopping track:', track.kind, track.readyState)
        track.stop()
      })
      localStreamRef.current = null
      console.log('[DEBUG] Local stream cleared')
    } else {
      console.log('[DEBUG] No local stream to stop')
    }
    setCameraActive(false)
  }

  const stopServerCamera = async () => {
    try {
      console.log('[DEBUG] Calling /api/camera/stop...')
      const res = await fetch(`${API_BASE_URL}/api/camera/stop`, { method: 'POST' })
      const data = await res.json()
      console.log('[DEBUG] Stop response:', data, 'Status:', res.status)
      
      if (res.ok) {
        setServerCameraActive(false)
        console.log('✓ Surveillance camera feed released by server')
        return true
      } else {
        console.error('✗ Stop camera failed:', data)
        return false
      }
    } catch (err) {
      console.error('✗ Failed to stop server camera', err)
      return false
    }
  }

  const startServerCamera = async () => {
    try {
      console.log('[DEBUG] Calling /api/camera/start...')
      const res = await fetch(`${API_BASE_URL}/api/camera/start`, { method: 'POST' })
      const data = await res.json()
      console.log('[DEBUG] Start response:', data, 'Status:', res.status)
      
      if (res.ok) {
        setServerCameraActive(true)
        console.log('✓ Surveillance camera feed re-acquired by server')
        return true
      } else {
        console.error('✗ Start camera failed:', data)
        return false
      }
    } catch (err) {
      console.error('✗ Failed to start server camera', err)
      return false
    }
  }

  // Trigger camera startup/shutdown depending on toggle states
  useEffect(() => {
    let isCurrent = true
    
    const manageCameras = async () => {
      console.log(`[DEBUG] Camera manager: activeTab=${activeTab}, enrollMode=${enrollMode}`)
      
      if (activeTab === 'enroll' && enrollMode === 'camera') {
        console.log('[DEBUG] ENTERING enrollment mode')
        // Stop the server surveillance camera to release device 0
        setEnrollStatus({ type: 'loading', message: 'Releasing server camera lock...' })
        const stopSuccess = await stopServerCamera()
        if (!isCurrent) return
        
        if (!stopSuccess) {
          setEnrollStatus({ type: 'error', message: 'Failed to release server camera. Check browser console for details.' })
          return
        }
        
        // Wait for camera hardware to fully release
        await new Promise(resolve => setTimeout(resolve, 1000))
        if (!isCurrent) return
        
        setEnrollStatus({ type: 'loading', message: 'Initializing local browser camera...' })
        await startLocalCamera()
      } else {
        console.log('[DEBUG] EXITING enrollment mode, returning to live surveillance')
        // Leaving enrollment mode - stop browser camera first
        stopLocalCamera()
        
        // Wait for browser to fully release the camera (Windows needs this)
        console.log('[DEBUG] Waiting for browser camera to fully release...')
        await new Promise(resolve => setTimeout(resolve, 1500))
        if (!isCurrent) return
        
        // Then restart backend camera
        console.log('[DEBUG] Browser camera released, restarting backend surveillance...')
        const startSuccess = await startServerCamera()
        
        if (!startSuccess) {
          console.error('[ERROR] Failed to restart backend camera! Trying again in 2s...')
          await new Promise(resolve => setTimeout(resolve, 2000))
          if (isCurrent) {
            await startServerCamera()
          }
        }
      }
    }
    
    manageCameras()
    
    return () => {
      console.log('[DEBUG] Cleanup: Effect unmounting')
      isCurrent = false
      stopLocalCamera()
    }
  }, [activeTab, enrollMode])

  // Capture face snapshot from canvas
  const handleCaptureSnap = () => {
    if (!localVideoRef.current || !cameraActive) return
    if (capturedSnaps.length >= 5) {
      setEnrollStatus({ type: 'error', message: 'Maximum 5 samples reached. You are ready to enroll.' })
      return
    }

    const video = localVideoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    
    const ctx = canvas.getContext('2d')
    // Flip canvas horizontally for mirror effect matching preview
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    
    // Draw flash overlay effect on HUD
    setCapturingCount(c => c + 1)
    setTimeout(() => setCapturingCount(0), 100)

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `snap_${capturedSnaps.length + 1}.jpg`, { type: 'image/jpeg' })
        setCapturedSnaps(prev => [...prev, file])
        setEnrollStatus({ 
          type: 'success', 
          message: `Captured sample ${capturedSnaps.length + 1}/5!` 
        })
      }
    }, 'image/jpeg', 0.95)
  }

  // Remove a captured snapshot
  const handleRemoveSnap = (index) => {
    setCapturedSnaps(prev => prev.filter((_, idx) => idx !== index))
  }

  // Handle Drag & Drop Events
  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      addUploadFiles(e.dataTransfer.files)
    }
  }

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      addUploadFiles(e.target.files)
    }
  }

  const addUploadFiles = (filesList) => {
    setEnrollStatus({ type: '', message: '' })
    const validImageTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/jpg']
    const newFiles = []

    for (let i = 0; i < filesList.length; i++) {
      const file = filesList[i]
      if (validImageTypes.includes(file.type)) {
        newFiles.push(file)
      }
    }

    setUploadFiles(prev => {
      const total = [...prev, ...newFiles]
      if (total.length > 5) {
        setEnrollStatus({ type: 'error', message: 'You can upload a maximum of 5 images. Trimmed files.' })
        return total.slice(0, 5)
      }
      return total
    })
  }

  const handleRemoveUploadFile = (index) => {
    setUploadFiles(prev => prev.filter((_, idx) => idx !== index))
  }

  // Form submission for Face Enrollment
  const handleEnrollSubmit = async (e) => {
    e.preventDefault()
    setEnrollStatus({ type: '', message: '' })

    if (!enrollUserId.trim() || !enrollName.trim()) {
      setEnrollStatus({ type: 'error', message: 'Please enter both a Unique User ID and Display Name.' })
      return
    }

    const filesToUpload = enrollMode === 'camera' ? capturedSnaps : uploadFiles

    if (filesToUpload.length === 0) {
      setEnrollStatus({ 
        type: 'error', 
        message: enrollMode === 'camera' 
          ? 'Please capture at least 1 face sample.' 
          : 'Please select or drag at least 1 image file.' 
      })
      return
    }

    setEnrolling(true)
    setEnrollStatus({ type: 'loading', message: 'Uploading and processing facial embeddings...' })

    const formData = new FormData()
    formData.append('user_id', enrollUserId)
    formData.append('name', enrollName)
    
    filesToUpload.forEach(file => {
      formData.append('files', file)
    })

    try {
      const response = await fetch(`${API_BASE_URL}/api/enroll`, {
        method: 'POST',
        body: formData
      })

      const result = await response.json()

      if (response.ok) {
        setEnrollStatus({ 
          type: 'success', 
          message: `✓ Enrollment Success: User '${enrollName}' enrolled with ${result.processed} samples!` 
        })
        
        // Reset inputs
        setEnrollUserId('')
        setEnrollName('')
        setCapturedSnaps([])
        setUploadFiles([])
        
        // Push notification log
        setSystemLogs(prev => [
          ...prev,
          {
            id: String(Date.now()),
            timestamp: new Date().toLocaleTimeString(),
            level: 'INFO',
            message: `[ENROLL] User successfully registered: ${enrollName} (${enrollUserId})`
          }
        ])
      } else {
        const errorDetail = result.detail || {}
        const errorMsg = errorDetail.message || 'Face extraction failed. Try another picture.'
        const errorsList = errorDetail.errors || []
        
        setEnrollStatus({ 
          type: 'error', 
          message: `✗ Enrollment Failed: ${errorMsg}. ${errorsList.join(', ')}` 
        })
      }
    } catch (err) {
      console.error(err)
      setEnrollStatus({ type: 'error', message: 'Failed to communicate with enrollment API.' })
    } finally {
      setEnrolling(false)
    }
  }

  // Get status color coding
  const getStatusDetails = () => {
    switch (currentStatus) {
      case 'WARNING':
        return { text: 'Alert: Single Person', badgeClass: 'badge-warning', panelClass: 'security-alert-single' }
      case 'ALERT':
        return { text: 'Breach: Security Alert', badgeClass: 'badge-alert', panelClass: 'security-alert-breach' }
      case 'NORMAL':
      default:
        return { text: 'Secured', badgeClass: 'badge-normal', panelClass: 'security-alert-normal' }
    }
  }

  const statusInfo = getStatusDetails()

  // Filter terminal logs
  const filteredLogs = systemLogs.filter(log => {
    if (logFilter === 'all') return true
    if (logFilter === 'info') return log.level === 'INFO'
    if (logFilter === 'warning') return log.level === 'WARNING'
    if (logFilter === 'error') return log.level === 'ERROR'
    return true
  })

  return (
    <div className="dashboard-container">
      {/* Header Panel */}
      <header className="dashboard-header">
        <div className="header-left">
          <div className="system-status-indicator"></div>
          <div className="header-title">
            <h1>Strongroom Surveillance</h1>
            <span>AI Multi-Person Tracking System</span>
          </div>
        </div>
        
        <div className="header-metrics">
          <div className="metric-item">
            <span className="metric-label">API SERVER</span>
            <span className="metric-value" style={{ color: connectionStatus === 'connected' ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {connectionStatus === 'connected' ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">FEED SPEED</span>
            <span className="metric-value">5 Hz / 30 FPS</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">SYS STATE</span>
            <span className={`status-state-badge ${statusInfo.badgeClass}`}>
              {statusInfo.text}
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid Panel */}
      <main className="dashboard-grid">
        {/* Left Column: Live Camera Video monitors */}
        <section className="left-column" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className={`panel ${statusInfo.panelClass}`}>
            <div className="panel-header">
              <div className="panel-title">
                <span style={{ color: serverCameraActive ? 'var(--accent-green)' : 'var(--accent-red)' }}>■</span> Live Surveillance Monitor
                {!serverCameraActive && <span style={{ fontSize: '0.7rem', color: 'var(--accent-red)', marginLeft: '0.5rem', fontFamily: 'var(--font-mono)' }}>(STANDBY)</span>}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <button
                  type="button"
                  onClick={() => serverCameraActive ? stopServerCamera() : startServerCamera()}
                  style={{
                    background: 'rgba(8, 12, 24, 0.6)',
                    border: '1px solid var(--border-color)',
                    color: serverCameraActive ? 'var(--accent-red)' : 'var(--accent-cyan)',
                    fontSize: '0.65rem',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    fontFamily: 'var(--font-sans)',
                    transition: 'all var(--transition-fast)'
                  }}
                >
                  {serverCameraActive ? 'Pause Monitor' : 'Resume Monitor'}
                </button>
                <div className="rec-badge">
                  {serverCameraActive ? (
                    <>
                      <div className="rec-dot"></div>
                      <span>LIVE FEED</span>
                    </>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>STANDBY</span>
                  )}
                </div>
              </div>
            </div>

            {/* Video stream container */}
            <div className="video-viewport">
              {serverCameraActive ? (
                <img 
                  src={`${API_BASE_URL}/api/video`}
                  alt="Camera feed is offline. Please check connection."
                  className="surveillance-stream" 
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
              ) : null}
              
              {!serverCameraActive ? (
                <div style={{ display: 'flex', position: 'absolute', inset: 0, flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#070a13', color: 'var(--accent-cyan)', gap: '1rem', fontFamily: 'var(--font-mono)' }}>
                  <span style={{ fontSize: '2rem', color: 'var(--accent-yellow)', animation: 'blink 1.5s infinite steps(2)' }}>⚠</span>
                  <span style={{ fontSize: '0.8rem', letterSpacing: '0.1em' }}>SURVEILLANCE STANDBY</span>
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', width: '70%', textAlign: 'center', lineHeight: '1.4' }}>
                    Camera hardware has been released. Ready for face enrollment or manual resume.
                  </span>
                </div>
              ) : (
                <div className="no-video-placeholder" style={{ display: 'none', position: 'absolute', inset: 0, flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#070a13', color: 'var(--text-muted)', gap: '1rem', fontFamily: 'var(--font-mono)' }}>
                  <span style={{ fontSize: '2rem' }}>✗</span>
                  <span>Webcam Surveillance Stream Offline</span>
                </div>
              )}
              
              <div className="viewport-overlay">
                <div className="scanning-line" style={{ display: serverCameraActive ? 'block' : 'none' }}></div>
              </div>

              <div className="camera-hud">
                <div className="hud-top">
                  <span>CAM_01 // SECURE_ROOM</span>
                  <span>UTC {new Date().toISOString().substring(11, 19)}</span>
                </div>
                <div className="hud-bottom">
                  <span>YOLOv8 DETECTOR {serverCameraActive ? 'ENABLED' : 'PAUSED'}</span>
                  <span>INSIGHTFACE AUTH {serverCameraActive ? 'ACTIVE' : 'PAUSED'}</span>
                </div>
              </div>
            </div>

            {/* Active monitor live metrics status bar */}
            <div className="status-panel-grid">
              <div className="status-card">
                <span className="status-card-label">Overall State</span>
                <span className={`status-state-badge ${statusInfo.badgeClass}`} style={{ alignSelf: 'flex-start', marginTop: '0.2rem' }}>
                  {currentStatus}
                </span>
              </div>
              <div className="status-card">
                <span className="status-card-label">People Detected</span>
                <span className="status-card-value">{detectedCount}</span>
              </div>
              <div className="status-card">
                <span className="status-card-label">Authenticated</span>
                <span className="status-card-value" style={{ color: 'var(--accent-green)' }}>{authCount}</span>
              </div>
              <div className="status-card">
                <span className="status-card-label">Unknowns</span>
                <span className="status-card-value" style={{ color: 'var(--accent-red)' }}>{unknownCount}</span>
              </div>
            </div>
          </div>
          
          {/* Quick HUD reference helper info */}
          {activeUsers.length > 0 && (
            <div className="panel" style={{ padding: '1rem 1.5rem', background: 'rgba(16, 185, 129, 0.05)', borderColor: 'var(--accent-green-glow)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--accent-green)', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Currently Inside
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                {activeUsers.map((user, idx) => (
                  <span key={idx} style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid var(--accent-green)', padding: '0.2rem 0.6rem', borderRadius: '4px', fontSize: '0.8rem', color: 'white', fontWeight: 600 }}>
                    ✓ {user}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Tab systems for Logs, Enrollment & User DB management */}
        <section className="right-column">
          <div className="panel" style={{ height: '100%' }}>
            <div className="tabs-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
                  onClick={() => setActiveTab('logs')}
                >
                  Security Logs
                </button>
                <button 
                  className={`tab-btn ${activeTab === 'enroll' ? 'active' : ''}`}
                  onClick={() => setActiveTab('enroll')}
                >
                  Enroll Face
                </button>
                <button 
                  className={`tab-btn ${activeTab === 'users' ? 'active' : ''}`}
                  onClick={() => setActiveTab('users')}
                >
                  Users DB
                </button>
              </div>
              {activeTab !== 'logs' && (
                <button 
                  onClick={async () => {
                    console.log('[MANUAL] User clicked Return to Live')
                    await startServerCamera()
                    setActiveTab('logs')
                  }}
                  style={{
                    padding: '0.5rem 1rem',
                    background: 'linear-gradient(135deg, var(--accent-green) 0%, #059669 100%)',
                    border: 'none',
                    borderRadius: '6px',
                    color: 'white',
                    fontSize: '0.85rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap'
                  }}
                  onMouseEnter={(e) => e.target.style.opacity = '0.9'}
                  onMouseLeave={(e) => e.target.style.opacity = '1'}
                >
                  [OK] Return to Live
                </button>
              )}
            </div>

            {/* TAB CONTENT: Security Logs Panel */}
            {activeTab === 'logs' && (
              <div className="panel-body-scrollable" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="logs-filter-container">
                  <button className={`filter-chip ${logFilter === 'all' ? 'active' : ''}`} onClick={() => setLogFilter('all')}>ALL</button>
                  <button className={`filter-chip ${logFilter === 'info' ? 'active' : ''}`} onClick={() => setLogFilter('info')}>INFO</button>
                  <button className={`filter-chip ${logFilter === 'warning' ? 'active' : ''}`} onClick={() => setLogFilter('warning')}>WARNINGS</button>
                  <button className={`filter-chip ${logFilter === 'error' ? 'active' : ''}`} onClick={() => setLogFilter('error')}>ALERTS</button>
                </div>

                <div className="logs-console">
                  {filteredLogs.length === 0 ? (
                    <div className="no-logs">
                      <span>■ READY FOR TELEMETRY STREAM</span>
                      <span style={{ color: 'var(--text-muted)' }}>Standing by. Access room to trigger logs.</span>
                    </div>
                  ) : (
                    filteredLogs.map((log) => (
                      <div key={log.id} className="log-entry">
                        <span className="log-time">{log.timestamp}</span>
                        <span className={`log-level ${log.level}`}>{log.level}</span>
                        <span className="log-text">{log.message}</span>
                      </div>
                    ))
                  )}
                  <div ref={logsEndRef}></div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: Face Enrollment Forms */}
            {activeTab === 'enroll' && (
              <div className="panel-body-scrollable">
                <form onSubmit={handleEnrollSubmit} className="enroll-form">
                  <div className="form-group">
                    <label className="form-label">User ID (Unique identification)</label>
                    <input 
                      type="text" 
                      placeholder="e.g. employee_991" 
                      value={enrollUserId} 
                      onChange={(e) => setEnrollUserId(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
                      className="form-input"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Full Name</label>
                    <input 
                      type="text" 
                      placeholder="e.g. John Doe" 
                      value={enrollName} 
                      onChange={(e) => setEnrollName(e.target.value)}
                      className="form-input"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Enrollment Method</label>
                    <div className="enroll-options-toggle">
                      <button 
                        type="button" 
                        className={`toggle-option-btn ${enrollMode === 'camera' ? 'active' : ''}`}
                        onClick={() => setEnrollMode('camera')}
                      >
                        Webcam Capture
                      </button>
                      <button 
                        type="button" 
                        className={`toggle-option-btn ${enrollMode === 'upload' ? 'active' : ''}`}
                        onClick={() => setEnrollMode('upload')}
                      >
                        Upload Images
                      </button>
                    </div>
                  </div>

                  {/* SUB-MODE: Webcam snapshot capture */}
                  {enrollMode === 'camera' && (
                    <div className="local-webcam-container">
                      <span className="form-label" style={{ fontSize: '0.65rem' }}>Browser Camera Source</span>
                      <div className="local-webcam-preview" style={{ transform: capturingCount > 0 ? 'scale(0.995)' : 'none', border: capturingCount > 0 ? '2px solid var(--accent-red)' : '1px solid var(--border-color)' }}>
                        {cameraActive ? (
                          <video 
                            ref={localVideoRef} 
                            autoPlay 
                            playsInline 
                            muted
                            controls
                            className="local-webcam-video"
                            style={{ transform: 'scaleX(-1)' }} // Mirror video
                          />
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
                            Starting camera stream...
                          </div>
                        )}
                      </div>
                      
                      <div className="capture-controls">
                        <button 
                          type="button" 
                          onClick={handleCaptureSnap}
                          disabled={!cameraActive || capturedSnaps.length >= 5}
                          className="capture-btn"
                        >
                          Snap Photo ({capturedSnaps.length}/5)
                        </button>
                      </div>

                      {capturedSnaps.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginTop: '0.25rem' }}>
                          <span className="form-label" style={{ fontSize: '0.65rem' }}>Face Snap Samples:</span>
                          <div className="preview-grid">
                            {capturedSnaps.map((snap, idx) => (
                              <div key={idx} className="preview-thumbnail-wrapper">
                                <img 
                                  src={URL.createObjectURL(snap)} 
                                  alt="Preview" 
                                  className="preview-thumbnail"
                                />
                                <button 
                                  type="button" 
                                  onClick={() => handleRemoveSnap(idx)}
                                  className="remove-thumbnail-btn"
                                >
                                  ×
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* SUB-MODE: Drag & Drop files */}
                  {enrollMode === 'upload' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                      <div 
                        className={`drag-drop-zone ${dragActive ? 'active' : ''}`}
                        onDragEnter={handleDrag}
                        onDragOver={handleDrag}
                        onDragLeave={handleDrag}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById('manual-file-input').click()}
                      >
                        <span className="upload-icon">⇪</span>
                        <div className="upload-text">
                          <span className="upload-text-highlight">Click to upload</span> or drag images here
                        </div>
                        <span className="upload-limit">Accepts JPG, PNG, BMP (Up to 5 images)</span>
                        <input 
                          type="file" 
                          id="manual-file-input" 
                          multiple 
                          accept="image/*"
                          onChange={handleFileInput}
                          style={{ display: 'none' }}
                        />
                      </div>

                      {uploadFiles.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                          <span className="form-label" style={{ fontSize: '0.65rem' }}>Selected Files ({uploadFiles.length}/5):</span>
                          <div className="preview-grid">
                            {uploadFiles.map((file, idx) => (
                              <div key={idx} className="preview-thumbnail-wrapper">
                                <img 
                                  src={URL.createObjectURL(file)} 
                                  alt="Upload preview" 
                                  className="preview-thumbnail"
                                />
                                <button 
                                  type="button" 
                                  onClick={() => handleRemoveUploadFile(idx)}
                                  className="remove-thumbnail-btn"
                                >
                                  ×
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Status Banner */}
                  {enrollStatus.message && (
                    <div className={`server-status-banner`} style={{
                      background: enrollStatus.type === 'error' ? 'rgba(244, 63, 94, 0.1)' : enrollStatus.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(6, 182, 212, 0.1)',
                      color: enrollStatus.type === 'error' ? 'var(--accent-red)' : enrollStatus.type === 'success' ? 'var(--accent-green)' : 'var(--accent-cyan)',
                      borderColor: enrollStatus.type === 'error' ? 'rgba(244, 63, 94, 0.2)' : enrollStatus.type === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(6, 182, 212, 0.2)'
                    }}>
                      {enrollStatus.type === 'loading' && <div className="spinner"></div>}
                      <span>{enrollStatus.message}</span>
                    </div>
                  )}

                  <button 
                    type="submit" 
                    className="submit-btn"
                    disabled={enrolling || (enrollMode === 'camera' ? capturedSnaps.length === 0 : uploadFiles.length === 0)}
                  >
                    {enrolling ? 'Enrolling...' : 'Register User'}
                  </button>
                </form>
              </div>
            )}

            {/* TAB CONTENT: Enrolled Users Management DB */}
            {activeTab === 'users' && (
              <div className="panel-body-scrollable">
                <div className="users-list">
                  {enrolledUsers.length === 0 ? (
                    <div className="no-users">
                      <span>No users registered in system database.</span>
                    </div>
                  ) : (
                    enrolledUsers.map((user) => (
                      <div key={user.id} className="user-item">
                        <div className="user-info">
                          <span className="user-name">{user.name}</span>
                          <span className="user-id">ID: {user.id}</span>
                        </div>
                        <button 
                          className="delete-user-btn"
                          onClick={() => handleDeleteUser(user.id)}
                        >
                          Delete
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
