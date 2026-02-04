/**
 * Student Behavior Monitoring System - React Frontend
 * =====================================================
 * 
 * A real-time dashboard that:
 * 1. Captures webcam feed using react-webcam
 * 2. Sends frames to Flask backend via Socket.IO
 * 3. Displays behavior status (Attentive/Drowsy/Yawning/Distracted)
 * 4. Shows real-time attentiveness chart using recharts
 * 5. Real JWT authentication with login/register
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { io } from 'socket.io-client';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Area,
    AreaChart
} from 'recharts';
import './App.css';

// Backend server URL
const API_URL = 'http://localhost:5000';
const SOCKET_SERVER_URL = 'http://localhost:5000';

// Maximum data points to show in chart
const MAX_CHART_POINTS = 30;

// Status color mapping
const STATUS_COLORS = {
    'Attentive': '#00ff88',
    'Drowsy': '#ff4444',
    'Yawning': '#ffaa00',
    'Distracted': '#ff6600',
    'No Face Detected': '#888888'
};

function App() {
    // State management
    const [isConnected, setIsConnected] = useState(false);
    const [currentStatus, setCurrentStatus] = useState('Waiting...');
    const [earScore, setEarScore] = useState(0);
    const [marScore, setMarScore] = useState(0);
    const [chartData, setChartData] = useState([]);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [user, setUser] = useState(null);

    // Auth modal state
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
    const [authForm, setAuthForm] = useState({ email: '', password: '', name: '' });
    const [authError, setAuthError] = useState('');
    const [authLoading, setAuthLoading] = useState(false);

    // Refs
    const webcamRef = useRef(null);
    const socketRef = useRef(null);
    const captureIntervalRef = useRef(null);

    /**
     * Initialize Socket.IO connection
     */
    useEffect(() => {
        // Create socket connection
        socketRef.current = io(SOCKET_SERVER_URL, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
        });

        // Connection event handlers
        socketRef.current.on('connect', () => {
            console.log('Connected to server');
            setIsConnected(true);
        });

        socketRef.current.on('disconnect', () => {
            console.log('Disconnected from server');
            setIsConnected(false);
        });

        socketRef.current.on('connection_response', (data) => {
            console.log('Server response:', data);
        });

        // Status update handler - receives analysis results from backend
        socketRef.current.on('status_update', (data) => {
            const { status, ear_score, mar_score } = data;

            setCurrentStatus(status);
            setEarScore(ear_score);
            setMarScore(mar_score);

            // Calculate attentiveness score (0-100)
            let attentivenessScore = 100;

            if (status === 'Drowsy') {
                attentivenessScore = Math.max(20, ear_score * 200);
            } else if (status === 'Yawning') {
                attentivenessScore = Math.max(30, 100 - (mar_score * 100));
            } else if (status === 'Distracted') {
                attentivenessScore = 25;
            } else if (status === 'No Face Detected') {
                attentivenessScore = 0;
            }

            // Add to chart data
            const timestamp = new Date().toLocaleTimeString();
            setChartData(prevData => {
                const newData = [...prevData, {
                    time: timestamp,
                    score: Math.round(attentivenessScore),
                    ear: ear_score * 100,
                    mar: mar_score * 100
                }];
                return newData.slice(-MAX_CHART_POINTS);
            });
        });

        // Cleanup on unmount
        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            if (captureIntervalRef.current) {
                clearInterval(captureIntervalRef.current);
            }
        };
    }, []);

    /**
     * Check for existing login on mount
     */
    useEffect(() => {
        const token = localStorage.getItem('sbms_auth_token');
        const savedUser = localStorage.getItem('sbms_user');
        if (token && savedUser) {
            setIsLoggedIn(true);
            setUser(JSON.parse(savedUser));
        }
    }, []);

    /**
     * Capture and send webcam frame to backend
     */
    const captureFrame = useCallback(() => {
        if (webcamRef.current && socketRef.current && socketRef.current.connected) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                socketRef.current.emit('video_frame', { frame: imageSrc });
            }
        }
    }, []);

    /**
     * Start/Stop streaming webcam frames
     */
    const toggleStreaming = () => {
        if (isStreaming) {
            if (captureIntervalRef.current) {
                clearInterval(captureIntervalRef.current);
                captureIntervalRef.current = null;
            }
            setIsStreaming(false);
            setCurrentStatus('Paused');
        } else {
            captureIntervalRef.current = setInterval(captureFrame, 200);
            setIsStreaming(true);
        }
    };

    /**
     * Handle login form submission
     */
    const handleLogin = async (e) => {
        e.preventDefault();
        setAuthError('');
        setAuthLoading(true);

        try {
            const response = await fetch(`${API_URL}/api/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: authForm.email,
                    password: authForm.password
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Login failed');
            }

            // Store token and user info
            localStorage.setItem('sbms_auth_token', data.access_token);
            localStorage.setItem('sbms_user', JSON.stringify(data.user));

            setUser(data.user);
            setIsLoggedIn(true);
            setShowAuthModal(false);
            setAuthForm({ email: '', password: '', name: '' });
            console.log('Logged in successfully:', data.user.email);
        } catch (error) {
            setAuthError(error.message);
        } finally {
            setAuthLoading(false);
        }
    };

    /**
     * Handle register form submission
     */
    const handleRegister = async (e) => {
        e.preventDefault();
        setAuthError('');
        setAuthLoading(true);

        try {
            const response = await fetch(`${API_URL}/api/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: authForm.email,
                    password: authForm.password,
                    name: authForm.name
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Registration failed');
            }

            // Registration successful, switch to login
            setAuthMode('login');
            setAuthError('');
            alert('Registration successful! Please login.');
            setAuthForm({ ...authForm, name: '' });
        } catch (error) {
            setAuthError(error.message);
        } finally {
            setAuthLoading(false);
        }
    };

    /**
     * Logout handler
     */
    const handleLogout = () => {
        localStorage.removeItem('sbms_auth_token');
        localStorage.removeItem('sbms_user');
        setIsLoggedIn(false);
        setUser(null);

        if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current);
            captureIntervalRef.current = null;
        }
        setIsStreaming(false);
    };

    /**
     * Open auth modal
     */
    const openAuthModal = (mode) => {
        setAuthMode(mode);
        setAuthError('');
        setAuthForm({ email: '', password: '', name: '' });
        setShowAuthModal(true);
    };

    // Get status badge color
    const statusColor = STATUS_COLORS[currentStatus] || '#888888';

    return (
        <div className="app">
            {/* Auth Modal */}
            {showAuthModal && (
                <div className="modal-overlay" onClick={() => setShowAuthModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h2>{authMode === 'login' ? 'Login' : 'Register'}</h2>

                        <form onSubmit={authMode === 'login' ? handleLogin : handleRegister}>
                            {authMode === 'register' && (
                                <div className="form-group">
                                    <label>Name</label>
                                    <input
                                        type="text"
                                        value={authForm.name}
                                        onChange={(e) => setAuthForm({ ...authForm, name: e.target.value })}
                                        placeholder="Your name"
                                        required
                                    />
                                </div>
                            )}

                            <div className="form-group">
                                <label>Email</label>
                                <input
                                    type="email"
                                    value={authForm.email}
                                    onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                                    placeholder="your@email.com"
                                    required
                                />
                            </div>

                            <div className="form-group">
                                <label>Password</label>
                                <input
                                    type="password"
                                    value={authForm.password}
                                    onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                                    placeholder="••••••••"
                                    required
                                />
                            </div>

                            {authError && <div className="auth-error">{authError}</div>}

                            <button type="submit" className="btn btn-primary" disabled={authLoading}>
                                {authLoading ? 'Please wait...' : (authMode === 'login' ? 'Login' : 'Register')}
                            </button>
                        </form>

                        <div className="auth-switch">
                            {authMode === 'login' ? (
                                <p>Don't have an account? <button onClick={() => setAuthMode('register')}>Register</button></p>
                            ) : (
                                <p>Already have an account? <button onClick={() => setAuthMode('login')}>Login</button></p>
                            )}
                        </div>

                        <button className="modal-close" onClick={() => setShowAuthModal(false)}>×</button>
                    </div>
                </div>
            )}

            {/* Header */}
            <header className="header">
                <div className="header-left">
                    <div className="logo">
                        <span className="logo-icon">👁️</span>
                        <span className="logo-text">SBMS</span>
                    </div>
                    <span className="header-subtitle">Student Behavior Monitoring System</span>
                </div>
                <div className="header-right">
                    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                        <span className="status-dot"></span>
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </div>
                    {isLoggedIn ? (
                        <div className="user-info">
                            <span className="user-name">👤 {user?.name}</span>
                            <button className="btn btn-logout" onClick={handleLogout}>
                                Logout
                            </button>
                        </div>
                    ) : (
                        <button className="btn btn-login" onClick={() => openAuthModal('login')}>
                            Login
                        </button>
                    )}
                </div>
            </header>

            {/* Main Content */}
            <main className="main-content">
                {/* Left Panel - Webcam */}
                <section className="panel webcam-panel">
                    <h2 className="panel-title">Live Feed</h2>
                    <div className="webcam-container">
                        <Webcam
                            ref={webcamRef}
                            audio={false}
                            screenshotFormat="image/jpeg"
                            screenshotQuality={0.6}
                            videoConstraints={{
                                width: 640,
                                height: 480,
                                facingMode: "user"
                            }}
                            className="webcam"
                        />
                        <div className="webcam-overlay">
                            {!isStreaming && (
                                <div className="overlay-message">
                                    Click "Start Monitoring" to begin
                                </div>
                            )}
                        </div>
                    </div>
                    <button
                        className={`btn btn-stream ${isStreaming ? 'streaming' : ''}`}
                        onClick={toggleStreaming}
                        disabled={!isConnected}
                    >
                        {isStreaming ? '⏸️ Stop Monitoring' : '▶️ Start Monitoring'}
                    </button>
                </section>

                {/* Right Panel - Status & Chart */}
                <section className="panel status-panel">
                    {/* Status Badge */}
                    <div className="status-section">
                        <h2 className="panel-title">Current Status</h2>
                        <div
                            className="status-badge"
                            style={{
                                backgroundColor: statusColor,
                                boxShadow: `0 0 30px ${statusColor}40`
                            }}
                        >
                            <span className="status-icon">
                                {currentStatus === 'Attentive' && '✓'}
                                {currentStatus === 'Drowsy' && '😴'}
                                {currentStatus === 'Yawning' && '🥱'}
                                {currentStatus === 'Distracted' && '👀'}
                                {currentStatus === 'No Face Detected' && '❓'}
                                {currentStatus === 'Waiting...' && '⏳'}
                                {currentStatus === 'Paused' && '⏸️'}
                            </span>
                            <span className="status-text">{currentStatus}</span>
                        </div>
                    </div>

                    {/* Metrics */}
                    <div className="metrics-section">
                        <div className="metric">
                            <span className="metric-label">Eye Aspect Ratio (EAR)</span>
                            <span className="metric-value">{earScore.toFixed(3)}</span>
                            <div className="metric-bar">
                                <div
                                    className="metric-bar-fill ear"
                                    style={{ width: `${Math.min(earScore * 200, 100)}%` }}
                                ></div>
                            </div>
                        </div>
                        <div className="metric">
                            <span className="metric-label">Mouth Aspect Ratio (MAR)</span>
                            <span className="metric-value">{marScore.toFixed(3)}</span>
                            <div className="metric-bar">
                                <div
                                    className="metric-bar-fill mar"
                                    style={{ width: `${Math.min(marScore * 100, 100)}%` }}
                                ></div>
                            </div>
                        </div>
                    </div>

                    {/* Chart */}
                    <div className="chart-section">
                        <h2 className="panel-title">Attentiveness Score Over Time</h2>
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height={200}>
                                <AreaChart data={chartData}>
                                    <defs>
                                        <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#00ff88" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#00ff88" stopOpacity={0.1} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis
                                        dataKey="time"
                                        stroke="#666"
                                        tick={{ fill: '#888', fontSize: 10 }}
                                        interval="preserveStartEnd"
                                    />
                                    <YAxis
                                        domain={[0, 100]}
                                        stroke="#666"
                                        tick={{ fill: '#888', fontSize: 10 }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1a1a2e',
                                            border: '1px solid #333',
                                            borderRadius: '8px'
                                        }}
                                        labelStyle={{ color: '#fff' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="score"
                                        stroke="#00ff88"
                                        fill="url(#scoreGradient)"
                                        strokeWidth={2}
                                        name="Attentiveness"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Info Box */}
                    <div className="info-box">
                        <h3>How It Works</h3>
                        <ul>
                            <li><strong>EAR {'<'} 0.22:</strong> Eyes closing → Drowsy</li>
                            <li><strong>MAR {'>'} 0.6:</strong> Mouth open → Yawning</li>
                            <li><strong>Head turned:</strong> Looking away → Distracted</li>
                        </ul>
                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="footer">
                <p>Student Behavior Monitoring System • Powered by MediaPipe & React</p>
            </footer>
        </div>
    );
}

export default App;

