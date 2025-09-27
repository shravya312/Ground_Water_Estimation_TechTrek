import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { signOut, onAuthStateChanged } from 'firebase/auth'
import { auth, db } from '../firebase'
import { collection, onSnapshot, orderBy, query } from 'firebase/firestore'
import { saveChatMessage, getUserChatHistory, clearChatHistory } from '../services/chatHistoryService'
import EnhancedMarkdownRenderer from '../components/EnhancedMarkdownRenderer'
import LanguageSelector from '../components/LanguageSelector'
import VisualizationPanel from '../components/VisualizationPanel'
import LocationMap from '../components/LocationMap'
import VisualizationModal from '../components/VisualizationModal'
import voiceService from '../services/voiceService'

function Chat1() {
  console.log('Chat1 component loaded with voice features!')
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', text: 'Hi! Ask me anything about groundwater estimation.' },
  ])
  const [input, setInput] = useState('')
  const [history, setHistory] = useState([])
  const [sending, setSending] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [fullChatHistory, setFullChatHistory] = useState([])
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState(-1)
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [selectedLanguage, setSelectedLanguage] = useState('en')
  const [showVisualizationPanel, setShowVisualizationPanel] = useState(false)
  const [showLocationMap, setShowLocationMap] = useState(false)
  const [userLocation, setUserLocation] = useState(null)
  const [isGettingLocation, setIsGettingLocation] = useState(false)
  const [showVisualizationModal, setShowVisualizationModal] = useState(false)
  const [selectedVisualization, setSelectedVisualization] = useState(null)
  const [analysisData, setAnalysisData] = useState(null)
  const [recording, setRecording] = useState(false)
  const [liveTranscript, setLiveTranscript] = useState('')
  const recognitionRef = useRef(null)
  const bottomRef = useRef(null)

  const handleLocationChange = (location, analysisResult = null) => {
    setUserLocation(location)
    setIsGettingLocation(true)
    
    // Simulate analysis delay
    setTimeout(() => {
      setIsGettingLocation(false)
      
      let locationMsg
      if (analysisResult && !analysisResult.error) {
        // If we have analysis results, show them with enhanced structure
        locationMsg = {
          id: Date.now(),
          role: 'assistant',
          text: `📍 **Groundwater Analysis for ${analysisResult.state}**\n\n**Location:** ${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}\n**Data Points:** ${analysisResult.data_points}\n**Districts Covered:** ${analysisResult.summary?.districts_covered || 0}\n**Years:** ${analysisResult.summary?.years_covered?.join(', ') || 'N/A'}\n\n**Analysis:**\n${analysisResult.analysis}\n\nYou can ask me more specific questions about groundwater conditions in ${analysisResult.state}!`,
          analysisData: analysisResult
        }
        setAnalysisData(analysisResult)
      } else {
        // Default location message
        locationMsg = {
          id: Date.now(),
          role: 'assistant',
          text: `📍 **Location Analysis Complete!**\n\nI've detected your location at:\n- **Latitude:** ${location.lat.toFixed(6)}\n- **Longitude:** ${location.lng.toFixed(6)}\n\nI can now provide location-specific groundwater data and analysis for your area. Ask me about groundwater conditions, recharge rates, or extraction levels in your region!`
        }
      }
      setMessages(prev => [...prev, locationMsg])
    }, 2000)
  }

  const handleVisualizationClick = (visualization) => {
    setSelectedVisualization(visualization)
    setShowVisualizationModal(true)
  }

  async function handleSend(e) {
    e.preventDefault()
    
    // If voice recording is active, stop it first
    if (recording) {
      setRecording(false)
      if (recognitionRef.current) {
        recognitionRef.current.stop()
        recognitionRef.current = null
      }
      // Use live transcript if available, otherwise use input
      const textToSend = liveTranscript.trim() || input.trim()
      if (textToSend) {
        setInput(textToSend)
        setLiveTranscript('')
      }
    }
    
    const trimmed = (liveTranscript.trim() || input.trim())
    if (!trimmed) return
    
    // Generate conversation ID if starting new conversation
    const conversationId = currentConversationId || `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    if (!currentConversationId) {
      setCurrentConversationId(conversationId)
    }
    
    const userMsg = { 
      id: Date.now(), 
      role: 'user', 
      text: trimmed,
      conversationId: conversationId
    }
    setMessages(prev => [...prev, userMsg])
    setHistory(prev => [trimmed, ...prev.slice(0, 19)])
    setInput('')
    setLiveTranscript('')
    
    // Save user message to Firebase
    if (auth.currentUser) {
      try {
        console.log('Saving user message to Firebase...')
        const messageId = await saveChatMessage(auth.currentUser.uid, userMsg)
        console.log('User message saved successfully:', messageId)
      } catch (error) {
        console.error('Error saving user message:', error)
        alert('Failed to save message. Please check console for details.')
      }
    } else {
      console.warn('No authenticated user found')
    }
    
    try {
      setSending(true)
      const apiBase = import.meta.env?.VITE_API_URL || ''
      
      // Try INGRES API first for structured responses
      let res, data
      try {
        res = await fetch(`${apiBase}/ingres/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            query: trimmed,
            language: selectedLanguage,
            include_visualizations: true
          })
        })
        
        if (res.ok) {
          data = await res.json()
          
          // Check if this is a structured INGRES response
          if (data.criticality_status || data.visualizations) {
            const assistantMsg = { 
              id: Date.now() + 1, 
              role: 'assistant', 
              text: `# 💧 Groundwater Analysis Report\n\n**Query:** ${trimmed}\n\n## Analysis\n${data.data?.analysis || 'Comprehensive groundwater analysis completed.'}`,
              analysisData: data,
              conversationId: conversationId
            }
            setMessages(prev => [...prev, assistantMsg])
            setAnalysisData(data)
            
            // Save assistant message to Firebase
            if (auth.currentUser) {
              try {
                const messageId = await saveChatMessage(auth.currentUser.uid, assistantMsg)
                console.log('Assistant message saved successfully:', messageId)
              } catch (error) {
                console.error('Error saving assistant message:', error)
              }
            }
            return
          }
        }
      } catch (ingresError) {
        console.warn('INGRES API failed, falling back to regular API:', ingresError)
      }
      
      // Fallback to regular API
      res = await fetch(`${apiBase}/ask-formatted`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: trimmed,
          language: selectedLanguage
        })
      })
      
      try {
        data = await res.json()
      } catch (e) {
        throw new Error(`Unexpected response from server (status ${res.status})`)
      }
      if (!res.ok) {
        throw new Error(data?.error || `Server error (status ${res.status})`)
      }
      const answer = data?.answer || data?.error || 'No answer returned.'
      const assistantMsg = { 
        id: Date.now() + 1, 
        role: 'assistant', 
        text: String(answer),
        conversationId: conversationId
      }
      setMessages(prev => [...prev, assistantMsg])
      
      // Save assistant message to Firebase
      if (auth.currentUser) {
        try {
          console.log('Saving assistant message to Firebase...')
          const messageId = await saveChatMessage(auth.currentUser.uid, assistantMsg)
          console.log('Assistant message saved successfully:', messageId)
        } catch (error) {
          console.error('Error saving assistant message:', error)
          alert('Failed to save assistant response. Please check console for details.')
        }
      }
    } catch (err) {
      const msg = err?.message ? `Error: ${err.message}` : 'Error contacting server.'
      const errorMsg = { 
        id: Date.now() + 2, 
        role: 'assistant', 
        text: msg,
        conversationId: conversationId
      }
      setMessages(prev => [...prev, errorMsg])
      
      // Save error message to Firebase
      if (auth.currentUser) {
        try {
          console.log('Saving error message to Firebase...')
          const messageId = await saveChatMessage(auth.currentUser.uid, errorMsg)
          console.log('Error message saved successfully:', messageId)
        } catch (error) {
          console.error('Error saving error message:', error)
        }
      }
    } finally {
      setSending(false)
    }
  }

  // Helper: populate state from chat messages array
  function populateFromChatHistory(chatHistory) {
    setFullChatHistory(chatHistory)
    if (!chatHistory || chatHistory.length === 0) return
    const conversations = {}
    chatHistory.forEach(msg => {
      const convId = msg.conversationId || 'default'
      if (!conversations[convId]) conversations[convId] = []
      conversations[convId].push(msg)
    })
    const conversationIds = Object.keys(conversations)
    const mostRecentConvId = conversationIds[conversationIds.length - 1]
    const mostRecentConversation = conversations[mostRecentConvId] || []
    const formattedMessages = mostRecentConversation.map(msg => ({
      id: msg.id || Date.now() + Math.random(),
      role: msg.role,
      text: msg.text
    }))
    setMessages(formattedMessages)
    setCurrentConversationId(mostRecentConvId)
    // Build sidebar history as distinct conversations (latest first)
    const convIdsByLatestTs = Object.entries(conversations)
      .map(([cid, msgs]) => ({
        cid,
        lastTs: (msgs[msgs.length - 1]?.timestamp?.seconds) || 0,
        firstUserText: (msgs.find(m => m.role === 'user')?.text) || '(no prompt)'
      }))
      .sort((a, b) => a.lastTs - b.lastTs)
      .slice(-20)
      .reverse()
    setHistory(convIdsByLatestTs.map(x => ({ conversationId: x.cid, text: x.firstUserText })))
  }

  // Load chat history when auth state is ready. Also subscribe realtime to changes
  useEffect(() => {
    let unsubChats = null
    const unsubscribeAuth = onAuthStateChanged(auth, async (user) => {
      if (!user) {
        setFullChatHistory([])
        setHistory([])
        setCurrentConversationId(null)
        if (unsubChats) { unsubChats(); unsubChats = null }
        return
      }
      try {
        setLoadingHistory(true)
        // Initial load via REST
        const initial = await getUserChatHistory(user.uid)
        populateFromChatHistory(initial)
        // Realtime subscription for updates
        try {
          const chatsRef = collection(db, 'users', user.uid, 'chats')
          const q = query(chatsRef, orderBy('timestamp', 'asc'))
          unsubChats = onSnapshot(q, (snap) => {
            const msgs = snap.docs.map(d => ({ id: d.id, ...d.data() }))
            populateFromChatHistory(msgs)
          })
        } catch (e) {
          console.warn('Realtime subscription failed, using static load only:', e)
        }
      } catch (err) {
        console.error('Error loading chat history:', err)
      } finally {
        setLoadingHistory(false)
      }
    })
    return () => { if (unsubChats) unsubChats(); unsubscribeAuth() }
  }, [])

  // Load conversation from history
  const loadConversationFromHistory = (item, historyIndex) => {
    if (!fullChatHistory.length) return
    // Prefer conversationId if available from sidebar item
    if (item?.conversationId) {
      const conversationMessages = fullChatHistory.filter(msg => msg.conversationId === item.conversationId)
      const formattedMessages = conversationMessages.map(msg => ({
        id: msg.id || Date.now() + Math.random(),
        role: msg.role,
        text: msg.text
      }))
      setMessages(formattedMessages)
      setCurrentConversationId(item.conversationId)
      setSelectedHistoryIndex(historyIndex)
      return
    }
    // Fallback by prompt text match
    const candidates = fullChatHistory.filter(msg => msg.role === 'user' && msg.text === item)
    if (candidates.length === 0) return
    const sorted = candidates.sort((a, b) => ((a.timestamp?.seconds||0) - (b.timestamp?.seconds||0)))
    const selectedMessage = sorted[sorted.length - 1]
    if (!selectedMessage?.conversationId) return
    const conversationMessages = fullChatHistory.filter(msg => msg.conversationId === selectedMessage.conversationId)
    const formattedMessages = conversationMessages.map(msg => ({
      id: msg.id || Date.now() + Math.random(),
      role: msg.role,
      text: msg.text
    }))
    setMessages(formattedMessages)
    setCurrentConversationId(selectedMessage.conversationId)
    setSelectedHistoryIndex(historyIndex)
  }


  // Clear chat history function
  const handleClearHistory = async () => {
    if (auth.currentUser && window.confirm('Are you sure you want to clear all chat history?')) {
      try {
        await clearChatHistory(auth.currentUser.uid)
        setMessages([{ id: 1, role: 'assistant', text: 'Hi! Ask me anything about groundwater estimation.' }])
        setHistory([])
        setFullChatHistory([])
        setSelectedHistoryIndex(-1)
        setCurrentConversationId(null)
      } catch (error) {
        console.error('Error clearing chat history:', error)
        alert('Failed to clear chat history. Please try again.')
      }
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  return (
    <div style={{ 
      height: '100vh', 
      display: 'flex',
      backgroundColor: '#f8fafc',
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      overflow: 'hidden'
    }}>
      {/* Sidebar */}
      <aside style={{
        width: '320px',
        backgroundColor: 'white',
        borderRight: '1px solid #e2e8f0',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh'
      }}>
        {/* Sidebar Header */}
        <div style={{
          padding: '1.5rem',
          borderBottom: '1px solid #e2e8f0',
          backgroundColor: '#f8fafc'
        }}>
          <div style={{ 
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1rem'
          }}>
            {/* Logo */}
            <div style={{ 
              width: '40px',
              height: '40px',
              backgroundColor: '#1e3a8a',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <img 
                src="/logo1.png" 
                alt="Logo" 
                style={{
                  height: '24px',
                  width: 'auto',
                  objectFit: 'contain'
                }}
              />
            </div>
            <div>
              <h3 style={{ 
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#1e293b',
                margin: 0
              }}>
                Groundwater Assistant
              </h3>
              <p style={{ 
                fontSize: '0.75rem',
                color: '#64748b',
                margin: 0
              }}>
                AI-Powered Analysis
              </p>
            </div>
          </div>
          
          <LanguageSelector 
            selectedLanguage={selectedLanguage}
            onLanguageChange={setSelectedLanguage}
            className="sidebar-language-selector"
          />
        </div>
        
        {/* Chat Controls */}
        <div style={{
          padding: '1rem 1.5rem',
          borderBottom: '1px solid #e2e8f0'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h4 style={{ 
              fontSize: '0.875rem',
              fontWeight: '600',
              color: '#374151',
              margin: 0
            }}>
              Conversations
            </h4>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {history.length > 0 && (
                <button 
                  onClick={handleClearHistory}
                  style={{
                    padding: '0.25rem 0.5rem',
                    backgroundColor: '#fef2f2',
                    border: '1px solid #fecaca',
                    color: '#dc2626',
                    fontSize: '0.75rem',
                    borderRadius: '0.375rem',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseOver={(e) => e.target.style.backgroundColor = '#fee2e2'}
                  onMouseOut={(e) => e.target.style.backgroundColor = '#fef2f2'}
                  title="Clear all chat history"
                >
                  Clear
                </button>
              )}
              <Link 
                to="/" 
                style={{
                  padding: '0.25rem 0.5rem',
                  color: '#6b7280',
                  fontSize: '0.75rem',
                  textDecoration: 'none',
                  borderRadius: '0.375rem',
                  transition: 'all 0.2s ease'
                }}
                onMouseOver={(e) => e.target.style.color = '#0ea5e9'}
                onMouseOut={(e) => e.target.style.color = '#6b7280'}
              >
                Home
              </Link>
            </div>
          </div>
          
          <button 
            onClick={() => {
              setMessages([{ id: 1, role: 'assistant', text: 'Hi! Ask me anything about groundwater estimation.' }])
              setSelectedHistoryIndex(-1)
              setCurrentConversationId(null)
            }}
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: '0.5rem',
              border: '1px solid #d1d5db',
              backgroundColor: selectedHistoryIndex === -1 ? '#0ea5e9' : 'white',
              color: selectedHistoryIndex === -1 ? 'white' : '#374151',
              fontSize: '0.875rem',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
            onMouseOver={(e) => {
              if (selectedHistoryIndex !== -1) {
                e.target.style.backgroundColor = '#f3f4f6'
                e.target.style.borderColor = '#9ca3af'
              }
            }}
            onMouseOut={(e) => {
              if (selectedHistoryIndex !== -1) {
                e.target.style.backgroundColor = 'white'
                e.target.style.borderColor = '#d1d5db'
              }
            }}
          >
            <span style={{ fontSize: '1rem' }}>+</span>
            New Chat
          </button>
        </div>
        
        {/* Chat History List */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '0.5rem' }}>
          {loadingHistory ? (
            <div style={{
              textAlign: 'center',
              padding: '2rem',
              color: '#6b7280',
              fontSize: '0.875rem'
            }}>
              Loading chat history...
            </div>
          ) : (
            <div>
              {history.length === 0 && (
                <div style={{
                  textAlign: 'center',
                  padding: '2rem',
                  color: '#9ca3af',
                  fontSize: '0.875rem'
                }}>
                  No conversations yet. Start chatting!
                </div>
              )}
              {history.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.75rem',
                    margin: '0.25rem',
                    borderRadius: '0.5rem',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    backgroundColor: selectedHistoryIndex === idx ? '#0ea5e9' : 'transparent',
                    color: selectedHistoryIndex === idx ? 'white' : '#374151',
                    border: selectedHistoryIndex === idx ? 'none' : '1px solid transparent'
                  }}
                  onClick={() => loadConversationFromHistory(item, idx)}
                  onMouseOver={(e) => {
                    if (selectedHistoryIndex !== idx) {
                      e.target.style.backgroundColor = '#f3f4f6'
                    }
                  }}
                  onMouseOut={(e) => {
                    if (selectedHistoryIndex !== idx) {
                      e.target.style.backgroundColor = 'transparent'
                    }
                  }}
                  title="Click to load this conversation"
                >
                  <div style={{
                    fontSize: '0.875rem',
                    lineHeight: '1.4',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {typeof item === 'string' ? item : (item?.text || '(no prompt)')}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#ffffff',
        height: '100vh'
      }}>
        {/* Chat Header */}
        <header style={{
          backgroundColor: 'white',
          borderBottom: '1px solid #e2e8f0',
          padding: '1rem 1.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '40px',
              height: '40px',
              background: 'linear-gradient(135deg, #0ea5e9, #1e40af)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1.25rem',
              color: 'white'
            }}>
              💧
            </div>
            <div>
              <h1 style={{
                fontSize: '1.125rem',
                fontWeight: '600',
                color: '#1e293b',
                margin: 0
              }}>
                Groundwater Assistant
              </h1>
              <p style={{
                fontSize: '0.75rem',
                color: '#64748b',
                margin: 0
              }}>
                AI-Powered • Voice Enabled
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button 
              onClick={() => setShowLocationMap(true)}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: '500',
                color: '#374151',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = '#f1f5f9'
                e.target.style.borderColor = '#cbd5e1'
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = '#f8fafc'
                e.target.style.borderColor = '#e2e8f0'
              }}
            >
              📍 Location
            </button>
            <button 
              onClick={() => setShowVisualizationPanel(true)}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: '500',
                color: '#374151',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = '#f1f5f9'
                e.target.style.borderColor = '#cbd5e1'
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = '#f8fafc'
                e.target.style.borderColor = '#e2e8f0'
              }}
            >
              📊 Charts
            </button>
            <button 
              onClick={() => signOut(auth)} 
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#ef4444',
                border: '1px solid #dc2626',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: '500',
                color: 'white',
                transition: 'all 0.2s ease'
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = '#dc2626'
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = '#ef4444'
              }}
            >
              Sign out
            </button>
          </div>
        </header>

        {/* Messages Area */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1rem',
          backgroundColor: '#f8fafc'
        }}>
          <div style={{ maxWidth: '800px', margin: '0 auto', paddingBottom: '1rem' }}>
            {messages.map(m => (
              <div key={m.id} style={{
                display: 'flex',
                justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
                marginBottom: '1rem'
              }}>
                <div style={{
                  maxWidth: '70%',
                  padding: '0.75rem 1rem',
                  borderRadius: m.role === 'user' ? '1rem 1rem 0.25rem 1rem' : '1rem 1rem 1rem 0.25rem',
                  backgroundColor: m.role === 'user' ? '#0ea5e9' : 'white',
                  color: m.role === 'user' ? 'white' : '#1e293b',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                  border: m.role === 'user' ? 'none' : '1px solid #e2e8f0',
                  fontSize: '0.875rem',
                  lineHeight: '1.5'
                }}>
                  {m.role === 'assistant' ? (
                    <EnhancedMarkdownRenderer 
                      content={m.text} 
                      analysisData={m.analysisData || analysisData}
                      onVisualizationClick={handleVisualizationClick}
                    />
                  ) : (
                    m.text
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input Form */}
        <div style={{
          backgroundColor: 'white',
          borderTop: '1px solid #e2e8f0',
          padding: '1rem 1.5rem',
          flexShrink: 0
        }}>
          <form onSubmit={handleSend} style={{ maxWidth: '800px', margin: '0 auto' }}>
            <div style={{
              display: 'flex',
              alignItems: 'flex-end',
              gap: '0.75rem',
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '1rem',
              padding: '0.5rem',
              transition: 'all 0.2s ease'
            }}>
            <button
              type="button"
              onClick={() => {
                if (!recording) {
                  // Start recording
                  setRecording(true)
                  setLiveTranscript('')
                  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
                  if (SpeechRecognition) {
                    const recognition = new SpeechRecognition()
                    recognition.continuous = true
                    recognition.interimResults = true
                    recognition.lang = selectedLanguage || 'en'
                    recognition.onresult = (event) => {
                      let interim = ''
                      let final = ''
                      for (let i = 0; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                          final += event.results[i][0].transcript
                        } else {
                          interim += event.results[i][0].transcript
                        }
                      }
                      setLiveTranscript(final + interim)
                    }
                    recognition.onend = () => {
                      setRecording(false)
                      recognitionRef.current = null
                    }
                    recognition.onerror = (e) => {
                      setRecording(false)
                      setLiveTranscript('')
                      recognitionRef.current = null
                    }
                    recognitionRef.current = recognition
                    recognition.start()
                  }
                } else {
                  // Stop recording
                  setRecording(false)
                  if (recognitionRef.current) {
                    recognitionRef.current.stop()
                    recognitionRef.current = null
                  }
                  // Put transcript in input box, don't send automatically
                  if (liveTranscript.trim()) {
                    setInput(liveTranscript.trim())
                    setLiveTranscript('')
                  }
                }
              }}
              className={`w-11 h-11 rounded-xl cursor-pointer flex items-center justify-center text-lg transition-all duration-300 ${
                recording 
                  ? 'border border-red-300 bg-red-50 text-red-600 animate-pulse' 
                  : 'border border-gray-200 bg-gray-50 text-gray-600 hover:border-blue-500 hover:bg-blue-50'
              }`}
              title={recording ? 'Stop voice input' : 'Start voice input'}
            >
              {recording ? '🛑' : '🎤'}
            </button>
              <input
                value={recording ? liveTranscript : input}
                onChange={e => {
                  if (!recording) {
                    setInput(e.target.value)
                  }
                }}
                placeholder={recording ? "Listening... speak now" : "Ask about groundwater data..."}
                style={{
                  flex: 1,
                  padding: '0.75rem 1rem',
                  border: 'none',
                  outline: 'none',
                  backgroundColor: 'transparent',
                  fontSize: '0.875rem',
                  color: '#374151',
                  resize: 'none',
                  maxHeight: '120px',
                  minHeight: '24px'
                }}
                readOnly={recording}
              />
              <button 
                type="submit" 
                disabled={sending} 
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: '#0ea5e9',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.75rem',
                  cursor: sending ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  fontSize: '0.875rem',
                  transition: 'all 0.2s ease',
                  opacity: sending ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
                onMouseOver={(e) => {
                  if (!sending) {
                    e.target.style.backgroundColor = '#0284c7'
                  }
                }}
                onMouseOut={(e) => {
                  if (!sending) {
                    e.target.style.backgroundColor = '#0ea5e9'
                  }
                }}
              >
                {sending ? '⏳' : '→'}
                {sending ? 'Sending' : 'Send'}
              </button>
            </div>
          </form>
        </div>
      </main>
      
      {/* Visualization Panel */}
      <VisualizationPanel 
        isOpen={showVisualizationPanel}
        onClose={() => setShowVisualizationPanel(false)}
      />
      
      {/* Visualization Modal */}
      <VisualizationModal
        isOpen={showVisualizationModal}
        onClose={() => setShowVisualizationModal(false)}
        visualizationData={selectedVisualization}
      />
      
      {/* Location Map Modal */}
      {showLocationMap && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-8">
          <div style={{
            background: 'white',
            borderRadius: '20px',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
            width: '100%',
            maxWidth: '4xl',
            height: '70vh',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            border: '2px solid #e0f2fe'
          }}>
            {/* Header */}
            <div style={{
              padding: '1.5rem',
              borderBottom: '1px solid #e0f2fe',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <h2 style={{
                fontSize: '1.25rem',
                fontWeight: 'bold',
                color: '#1e293b',
                margin: 0
              }}>
                📍 Location-Based Groundwater Analysis
              </h2>
              <button 
                onClick={() => setShowLocationMap(false)}
                style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: 'none',
                  border: 'none',
                  fontSize: '1.25rem',
                  cursor: 'pointer',
                  color: '#64748b',
                  padding: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.target.style.backgroundColor = '#f1f5f9'
                  e.target.style.color = '#0ea5e9'
                }}
                onMouseOut={(e) => {
                  e.target.style.backgroundColor = 'transparent'
                  e.target.style.color = '#64748b'
                }}
              >
                ✕
              </button>
            </div>

            {/* Map Content */}
            <div className="flex-1 p-6">
              <LocationMap
                location={userLocation}
                onLocationChange={handleLocationChange}
                isGettingLocation={isGettingLocation}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Chat1


