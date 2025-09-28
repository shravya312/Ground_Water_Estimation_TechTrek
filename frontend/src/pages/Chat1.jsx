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
    { 
      id: 1, 
      role: 'assistant', 
      text: '# Welcome to **Jal**Sanchay! 🌊\n\n**Your Smart Groundwater Assistant**\n\nI\'m here to help you with comprehensive groundwater analysis and estimation. Whether you\'re a researcher, environmental consultant, or water resource manager, I can assist you with:\n\n## 🎯 **Core Services**\n\n**📍 Location Intelligence**\nAnalyze groundwater potential, aquifer characteristics, and site-specific water resources\n\n**📊 Data Visualization**\nTransform complex groundwater data into interactive charts, maps, and reports\n\n**🔬 Advanced Estimation**\nApply scientific methods for groundwater recharge, storage, and flow calculations\n\n**💧 Quality Assessment**\nEvaluate water quality parameters, contamination risks, and treatment needs\n\n**🌱 Sustainability Planning**\nDevelop strategies for sustainable water resource management and conservation\n\n---\n\n**Ready to dive in?** Choose a service below or ask me anything about groundwater! 🚀' 
    },
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
  const [sidebarOpen, setSidebarOpen] = useState(true)
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

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
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
      fontFamily: '"Lato", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      overflow: 'hidden'
    }}>
      {/* Sidebar */}
      <aside style={{
        width: sidebarOpen ? '320px' : '0px',
        backgroundColor: 'white',
        borderRight: '1px solid #e2e8f0',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        transition: 'width 0.3s ease',
        overflow: 'hidden'
      }}>
        {/* Sidebar Header */}
        <div style={{
          padding: '1.5rem',
          borderBottom: '1px solid #e2e8f0',
          backgroundColor: '#f8fafc',
          position: 'relative'
        }}>
          {/* Close Button */}
          <button
            onClick={toggleSidebar}
            style={{
              position: 'absolute',
              top: '1rem',
              right: '1rem',
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              color: '#6b7280',
              cursor: 'pointer',
              padding: '0.25rem',
              borderRadius: '50%',
              width: '32px',
              height: '32px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => {
              e.target.style.color = '#ef4444'
              e.target.style.backgroundColor = 'rgba(239, 68, 68, 0.1)'
            }}
            onMouseOut={(e) => {
              e.target.style.color = '#6b7280'
              e.target.style.backgroundColor = 'transparent'
            }}
            title="Close sidebar"
          >
            ✕
          </button>
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
                  fontSize: '1rem',
                  fontWeight: '800',
                  margin: 0,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.125rem'
                }}>
                  <span style={{ 
                    color: '#1e40af',
                    fontWeight: '900',
                    textShadow: '0 2px 4px rgba(30, 64, 175, 0.3)'
                  }}>Jal</span>
                  <span style={{ 
                    color: '#0ea5e9',
                    fontWeight: '800',
                    textShadow: '0 2px 4px rgba(14, 165, 233, 0.3)'
                  }}>Sanchay</span>
                </h3>
              </div>
            </div>
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
                    padding: '6px 12px',
                    backgroundColor: '#E0F2FE',
                    border: 'none',
                    color: '#1E40AF',
                    fontSize: '0.75rem',
                    fontWeight: '600',
                    borderRadius: '15px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseOver={(e) => e.target.style.backgroundColor = '#BAE6FD'}
                  onMouseOut={(e) => e.target.style.backgroundColor = '#E0F2FE'}
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
              setMessages([{ 
                id: 1, 
                role: 'assistant', 
                text: '# Welcome to **Jal**Sanchay! 🌊\n\n**Your Smart Groundwater Assistant**\n\nI\'m here to help you with comprehensive groundwater analysis and estimation. Whether you\'re a researcher, environmental consultant, or water resource manager, I can assist you with:\n\n## 🎯 **Core Services**\n\n**📍 Location Intelligence**\nAnalyze groundwater potential, aquifer characteristics, and site-specific water resources\n\n**📊 Data Visualization**\nTransform complex groundwater data into interactive charts, maps, and reports\n\n**🔬 Advanced Estimation**\nApply scientific methods for groundwater recharge, storage, and flow calculations\n\n**💧 Quality Assessment**\nEvaluate water quality parameters, contamination risks, and treatment needs\n\n**🌱 Sustainability Planning**\nDevelop strategies for sustainable water resource management and conservation\n\n---\n\n**Ready to dive in?** Choose a service below or ask me anything about groundwater! 🚀' 
              }])
              setSelectedHistoryIndex(-1)
              setCurrentConversationId(null)
            }}
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: '20px',
              border: 'none',
              backgroundColor: '#E0F2FE',
              color: '#1E40AF',
              fontSize: '0.875rem',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
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
          background: 'linear-gradient(135deg, rgba(30, 64, 175, 0.9) 0%, rgba(59, 130, 246, 0.8) 50%, rgba(14, 165, 233, 0.9) 100%)',
          borderBottom: 'none',
          padding: '2rem 2rem 3rem 2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
          boxShadow: '0 8px 32px rgba(30, 64, 175, 0.4)',
          position: 'relative',
          overflow: 'hidden',
          clipPath: 'polygon(0 0, 100% 0, 100% 85%, 0 100%)',
          marginBottom: '-1rem'
        }}>
          {/* Background Images and Patterns */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `
              radial-gradient(circle at 10% 20%, rgba(255, 255, 255, 0.15) 0%, transparent 40%),
              radial-gradient(circle at 90% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
              radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.2) 0%, transparent 60%)
            `,
            pointerEvents: 'none'
          }} />
          
          {/* Wave Pattern Overlay */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
            pointerEvents: 'none'
          }} />
          
          {/* Floating Elements */}
          <div style={{
            position: 'absolute',
            top: '20%',
            right: '10%',
            width: '100px',
            height: '100px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '50%',
            filter: 'blur(20px)',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute',
            bottom: '10%',
            left: '15%',
            width: '60px',
            height: '60px',
            background: 'rgba(14, 165, 233, 0.2)',
            borderRadius: '50%',
            filter: 'blur(15px)',
            pointerEvents: 'none'
          }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', position: 'relative', zIndex: 2 }}>
            {/* Menu Button */}
            {!sidebarOpen && (
              <button
                onClick={toggleSidebar}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                  border: '2px solid rgba(255, 255, 255, 0.3)',
                  fontSize: '1.5rem',
                  color: 'white',
                  cursor: 'pointer',
                  padding: '0.5rem',
                  borderRadius: '0.5rem',
                  marginRight: '0.75rem',
                  transition: 'all 0.3s ease',
                  backdropFilter: 'blur(10px)',
                  boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
                }}
                title="Open sidebar"
              >
                ☰
              </button>
            )}
            <div>
              <h1 style={{
                fontSize: '1.75rem',
                fontWeight: '800',
                margin: 0,
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                textShadow: '0 4px 12px rgba(0, 0, 0, 0.4)'
              }}>
                <span style={{ 
                  color: '#ffffff',
                  fontWeight: '900',
                  textShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
                }}>Jal</span>
                <span style={{ 
                  color: '#e0f2fe',
                  fontWeight: '800',
                  textShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
                }}>Sanchay</span>
              </h1>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', position: 'relative', zIndex: 2 }}>
            <button 
              onClick={() => setShowLocationMap(true)}
              style={{
                padding: '10px 18px',
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '25px',
                cursor: 'pointer',
                fontSize: '0.9rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s ease',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
              }}
              title="Location Analysis"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
              </svg>
              Location
            </button>
            <LanguageSelector 
              selectedLanguage={selectedLanguage}
              onLanguageChange={setSelectedLanguage}
              className="header-language-selector"
            />
            <button 
              onClick={() => setShowVisualizationPanel(true)}
              style={{
                padding: '10px 18px',
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '25px',
                cursor: 'pointer',
                fontSize: '0.9rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s ease',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
              }}
              title="Charts & Visualizations"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v18h18"></path>
                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path>
              </svg>
              Charts
            </button>
            <button 
              onClick={() => signOut(auth)} 
              style={{
                padding: '10px 18px',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                color: 'white',
                border: '2px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '25px',
                cursor: 'pointer',
                fontSize: '0.9rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s ease',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 15px rgba(239, 68, 68, 0.2)'
              }}
              title="Sign out"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                <polyline points="16,17 21,12 16,7"></polyline>
                <line x1="21" y1="12" x2="9" y2="12"></line>
              </svg>
              Sign Out
            </button>
          </div>
          
          {/* Subtle Line at bottom of header */}
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: '10%',
            right: '10%',
            height: '2px',
            background: 'linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.3) 20%, rgba(255, 255, 255, 0.6) 50%, rgba(255, 255, 255, 0.3) 80%, transparent 100%)',
            pointerEvents: 'none'
          }} />
          
          {/* Wave SVG at bottom */}
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: '2rem',
            background: `url("data:image/svg+xml,%3Csvg viewBox='0 0 1200 120' preserveAspectRatio='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1110-46.29,1200,0V120H0Z' fill='%23ffffff' fill-opacity='0.1'/%3E%3C/svg%3E")`,
            backgroundSize: 'cover',
            backgroundRepeat: 'no-repeat',
            pointerEvents: 'none'
          }} />
        </header>

        {/* Messages Area */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1rem',
          background: `linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)`,
          position: 'relative'
        }}>
          {/* Centered Water Image with White Cover */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '300px',
            height: '200px',
            background: `url('/src/water.jpg')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
            borderRadius: '20px',
            pointerEvents: 'none',
            zIndex: 1
          }} />
          
          {/* White Cover Over Water Image */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '300px',
            height: '200px',
            background: 'rgba(255, 255, 255, 0.8)',
            borderRadius: '20px',
            pointerEvents: 'none',
            zIndex: 2
          }} />
          
          {/* Single Water Droplet on Right Side */}
          <div style={{
            position: 'absolute',
            top: '30%',
            right: '30%',
            width: '80px',
            height: '80px',
            background: `url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M40 10c-16.5 0-30 13.5-30 30 0 16.5 13.5 30 30 30s30-13.5 30-30c0-16.5-13.5-30-30-30zm0 5c13.8 0 25 11.2 25 25s-11.2 25-25 25-25-11.2-25-25 11.2-25 25-25z' fill='%231e40af' fill-opacity='0.08'/%3E%3Cpath d='M40 20c-11 0-20 9-20 20s9 20 20 20 20-9 20-20-9-20-20-20zm0 5c8.3 0 15 6.7 15 15s-6.7 15-15 15-15-6.7-15-15 6.7-15 15-15z' fill='%231e40af' fill-opacity='0.12'/%3E%3Cpath d='M40 30c-5.5 0-10 4.5-10 10s4.5 10 10 10 10-4.5 10-10-4.5-10-10-10zm0 3c3.9 0 7 3.1 7 7s-3.1 7-7 7-7-3.1-7-7 3.1-7 7-7z' fill='%231e40af' fill-opacity='0.15'/%3E%3C/svg%3E")`,
            backgroundSize: 'contain',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
            pointerEvents: 'none',
            zIndex: 3
          }} />
          
          <div style={{ maxWidth: '800px', margin: '0 auto', paddingBottom: '1rem', position: 'relative', zIndex: 3 }}>
            {messages.map(m => (
              <div key={m.id}>
                <div style={{
                display: 'flex',
                justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
                  marginBottom: '1rem',
                  alignItems: 'flex-start',
                  gap: '0.5rem'
                }}>
                  {/* Icon for assistant messages */}
                  {m.role === 'assistant' && (
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      backgroundColor: '#E0F2FE',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      marginTop: '0.25rem'
                    }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1E40AF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                        <circle cx="12" cy="5" r="2"/>
                        <path d="M12 7v4"/>
                        <line x1="8" y1="16" x2="8" y2="16"/>
                        <line x1="16" y1="16" x2="16" y2="16"/>
                      </svg>
                    </div>
                  )}
                  
                <div style={{
                  maxWidth: '70%',
                    padding: '1rem 1.25rem',
                    borderRadius: m.role === 'user' ? '1.5rem 1.5rem 0.5rem 1.5rem' : '1.5rem 1.5rem 1.5rem 0.5rem',
                    background: m.role === 'user' 
                      ? 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)'
                      : 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%)',
                  color: m.role === 'user' ? 'white' : '#1e293b',
                    boxShadow: m.role === 'user' 
                      ? '0 8px 25px rgba(14, 165, 233, 0.3), 0 4px 12px rgba(0, 0, 0, 0.1)'
                      : '0 8px 25px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.05)',
                    border: m.role === 'user' ? 'none' : '1px solid rgba(226, 232, 240, 0.5)',
                  fontSize: '0.875rem',
                    lineHeight: '1.6',
                    position: 'relative',
                    backdropFilter: 'blur(10px)',
                    overflow: 'hidden'
                  }}>
                    {/* Background pattern and water droplets for assistant messages */}
                    {m.role === 'assistant' && (
                      <>
                        <div style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          background: `url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%231e40af' fill-opacity='0.02'%3E%3Cpath d='M10 10c0-5.5-4.5-10-10-10s-10 4.5-10 10 4.5 10 10 10 10-4.5 10-10z'/%3E%3C/g%3E%3C/svg%3E")`,
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          top: '12px',
                          left: '15px',
                          width: '6px',
                          height: '6px',
                          background: 'rgba(30, 64, 175, 0.1)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          top: '25px',
                          left: '25px',
                          width: '4px',
                          height: '4px',
                          background: 'rgba(14, 165, 233, 0.08)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          bottom: '20px',
                          left: '20px',
                          width: '5px',
                          height: '5px',
                          background: 'rgba(59, 130, 246, 0.12)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          top: '18px',
                          left: '8px',
                          width: '3px',
                          height: '3px',
                          background: 'rgba(30, 64, 175, 0.15)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                      </>
                    )}
                    
                    {/* Water droplet effects for user messages */}
                    {m.role === 'user' && (
                      <>
                        <div style={{
                          position: 'absolute',
                          top: '10px',
                          right: '15px',
                          width: '8px',
                          height: '8px',
                          background: 'rgba(255, 255, 255, 0.3)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          top: '20px',
                          right: '25px',
                          width: '5px',
                          height: '5px',
                          background: 'rgba(255, 255, 255, 0.2)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          top: '15px',
                          right: '8px',
                          width: '3px',
                          height: '3px',
                          background: 'rgba(255, 255, 255, 0.4)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                        <div style={{
                          position: 'absolute',
                          bottom: '15px',
                          right: '20px',
                          width: '6px',
                          height: '6px',
                          background: 'rgba(255, 255, 255, 0.25)',
                          borderRadius: '50%',
                          pointerEvents: 'none'
                        }} />
                      </>
                    )}
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

                  {/* Icon for user messages */}
                  {m.role === 'user' && (
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      backgroundColor: '#0ea5e9',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      marginTop: '0.25rem'
                    }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                        <circle cx="12" cy="7" r="4"/>
                      </svg>
                    </div>
                  )}
                </div>

                {/* Suggestion prompts for the first assistant message */}
                {m.role === 'assistant' && m.id === 1 && (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '0.75rem',
                    marginLeft: '2.5rem',
                    marginBottom: '1rem',
                    maxWidth: '600px'
                  }}>
                    {[
                      { text: 'Analyze groundwater in my area', icon: '📍', color: '#10B981' },
                      { text: 'Show me groundwater data charts', icon: '📊', color: '#3B82F6' },
                      { text: 'Explain estimation methods', icon: '🔬', color: '#8B5CF6' },
                      { text: 'Check water quality parameters', icon: '💧', color: '#06B6D4' }
                    ].map((suggestion, index) => (
                      <div
                        key={index}
                        onClick={() => setInput(suggestion.text)}
                        style={{
                          padding: '1rem',
                          backgroundColor: 'white',
                          border: `2px solid ${suggestion.color}`,
                          borderRadius: '12px',
                          cursor: 'pointer',
                          transition: 'all 0.3s ease',
                          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.75rem',
                          position: 'relative',
                          overflow: 'hidden'
                        }}
                        onMouseOver={(e) => {
                          e.target.style.transform = 'translateY(-3px)'
                          e.target.style.boxShadow = `0 8px 25px ${suggestion.color}20`
                          e.target.style.borderColor = suggestion.color
                        }}
                        onMouseOut={(e) => {
                          e.target.style.transform = 'translateY(0)'
                          e.target.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)'
                          e.target.style.borderColor = suggestion.color
                        }}
                      >
                        <div style={{
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          backgroundColor: `${suggestion.color}15`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '1.2rem',
                          flexShrink: 0
                        }}>
                          {suggestion.icon}
                        </div>
                        <div style={{
                          flex: 1,
                          fontSize: '0.85rem',
                          fontWeight: '600',
                          color: '#374151',
                          lineHeight: '1.4'
                        }}>
                          {suggestion.text}
                        </div>
                        <div style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          backgroundColor: suggestion.color,
                          flexShrink: 0
                        }} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input Form */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%)',
          borderTop: '1px solid rgba(226, 232, 240, 0.5)',
          padding: '1rem 1.5rem',
          flexShrink: 0,
          position: 'relative',
          backdropFilter: 'blur(10px)'
        }}>
          {/* Water droplets in input area */}
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '20px',
            width: '8px',
            height: '8px',
            background: 'rgba(14, 165, 233, 0.1)',
            borderRadius: '50%',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute',
            top: '20px',
            right: '30px',
            width: '6px',
            height: '6px',
            background: 'rgba(30, 64, 175, 0.08)',
            borderRadius: '50%',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute',
            bottom: '15px',
            left: '50px',
            width: '4px',
            height: '4px',
            background: 'rgba(59, 130, 246, 0.12)',
            borderRadius: '50%',
            pointerEvents: 'none'
          }} />
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
              style={{
                width: '60px',
                height: '50px',
                borderRadius: '30px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.125rem',
                transition: 'all 0.3s ease',
                backgroundColor: recording ? '#FF0000' : '#00BFFF',
                color: '#FFFFFF',
                border: 'none',
                animation: recording ? 'pulse 2s infinite' : 'none',
                boxShadow: '0 6px 12px rgba(0, 0, 0, 0.5)',
                zIndex: 99999,
                position: 'relative',
                outline: 'none',
                fontWeight: 'bold',
                opacity: 1,
                visibility: 'visible',
                minWidth: '60px',
                minHeight: '50px'
              }}
              title={recording ? 'Stop voice input' : 'Start voice input'}
              onMouseOver={(e) => {
                if (!recording) {
                  e.target.style.backgroundColor = '#B0E0E6'
                  e.target.style.borderColor = 'transparent'
                  e.target.style.transform = 'scale(1.05)'
                }
              }}
              onMouseOut={(e) => {
                if (!recording) {
                  e.target.style.backgroundColor = '#87CEEB'
                  e.target.style.borderColor = 'transparent'
                  e.target.style.transform = 'scale(1)'
                }
              }}
            >
              {recording ? (
                <div style={{ 
                  fontSize: '20px', 
                  color: '#FFFFFF', 
                  fontWeight: 'bold',
                  textAlign: 'center',
                  lineHeight: '1'
                }}>⏹</div>
              ) : (
                <div style={{ 
                  fontSize: '20px', 
                  color: '#FFFFFF', 
                  fontWeight: 'bold',
                  textAlign: 'center',
                  lineHeight: '1'
                }}>🎤</div>
              )}
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
                  width: '50px',
                  height: '40px',
                  backgroundColor: '#87CEEB',
                  color: '#000000',
                  border: 'none',
                  borderRadius: '25px',
                  cursor: sending ? 'not-allowed' : 'pointer',
                  opacity: sending ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.125rem',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
                  zIndex: 9999,
                  position: 'relative',
                  outline: 'none',
                  fontWeight: 'bold',
                  visibility: 'visible'
                }}
                onMouseOver={(e) => {
                  if (!sending) {
                    e.target.style.backgroundColor = '#B0E0E6'
                    e.target.style.borderColor = 'transparent'
                    e.target.style.transform = 'scale(1.05)'
                  }
                }}
                onMouseOut={(e) => {
                  if (!sending) {
                    e.target.style.backgroundColor = '#87CEEB'
                    e.target.style.borderColor = 'transparent'
                    e.target.style.transform = 'scale(1)'
                  }
                }}
                title={sending ? 'Sending...' : 'Send message'}
              >
                {sending ? (
                  <div style={{ 
                    fontSize: '24px', 
                    color: '#000000', 
                    fontWeight: 'bold',
                    textAlign: 'center',
                    lineHeight: '1'
                  }}>⏳</div>
                ) : (
                  <div style={{ 
                    fontSize: '24px', 
                    color: '#000000', 
                    fontWeight: 'bold',
                    textAlign: 'center',
                    lineHeight: '1'
                  }}>➤</div>
                )}
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
                margin: 0,
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#374151" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color: '#374151'}}>
                  <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                  <circle cx="12" cy="10" r="3"></circle>
                </svg>
                Location-Based Groundwater Analysis
              </h2>
              <button 
                onClick={() => setShowLocationMap(false)}
                style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  backgroundColor: '#E0F2FE',
                  border: 'none',
                  fontSize: '1.25rem',
                  cursor: 'pointer',
                  color: '#1E40AF',
                  padding: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
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


