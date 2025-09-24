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

function Chat() {
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
    const trimmed = input.trim()
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
    <div className="chat-root" style={{ display: 'flex', height: '100vh', maxHeight: '100vh', overflow: 'hidden' }}>
      {/* History panel */}
      <aside className="chat-sidebar" style={{
        width: 320,
        background: 'var(--gradient-surface)',
        color: 'var(--color-text-primary)',
        padding: '1.5rem',
        overflowY: 'auto',
        borderRight: '1px solid var(--color-border)',
        boxShadow: 'var(--shadow-md)'
      }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <LanguageSelector 
            selectedLanguage={selectedLanguage}
            onLanguageChange={setSelectedLanguage}
            className="sidebar-language-selector"
          />
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0 }}>Chat History</h3>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {history.length > 0 && (
              <button 
                onClick={handleClearHistory}
                style={{
                  background: 'rgba(220, 38, 38, 0.1)',
                  border: '1px solid rgba(220, 38, 38, 0.3)',
                  color: '#dc2626',
                  padding: '0.25rem 0.5rem',
                  borderRadius: 4,
                  fontSize: '0.75rem',
                  cursor: 'pointer'
                }}
                title="Clear all chat history"
              >
                Clear
              </button>
            )}
            <Link to="/" style={{ color: '#0f172a', fontSize: '0.9rem' }}>Home</Link>
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
            padding: '0.6rem',
            marginBottom: '1rem',
            background: selectedHistoryIndex === -1 ? 'rgba(153,176,176,0.4)' : 'rgba(153,176,176,0.3)',
            border: selectedHistoryIndex === -1 ? '1px solid rgba(153,176,176,0.6)' : '1px solid rgba(153,176,176,0.5)',
            borderRadius: 6,
            color: '#0f172a',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '0.85rem',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            if (selectedHistoryIndex !== -1) {
              e.target.style.background = 'rgba(153,176,176,0.4)'
              e.target.style.transform = 'translateY(-1px)'
            }
          }}
          onMouseOut={(e) => {
            if (selectedHistoryIndex !== -1) {
              e.target.style.background = 'rgba(153,176,176,0.3)'
              e.target.style.transform = 'translateY(0)'
            }
          }}
        >
          + New Chat
        </button>
        
        {loadingHistory ? (
          <div style={{ textAlign: 'center', opacity: 0.7, padding: '1rem' }}>
            Loading chat history...
          </div>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>
            {history.length === 0 && (
              <li style={{ opacity: 0.7, padding: '1rem', textAlign: 'center' }}>
                No chat history yet. Start a conversation!
              </li>
            )}
            {history.map((item, idx) => (
              <li key={idx} className="glass" style={{ 
                padding: '0.6rem 0.8rem', 
                marginBottom: '0.6rem', 
                background: selectedHistoryIndex === idx ? 'rgba(153,176,176,0.4)' : 'rgba(252,250,240,0.6)',
                borderRadius: 6,
                fontSize: '0.85rem',
                lineHeight: 1.3,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                border: selectedHistoryIndex === idx ? '1px solid rgba(153,176,176,0.6)' : '1px solid transparent'
              }}
              onClick={() => loadConversationFromHistory(item, idx)}
              onMouseOver={(e) => {
                if (selectedHistoryIndex !== idx) {
                  e.target.style.background = 'rgba(153,176,176,0.3)'
                  e.target.style.border = '1px solid rgba(153,176,176,0.5)'
                  e.target.style.transform = 'translateY(-1px)'
                }
              }}
              onMouseOut={(e) => {
                if (selectedHistoryIndex !== idx) {
                  e.target.style.background = 'rgba(252,250,240,0.6)'
                  e.target.style.border = '1px solid transparent'
                  e.target.style.transform = 'translateY(0)'
                }
              }}
              title="Click to load this conversation"
              >
                {typeof item === 'string' ? item : (item?.text || '(no prompt)')}
              </li>
            ))}
          </ul>
        )}
      </aside>

      {/* Chat area */}
      <main className="chat-main" style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--color-background)' }}>
        <header className="glass" style={{ padding: '1.5rem 2rem', margin: '1.5rem', borderRadius: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{
                width: 40,
                height: 40,
                borderRadius: '50%',
                background: 'var(--gradient-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '1.2rem',
                fontWeight: '600'
              }}>
                💧
              </div>
              <h2 style={{ margin: 0, color: 'var(--color-text-primary)', fontSize: '1.5rem', fontWeight: '700' }}>Groundwater Assistant</h2>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <button 
                onClick={() => setShowLocationMap(true)}
                style={{ 
                  padding: '0.75rem 1.5rem',
                  fontSize: '0.9rem',
                  background: 'var(--gradient-secondary)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s ease'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-1px)'
                  e.target.style.boxShadow = 'var(--shadow-lg)'
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)'
                  e.target.style.boxShadow = 'none'
                }}
              >
                📍 Location Analysis
              </button>
              <button 
                onClick={() => setShowVisualizationPanel(true)}
                style={{ 
                  padding: '0.75rem 1.5rem',
                  fontSize: '0.9rem',
                  background: 'var(--gradient-primary)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s ease'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-1px)'
                  e.target.style.boxShadow = 'var(--shadow-lg)'
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)'
                  e.target.style.boxShadow = 'none'
                }}
              >
                📊 Visualizations
              </button>
              <LanguageSelector 
                selectedLanguage={selectedLanguage}
                onLanguageChange={setSelectedLanguage}
                className="header-language-selector"
              />
              <button onClick={() => signOut(auth)} style={{ 
                padding: '0.75rem 1.5rem',
                fontSize: '0.9rem',
                background: 'var(--gradient-secondary)'
              }}>Sign out</button>
            </div>
          </div>
        </header>

        <div className="chat-messages" style={{ flex: 1, overflowY: 'auto', padding: '0 1rem' }}>
          <div className="chat-messages-inner" style={{ maxWidth: 860, margin: '0 auto' }}>
            {messages.map(m => (
              <div key={m.id} className="fade-in-up" style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', margin: '0.5rem 0' }}>
                <div className="glass" style={{
                  background: m.role === 'user' 
                    ? 'var(--gradient-primary)' 
                    : 'var(--color-surface)',
                  color: m.role === 'user' ? 'white' : 'var(--color-text-primary)',
                  padding: '1.25rem 1.5rem',
                  borderRadius: 20,
                  maxWidth: '80%',
                  border: m.role === 'user' ? 'none' : '1px solid var(--color-border)',
                  boxShadow: m.role === 'user' 
                    ? 'var(--shadow-lg), 0 0 0 1px rgba(255, 255, 255, 0.2)' 
                    : 'var(--shadow-md)',
                  position: 'relative',
                  marginBottom: '1rem'
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

        <form onSubmit={handleSend} className="pulse-focus" style={{ padding: '1.5rem' }}>
          <div className="glass chat-input" style={{ 
            display: 'flex', 
            gap: 16, 
            maxWidth: 900, 
            margin: '0 auto', 
            padding: 20,
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            boxShadow: 'var(--shadow-lg)',
            borderRadius: 24
          }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask about groundwater data..."
              style={{
                flex: 1,
                border: '1px solid var(--color-border)',
                padding: '1rem 1.25rem',
                borderRadius: 16,
                background: 'var(--color-surface-elevated)',
                fontSize: '1rem',
                color: 'var(--color-text-primary)',
                outline: 'none',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                fontWeight: '500'
              }}
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--color-primary)';
                e.target.style.boxShadow = '0 0 0 4px rgba(59, 130, 246, 0.2)';
                e.target.style.background = 'var(--color-surface)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'var(--color-border)';
                e.target.style.boxShadow = 'none';
                e.target.style.background = 'var(--color-surface-elevated)';
              }}
            />
            <button 
              type="submit" 
              className="chat-send" 
              disabled={sending} 
              style={{ 
                borderRadius: 12,
                padding: '0.875rem 1.5rem',
                fontSize: '1rem',
                fontWeight: '600'
              }}
            >
              {sending ? 'Sending…' : 'Send'}
            </button>
          </div>
        </form>
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
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '2rem'
        }}>
          <div style={{
            background: 'var(--color-surface)',
            borderRadius: 20,
            boxShadow: 'var(--shadow-xl)',
            width: '100%',
            maxWidth: '800px',
            height: '70vh',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}>
            {/* Header */}
            <div style={{
              padding: '1.5rem 2rem',
              borderBottom: '1px solid var(--color-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <h2 style={{ margin: 0, color: 'var(--color-text-primary)' }}>
                📍 Location-Based Groundwater Analysis
              </h2>
              <button
                onClick={() => setShowLocationMap(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  color: 'var(--color-text-secondary)',
                  padding: '0.5rem',
                  borderRadius: '50%',
                  width: '40px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                onMouseOver={(e) => {
                  e.target.style.background = 'var(--color-surface-elevated)'
                }}
                onMouseOut={(e) => {
                  e.target.style.background = 'none'
                }}
              >
                ✕
              </button>
            </div>

            {/* Map Content */}
            <div style={{ flex: 1, padding: '1.5rem' }}>
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

export default Chat


