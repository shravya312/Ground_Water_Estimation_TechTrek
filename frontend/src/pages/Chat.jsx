import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { signOut, onAuthStateChanged } from 'firebase/auth'
import { auth, db } from '../firebase'
import { collection, onSnapshot, orderBy, query } from 'firebase/firestore'
import { saveChatMessage, getUserChatHistory, clearChatHistory } from '../services/chatHistoryService'
import MarkdownRenderer from '../components/MarkdownRenderer'

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
  const bottomRef = useRef(null)

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
      const res = await fetch(`${apiBase}/ask-formatted`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed })
      })
      let data
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
        width: 280,
        background: 'linear-gradient(180deg, var(--color-taupe), var(--color-blue-gray))',
        color: '#0f172a',
        padding: '1rem',
        overflowY: 'auto'
      }}>
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
      <main className="chat-main" style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--color-cream)' }}>
        <header className="glass" style={{ padding: '1rem 1.25rem', margin: '1rem', borderRadius: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <h2 style={{ margin: 0, color: 'var(--color-slate)' }}>Chatbot</h2>
            <button onClick={() => signOut(auth)} style={{ borderRadius: 8 }}>Sign out</button>
          </div>
        </header>

        <div className="chat-messages" style={{ flex: 1, overflowY: 'auto', padding: '0 1rem' }}>
          <div className="chat-messages-inner" style={{ maxWidth: 860, margin: '0 auto' }}>
            {messages.map(m => (
              <div key={m.id} className="fade-in-up" style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', margin: '0.5rem 0' }}>
                <div className="glass" style={{
                  background: m.role === 'user' ? 'rgba(153,176,176,0.25)' : 'rgba(252,250,240,0.8)',
                  color: '#0f172a',
                  padding: '0.75rem 1rem',
                  borderRadius: 12,
                  maxWidth: '75%'
                }}>
                  {m.role === 'assistant' ? (
                    <MarkdownRenderer content={m.text} />
                  ) : (
                    m.text
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <form onSubmit={handleSend} className="pulse-focus" style={{ padding: '1rem' }}>
          <div className="glass chat-input" style={{ display: 'flex', gap: 8, maxWidth: 860, margin: '0 auto', padding: 8 }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your message..."
              style={{
                flex: 1,
                border: '1px solid rgba(138,158,160,0.4)',
                padding: '0.8rem 1rem',
                borderRadius: 10,
                background: 'rgba(252,250,240,0.9)'
              }}
            />
            <button type="submit" className="chat-send" disabled={sending} style={{ borderRadius: 10 }}>
              {sending ? 'Sending…' : 'Send'}
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}

export default Chat


