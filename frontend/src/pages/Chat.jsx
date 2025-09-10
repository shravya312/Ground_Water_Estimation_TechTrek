import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { signOut } from 'firebase/auth'
import { auth } from '../firebase'

function Chat() {
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', text: 'Hi! Ask me anything about groundwater estimation.' },
  ])
  const [input, setInput] = useState('')
  const [history, setHistory] = useState([])
  const [sending, setSending] = useState(false)
  const bottomRef = useRef(null)

  async function handleSend(e) {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) return
    const userMsg = { id: Date.now(), role: 'user', text: trimmed }
    setMessages(prev => [...prev, userMsg])
    setHistory(prev => [trimmed, ...prev.slice(0, 19)])
    setInput('')
    try {
      setSending(true)
      const apiBase = import.meta.env?.VITE_API_URL || ''
      const res = await fetch(`${apiBase}/ask`, {
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
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', text: String(answer) }])
    } catch (err) {
      const msg = err?.message ? `Error: ${err.message}` : 'Error contacting server.'
      setMessages(prev => [...prev, { id: Date.now() + 2, role: 'assistant', text: msg }])
    } finally {
      setSending(false)
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
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ margin: 0 }}>History</h3>
          <Link to="/" style={{ color: '#0f172a' }}>Home</Link>
        </div>
        <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>
          {history.length === 0 && (
            <li style={{ opacity: 0.7 }}>No recent prompts</li>
          )}
          {history.map((item, idx) => (
            <li key={idx} className="glass" style={{ padding: '0.6rem 0.8rem', marginBottom: '0.6rem', background: 'rgba(252,250,240,0.6)' }}>
              {item}
            </li>
          ))}
        </ul>
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
                  {m.text}
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


