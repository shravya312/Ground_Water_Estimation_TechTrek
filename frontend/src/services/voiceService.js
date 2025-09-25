const API_BASE_URL = import.meta.env?.VITE_API_URL || ''

// Simple Web Speech API wrapper (if available)
function getBrowserSpeechRecognition() {
  const w = window
  const SpeechRecognition = w.SpeechRecognition || w.webkitSpeechRecognition
  if (!SpeechRecognition) return null
  const rec = new SpeechRecognition()
  rec.continuous = false
  rec.interimResults = false
  rec.maxAlternatives = 1
  rec.lang = 'auto' // browsers may ignore auto; UI will determine lang
  return rec
}

export const voiceService = {
  async recordAndTranscribe({ preferredLang } = {}) {
    // Try Web Speech API first
    const rec = getBrowserSpeechRecognition()
    if (rec) {
      return new Promise((resolve, reject) => {
        let finished = false
        rec.onresult = (e) => {
          if (finished) return
          finished = true
          const text = e.results?.[0]?.[0]?.transcript || ''
          rec.stop()
          resolve({ text, language: preferredLang || 'auto', source: 'browser' })
        }
        rec.onerror = (err) => {
          if (finished) return
          finished = true
          rec.stop()
          reject(err.error || 'speech_recognition_error')
        }
        rec.onend = () => {
          if (!finished) {
            reject('no_speech_detected')
          }
        }
        try {
          if (preferredLang) rec.lang = preferredLang
          rec.start()
        } catch (e) {
          reject(e)
        }
      })
    }

    // Fallback: record audio and send to backend Whisper
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const chunks = []
    const recorder = new MediaRecorder(stream)
    return new Promise((resolve, reject) => {
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data) }
      recorder.onstop = async () => {
        try {
          const blob = new Blob(chunks, { type: 'audio/webm' })
          const form = new FormData()
          form.append('audio', blob, 'speech.webm')
          const res = await fetch(`${API_BASE_URL}/voice/transcribe`, { method: 'POST', body: form })
          const ctype = res.headers.get('content-type') || ''
          const raw = await res.text()
          let data = {}
          try { data = ctype.includes('application/json') && raw ? JSON.parse(raw) : {} } catch { data = {} }
          if (!res.ok) throw new Error(data?.detail || raw || 'Transcription failed')
          resolve({ text: data.text, language: data.language || 'auto', source: 'backend' })
        } catch (err) {
          reject(err)
        } finally {
          stream.getTracks().forEach(t => t.stop())
        }
      }
      recorder.start()
      // Auto stop after 10s
      setTimeout(() => { if (recorder.state === 'recording') recorder.stop() }, 10000)
    })
  },

  async voiceChat({ preferredLang } = {}) {
    // Uses fallback recorder; if browser STT used, we just send text to backend for full pipeline
    try {
      const stt = await this.recordAndTranscribe({ preferredLang })
      if (stt.source === 'browser') {
        const res = await fetch(`${API_BASE_URL}/voice/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: stt.text, language: preferredLang || stt.language || 'auto' })
        })
        const ctype = res.headers.get('content-type') || ''
        const raw = await res.text()
        let data = {}
        try { data = ctype.includes('application/json') && raw ? JSON.parse(raw) : {} } catch { data = {} }
        if (!res.ok) throw new Error(data?.detail || raw || 'Voice chat failed')
        return data
      }

      // If backend STT already done, call voice/chat with audio route next time; here we use the result
      // For simplicity, re-call as text to ensure uniform pipeline
      const res = await fetch(`${API_BASE_URL}/voice/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: stt.text, language: preferredLang || stt.language || 'auto' })
      })
      const ctype = res.headers.get('content-type') || ''
      const raw = await res.text()
      let data = {}
      try { data = ctype.includes('application/json') && raw ? JSON.parse(raw) : {} } catch { data = {} }
      if (!res.ok) throw new Error(data?.detail || raw || 'Voice chat failed')
      return data
    } catch (e) {
      console.error('voiceChat error', e)
      throw e
    }
  }
}

export default voiceService