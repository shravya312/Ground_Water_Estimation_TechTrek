import React, { useState, useEffect } from 'react'

const LanguageSelector = ({ selectedLanguage, onLanguageChange, className = '' }) => {
  const [languages, setLanguages] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchLanguages = async () => {
      // Set a timeout to prevent indefinite loading
      const timeoutId = setTimeout(() => {
        setLanguages({
          'en': 'English',
          'hi': 'Hindi',
          'bn': 'Bengali',
          'ta': 'Tamil',
          'te': 'Telugu',
          'ml': 'Malayalam',
          'gu': 'Gujarati',
          'mr': 'Marathi',
          'pa': 'Punjabi',
          'kn': 'Kannada',
          'or': 'Odia',
          'as': 'Assamese',
          'ur': 'Urdu',
          'ne': 'Nepali',
          'si': 'Sinhala'
        })
        setLoading(false)
      }, 3000) // 3 second timeout

      try {
        const controller = new AbortController()
        const response = await fetch('http://localhost:8000/languages', {
          signal: controller.signal,
          timeout: 2000
        })
        
        clearTimeout(timeoutId)
        
        if (response.ok) {
          const data = await response.json()
          setLanguages(data.languages)
        } else {
          // Fallback languages if API fails
          setLanguages({
            'en': 'English',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'pa': 'Punjabi',
            'kn': 'Kannada',
            'or': 'Odia',
            'as': 'Assamese',
            'ur': 'Urdu',
            'ne': 'Nepali',
            'si': 'Sinhala'
          })
        }
      } catch (error) {
        clearTimeout(timeoutId)
        console.error('Failed to fetch languages:', error)
        // Fallback languages
        setLanguages({
          'en': 'English',
          'hi': 'Hindi',
          'bn': 'Bengali',
          'ta': 'Tamil',
          'te': 'Telugu',
          'ml': 'Malayalam',
          'gu': 'Gujarati',
          'mr': 'Marathi',
          'pa': 'Punjabi',
          'kn': 'Kannada',
          'or': 'Odia',
          'as': 'Assamese',
          'ur': 'Urdu',
          'ne': 'Nepali',
          'si': 'Sinhala'
        })
      } finally {
        setLoading(false)
      }
    }

    fetchLanguages()
  }, [])

  if (loading) {
    return (
      <div className={`language-selector ${className}`} style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '0.5rem',
        background: 'var(--color-slate-50)',
        borderRadius: '12px',
        border: '1px solid var(--color-border)'
      }}>
        <div style={{
          width: '16px',
          height: '16px',
          border: '2px solid var(--color-primary)',
          borderTop: '2px solid transparent',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}></div>
        <span style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>Loading languages...</span>
      </div>
    )
  }

  return (
      <div className={`language-selector ${className}`} style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '0.75rem 1rem',
        background: 'var(--color-surface-elevated)',
        borderRadius: '12px',
        border: '1px solid var(--color-border)',
        transition: 'all 0.3s ease'
      }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        color: 'var(--color-text-primary)',
        fontSize: '0.9rem',
        fontWeight: '600'
      }}>
        <span>🌐</span>
        <span>Language:</span>
      </div>
      
      <select
        value={selectedLanguage}
        onChange={(e) => onLanguageChange(e.target.value)}
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: '8px',
          padding: '0.5rem 0.75rem',
          fontSize: '0.9rem',
          color: 'var(--color-text-primary)',
          cursor: 'pointer',
          outline: 'none',
          transition: 'all 0.3s ease',
          minWidth: '140px'
        }}
        onFocus={(e) => {
          e.target.style.borderColor = 'var(--color-primary)'
          e.target.style.boxShadow = '0 0 0 3px rgba(14, 165, 233, 0.1)'
        }}
        onBlur={(e) => {
          e.target.style.borderColor = 'var(--color-border)'
          e.target.style.boxShadow = 'none'
        }}
      >
        {Object.entries(languages).map(([code, name]) => (
          <option key={code} value={code}>
            {name}
          </option>
        ))}
      </select>
    </div>
  )
}

export default LanguageSelector
