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
        gap: '8px',
        padding: '0.5rem 1rem',
        background: '#f8fafc',
        borderRadius: '0.5rem',
        border: '1px solid #e2e8f0',
        transition: 'all 0.2s ease',
        fontSize: '0.875rem',
        fontWeight: '500',
        color: '#374151',
        cursor: 'pointer'
      }}
      onMouseOver={(e) => {
        e.target.style.backgroundColor = '#f0f9ff'
        e.target.style.borderColor = '#0ea5e9'
      }}
      onMouseOut={(e) => {
        e.target.style.backgroundColor = '#f8fafc'
        e.target.style.borderColor = '#e2e8f0'
      }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        color: '#374151',
        fontSize: '0.875rem',
        fontWeight: '500'
      }}>
        <span>🌐</span>
        <span>Language:</span>
      </div>
      
      <select
        value={selectedLanguage}
        onChange={(e) => onLanguageChange(e.target.value)}
        style={{
          background: 'white',
          border: '1px solid #d1d5db',
          borderRadius: '0.375rem',
          padding: '0.375rem 0.75rem',
          fontSize: '0.875rem',
          color: '#374151',
          cursor: 'pointer',
          outline: 'none',
          transition: 'all 0.2s ease',
          minWidth: '120px'
        }}
        onFocus={(e) => {
          e.target.style.borderColor = '#0ea5e9'
          e.target.style.boxShadow = '0 0 0 3px rgba(14, 165, 233, 0.1)'
        }}
        onBlur={(e) => {
          e.target.style.borderColor = '#d1d5db'
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
