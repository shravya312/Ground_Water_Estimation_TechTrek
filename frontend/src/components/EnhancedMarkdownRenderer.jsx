import React from 'react'
import GroundwaterAnalysisCard from './GroundwaterAnalysisCard'

const EnhancedMarkdownRenderer = ({ content, analysisData, onVisualizationClick }) => {
  if (!content) return null

  // Check if this is a structured analysis response
  if (analysisData && (analysisData.criticality_status || analysisData.visualizations)) {
    return (
      <div>
        <GroundwaterAnalysisCard 
          analysisData={analysisData} 
          onVisualizationClick={onVisualizationClick}
        />
        <div style={{ marginTop: '1rem' }}>
          <MarkdownRenderer content={content} />
        </div>
      </div>
    )
  }

  // Fallback to regular markdown rendering
  return <MarkdownRenderer content={content} />
}

// Enhanced markdown renderer with better formatting
const MarkdownRenderer = ({ content }) => {
  if (!content) return null

  // Split content into lines
  const lines = content.split('\n')
  const renderedElements = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i].trim()
    
    // Check if this line is a markdown table header
    if (line.startsWith('|') && line.endsWith('|') && !line.includes('---')) {
      // Find the table separator line
      let separatorIndex = i + 1
      while (separatorIndex < lines.length && 
             !lines[separatorIndex].trim().match(/^\|[\s\-\|]+\|$/)) {
        separatorIndex++
      }
      
      if (separatorIndex < lines.length) {
        // We found a complete table
        const tableLines = []
        let tableEnd = separatorIndex + 1
        
        // Collect all table rows
        for (let j = i; j < lines.length; j++) {
          const currentLine = lines[j].trim()
          if (currentLine.startsWith('|') && currentLine.endsWith('|')) {
            tableLines.push(currentLine)
            tableEnd = j + 1
          } else {
            break
          }
        }
        
        if (tableLines.length >= 2) {
          // Render the table
          const headerCells = tableLines[0].split('|').slice(1, -1).map(cell => cell.trim())
          const dataRows = tableLines.slice(2).map(row => 
            row.split('|').slice(1, -1).map(cell => cell.trim())
          )
          
          renderedElements.push(
            <div key={`table-${i}`} style={{ 
              margin: '2rem 0',
              border: '1px solid var(--color-border)',
              borderRadius: '16px',
              boxShadow: 'var(--shadow-lg)',
              background: 'var(--color-surface)'
            }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '0.9rem',
                backgroundColor: 'var(--color-surface)'
              }}>
                <thead>
                  <tr style={{ 
                    background: 'var(--gradient-primary)',
                    color: 'white'
                  }}>
                    {headerCells.map((cell, idx) => (
                      <th key={idx} style={{
                        padding: '18px 24px',
                        textAlign: 'left',
                        fontWeight: '700',
                        fontSize: '0.95rem',
                        letterSpacing: '0.025em',
                        textTransform: 'uppercase'
                      }}>
                        {cell}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataRows.map((row, rowIdx) => (
                    <tr key={rowIdx} style={{
                      backgroundColor: rowIdx % 2 === 0 ? 'var(--color-surface)' : 'var(--color-surface-elevated)',
                      transition: 'all 0.2s ease'
                    }}>
                      {row.map((cell, cellIdx) => (
                        <td key={cellIdx} style={{
                          padding: '16px 24px',
                          borderBottom: '1px solid var(--color-border)',
                          color: 'var(--color-text-primary)',
                          lineHeight: '1.6',
                          fontWeight: '500'
                        }}>
                          {cell === 'No data available' ? (
                            <span style={{ 
                              color: 'var(--color-text-muted)', 
                              fontStyle: 'italic',
                              opacity: 0.7
                            }}>
                              No data available
                            </span>
                          ) : (
                            cell
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
          
          i = tableEnd
          continue
        }
      }
    }
    
    // Handle headers with enhanced styling
    if (line.startsWith('### ')) {
      renderedElements.push(
        <h3 key={`h3-${i}`} style={{
          fontSize: '1.1rem',
          fontWeight: '600',
          color: 'var(--color-text-primary)',
          margin: '1.5rem 0 0.75rem 0',
          paddingBottom: '0.5rem',
          borderBottom: '2px solid var(--color-border)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          {line.substring(4)}
        </h3>
      )
    } else if (line.startsWith('## ')) {
      renderedElements.push(
        <h2 key={`h2-${i}`} style={{
          fontSize: '1.25rem',
          fontWeight: '700',
          color: 'var(--color-text-primary)',
          margin: '2rem 0 1rem 0',
          paddingBottom: '0.75rem',
          borderBottom: '3px solid var(--color-primary)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          {line.substring(3)}
        </h2>
      )
    } else if (line.startsWith('# ')) {
      renderedElements.push(
        <h1 key={`h1-${i}`} style={{
          fontSize: '1.5rem',
          fontWeight: '800',
          color: 'var(--color-text-primary)',
          margin: '2.5rem 0 1.25rem 0',
          paddingBottom: '1rem',
          borderBottom: '4px solid var(--color-primary)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          {line.substring(2)}
        </h1>
      )
    } else if (line.startsWith('- ')) {
      // Handle bullet points with enhanced styling
      const bulletText = line.substring(2)
      renderedElements.push(
        <div key={`bullet-${i}`} style={{
          margin: '0.5rem 0',
          paddingLeft: '1.5rem',
          position: 'relative'
        }}>
          <span style={{
            position: 'absolute',
            left: '0.5rem',
            top: '0.5rem',
            width: '6px',
            height: '6px',
            backgroundColor: 'var(--color-primary)',
            borderRadius: '50%'
          }}></span>
          <span style={{ color: 'var(--color-text-primary)' }}>{bulletText}</span>
        </div>
      )
    } else if (line.startsWith('**') && line.endsWith('**')) {
      // Handle bold text with enhanced styling
      renderedElements.push(
        <div key={`bold-${i}`} style={{
          fontWeight: '600',
          color: 'var(--color-text-primary)',
          margin: '0.5rem 0',
          fontSize: '1.05rem'
        }}>
          {line.substring(2, line.length - 2)}
        </div>
      )
    } else if (line.trim() === '---') {
      // Handle horizontal rule
      renderedElements.push(
        <hr key={`hr-${i}`} style={{
          border: 'none',
          height: '2px',
          backgroundColor: 'var(--color-border)',
          margin: '2rem 0'
        }} />
      )
    } else if (line.trim() !== '') {
      // Regular paragraph with enhanced styling
      renderedElements.push(
        <p key={`p-${i}`} style={{
          margin: '0.75rem 0',
          lineHeight: '1.6',
          color: 'var(--color-text-primary)',
          fontSize: '0.95rem'
        }}>
          {line}
        </p>
      )
    } else {
      // Empty line
      renderedElements.push(<br key={`br-${i}`} />)
    }
    
    i++
  }

  return (
    <div className="markdown-content" style={{ 
      lineHeight: '1.6',
      color: 'var(--color-text-primary) !important'
    }}>
      {renderedElements}
    </div>
  )
}

export default EnhancedMarkdownRenderer
