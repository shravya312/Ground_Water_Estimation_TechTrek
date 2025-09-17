import React from 'react'

// Simple markdown table renderer component
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
              overflowX: 'auto', 
              margin: '1rem 0',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '0.9rem',
                backgroundColor: 'white'
              }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8fafc' }}>
                    {headerCells.map((cell, idx) => (
                      <th key={idx} style={{
                        padding: '0.75rem',
                        textAlign: 'left',
                        borderBottom: '2px solid #e2e8f0',
                        fontWeight: '600',
                        color: '#374151'
                      }}>
                        {cell}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataRows.map((row, rowIdx) => (
                    <tr key={rowIdx} style={{
                      backgroundColor: rowIdx % 2 === 0 ? 'white' : '#f8fafc'
                    }}>
                      {row.map((cell, cellIdx) => (
                        <td key={cellIdx} style={{
                          padding: '0.75rem',
                          borderBottom: '1px solid #e2e8f0',
                          color: '#4b5563'
                        }}>
                          {cell === 'No data available' ? (
                            <span style={{ 
                              color: '#9ca3af', 
                              fontStyle: 'italic' 
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
    
    // Handle headers
    if (line.startsWith('### ')) {
      renderedElements.push(
        <h3 key={`h3-${i}`} style={{
          fontSize: '1.1rem',
          fontWeight: '600',
          color: '#1f2937',
          margin: '1.5rem 0 0.75rem 0',
          paddingBottom: '0.5rem',
          borderBottom: '2px solid #e5e7eb'
        }}>
          {line.substring(4)}
        </h3>
      )
    } else if (line.startsWith('## ')) {
      renderedElements.push(
        <h2 key={`h2-${i}`} style={{
          fontSize: '1.25rem',
          fontWeight: '700',
          color: '#111827',
          margin: '2rem 0 1rem 0',
          paddingBottom: '0.75rem',
          borderBottom: '3px solid #3b82f6'
        }}>
          {line.substring(3)}
        </h2>
      )
    } else if (line.startsWith('# ')) {
      renderedElements.push(
        <h1 key={`h1-${i}`} style={{
          fontSize: '1.5rem',
          fontWeight: '800',
          color: '#0f172a',
          margin: '2.5rem 0 1.25rem 0',
          paddingBottom: '1rem',
          borderBottom: '4px solid #1e40af'
        }}>
          {line.substring(2)}
        </h1>
      )
    } else if (line.startsWith('- ')) {
      // Handle bullet points
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
            backgroundColor: '#3b82f6',
            borderRadius: '50%'
          }}></span>
          <span style={{ color: '#374151' }}>{bulletText}</span>
        </div>
      )
    } else if (line.startsWith('**') && line.endsWith('**')) {
      // Handle bold text
      renderedElements.push(
        <div key={`bold-${i}`} style={{
          fontWeight: '600',
          color: '#1f2937',
          margin: '0.5rem 0'
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
          backgroundColor: '#e5e7eb',
          margin: '2rem 0'
        }} />
      )
    } else if (line.trim() !== '') {
      // Regular paragraph
      renderedElements.push(
        <p key={`p-${i}`} style={{
          margin: '0.75rem 0',
          lineHeight: '1.6',
          color: '#374151'
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
    <div style={{ 
      lineHeight: '1.6',
      color: '#374151'
    }}>
      {renderedElements}
    </div>
  )
}

export default MarkdownRenderer
