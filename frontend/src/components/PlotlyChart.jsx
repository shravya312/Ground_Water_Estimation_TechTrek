import React, { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'

const PlotlyChart = ({ 
  plotData, 
  loading = false, 
  error = null, 
  title = "Data Visualization",
  height = 600,
  className = ""
}) => {
  const [plotConfig, setPlotConfig] = useState({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    responsive: true
  })

  const [plotLayout, setPlotLayout] = useState({
    autosize: true,
    margin: { l: 50, r: 50, t: 50, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, system-ui, sans-serif',
      size: 12,
      color: '#FFFFFF'
    },
    title: {
      text: title,
      font: {
        size: 16,
        color: '#FFFFFF'
      }
    },
    xaxis: {
      color: '#FFFFFF',
      gridcolor: 'rgba(255,255,255,0.2)',
      linecolor: '#FFFFFF',
      tickcolor: '#FFFFFF',
      title: {
        font: {
          color: '#FFFFFF'
        }
      }
    },
    yaxis: {
      color: '#FFFFFF',
      gridcolor: 'rgba(255,255,255,0.2)',
      linecolor: '#FFFFFF',
      tickcolor: '#FFFFFF',
      title: {
        font: {
          color: '#FFFFFF'
        }
      }
    }
  })

  useEffect(() => {
    if (plotData) {
      try {
        const parsedData = typeof plotData === 'string' ? JSON.parse(plotData) : plotData
        setPlotLayout(prev => ({
          ...prev,
          title: {
            text: title,
            font: {
              size: 16,
              color: '#FFFFFF'
            }
          }
        }))
      } catch (err) {
        console.error('Error parsing plot data:', err)
      }
    }
  }, [plotData, title])

  if (loading) {
    return (
      <div className={`plotly-chart-loading ${className}`} style={{
        height: height,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 12,
        margin: '1rem 0'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
          color: 'var(--color-text-secondary)'
        }}>
          <div style={{
            width: 40,
            height: 40,
            border: '3px solid var(--color-border)',
            borderTop: '3px solid var(--color-primary)',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <p style={{ margin: 0, fontSize: '0.9rem' }}>Loading visualization...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`plotly-chart-error ${className}`} style={{
        height: height,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(220, 38, 38, 0.05)',
        border: '1px solid rgba(220, 38, 38, 0.2)',
        borderRadius: 12,
        margin: '1rem 0',
        color: '#dc2626'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '0.5rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '2rem' }}>⚠️</div>
          <p style={{ margin: 0, fontSize: '0.9rem' }}>
            Error loading visualization: {error}
          </p>
        </div>
      </div>
    )
  }

  if (!plotData) {
    return (
      <div className={`plotly-chart-empty ${className}`} style={{
        height: height,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 12,
        margin: '1rem 0',
        color: 'var(--color-text-secondary)'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '0.5rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '2rem' }}>📊</div>
          <p style={{ margin: 0, fontSize: '0.9rem' }}>
            No visualization data available
          </p>
        </div>
      </div>
    )
  }

  try {
    const parsedData = typeof plotData === 'string' ? JSON.parse(plotData) : plotData
    
    return (
      <div className={`plotly-chart ${className}`} style={{
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 12,
        margin: '1rem 0',
        overflow: 'hidden'
      }}>
        <Plot
          data={parsedData.data || []}
          layout={{
            ...plotLayout,
            ...parsedData.layout,
            height: height,
            autosize: true
          }}
          config={plotConfig}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>
    )
  } catch (err) {
    console.error('Error rendering plot:', err)
    return (
      <div className={`plotly-chart-error ${className}`} style={{
        height: height,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(220, 38, 38, 0.05)',
        border: '1px solid rgba(220, 38, 38, 0.2)',
        borderRadius: 12,
        margin: '1rem 0',
        color: '#dc2626'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '0.5rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '2rem' }}>⚠️</div>
          <p style={{ margin: 0, fontSize: '0.9rem' }}>
            Error rendering visualization: Invalid data format
          </p>
        </div>
      </div>
    )
  }
}

export default PlotlyChart
