import React from 'react'
import PlotlyChart from './PlotlyChart'

const VisualizationModal = ({ isOpen, onClose, visualizationData }) => {
  if (!isOpen || !visualizationData) return null

  const { type, title, data } = visualizationData

  const getVisualizationIcon = (type) => {
    switch (type) {
      case 'pie_chart': return '🥧'
      case 'bar_chart': return '📊'
      case 'gauge_chart': return '🎯'
      default: return '📈'
    }
  }

  const getVisualizationDescription = (type) => {
    switch (type) {
      case 'pie_chart': return 'Interactive pie chart showing distribution data'
      case 'bar_chart': return 'Interactive bar chart comparing different metrics'
      case 'gauge_chart': return 'Interactive gauge chart displaying current status'
      default: return 'Interactive data visualization'
    }
  }

  return (
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
        borderRadius: '20px',
        boxShadow: 'var(--shadow-xl)',
        width: '100%',
        maxWidth: '1200px',
        height: '80vh',
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
          justifyContent: 'space-between',
          background: 'var(--gradient-surface)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '1.5rem' }}>
              {getVisualizationIcon(type)}
            </span>
            <div>
              <h2 style={{ 
                margin: 0, 
                color: 'var(--color-text-primary)',
                fontSize: '1.25rem',
                fontWeight: '700'
              }}>
                {title}
              </h2>
              <p style={{ 
                margin: 0, 
                fontSize: '0.9rem', 
                color: 'var(--color-text-secondary)'
              }}>
                {getVisualizationDescription(type)}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
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
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => {
              e.target.style.background = 'var(--color-surface-elevated)'
              e.target.style.color = 'var(--color-text-primary)'
            }}
            onMouseOut={(e) => {
              e.target.style.background = 'none'
              e.target.style.color = 'var(--color-text-secondary)'
            }}
          >
            ✕
          </button>
        </div>

        {/* Chart Content */}
        <div style={{ 
          flex: 1, 
          padding: '1.5rem',
          background: 'var(--color-background)',
          overflow: 'auto'
        }}>
          <PlotlyChart
            plotData={data}
            loading={false}
            error={null}
            title={title}
            height={500}
          />
        </div>

        {/* Footer with additional info */}
        <div style={{
          padding: '1rem 2rem',
          borderTop: '1px solid var(--color-border)',
          background: 'var(--color-surface-elevated)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>
            💡 Hover over chart elements for detailed information
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>
            📊 Data from INGRES Groundwater Assessment
          </div>
        </div>
      </div>
    </div>
  )
}

export default VisualizationModal
