import React, { useState, useEffect } from 'react'
import PlotlyChart from './PlotlyChart'
import visualizationService from '../services/visualizationService'

const VisualizationPanel = ({ isOpen, onClose }) => {
  const [activeVisualization, setActiveVisualization] = useState(null)
  const [plotData, setPlotData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [availableStates, setAvailableStates] = useState([])
  const [availableMetrics, setAvailableMetrics] = useState([])
  const [selectedState, setSelectedState] = useState('')
  const [selectedMetric, setSelectedMetric] = useState('Annual Ground water Recharge (ham) - Total - Total')

  // Load available states and metrics on component mount
  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const [statesResponse, metricsResponse] = await Promise.all([
          visualizationService.getAvailableStates(),
          visualizationService.getAvailableMetrics()
        ])
        
        if (statesResponse.success) {
          setAvailableStates(statesResponse.states || [])
        }
        
        if (metricsResponse.success) {
          setAvailableMetrics(metricsResponse.metrics || [])
        }
      } catch (err) {
        console.error('Error loading metadata:', err)
      }
    }

    if (isOpen) {
      loadMetadata()
    }
  }, [isOpen])

  const handleVisualizationClick = async (visualizationType) => {
    setActiveVisualization(visualizationType)
    setLoading(true)
    setError(null)
    setPlotData(null)

    try {
      let response
      
      switch (visualizationType) {
        case 'overview':
          response = await visualizationService.getOverviewDashboard()
          break
        case 'state-analysis':
          response = await visualizationService.getStateAnalysis(selectedState || null)
          break
        case 'temporal-analysis':
          response = await visualizationService.getTemporalAnalysis()
          break
        case 'geographical-heatmap':
          response = await visualizationService.getGeographicalHeatmap(selectedMetric)
          break
        case 'correlation-matrix':
          response = await visualizationService.getCorrelationMatrix()
          break
        case 'statistical-summary':
          response = await visualizationService.getStatisticalSummary()
          break
        default:
          throw new Error('Unknown visualization type')
      }

      if (response.success && response.plot_json) {
        setPlotData(response.plot_json)
      } else {
        throw new Error(response.message || 'Failed to load visualization')
      }
    } catch (err) {
      console.error(`Error loading ${visualizationType}:`, err)
      setError(err.message || 'Failed to load visualization')
    } finally {
      setLoading(false)
    }
  }

  const getVisualizationTitle = (type) => {
    const titles = {
      'overview': 'Overview Dashboard',
      'state-analysis': 'State Analysis',
      'temporal-analysis': 'Temporal Analysis',
      'geographical-heatmap': 'Geographical Heatmap',
      'correlation-matrix': 'Correlation Matrix',
      'statistical-summary': 'Statistical Summary'
    }
    return titles[type] || 'Visualization'
  }

  const visualizationButtons = [
    {
      id: 'overview',
      label: '📊 Overview Dashboard',
      description: 'Comprehensive groundwater data overview'
    },
    {
      id: 'state-analysis',
      label: '🏛️ State Analysis',
      description: 'Detailed analysis by state'
    },
    {
      id: 'temporal-analysis',
      label: '📈 Temporal Analysis',
      description: 'Trends over time'
    },
    {
      id: 'geographical-heatmap',
      label: '🗺️ Geographical Heatmap',
      description: 'Geographical distribution by state'
    },
    {
      id: 'correlation-matrix',
      label: '🔗 Correlation Matrix',
      description: 'Parameter correlations'
    },
    {
      id: 'statistical-summary',
      label: '📋 Statistical Summary',
      description: 'Statistical distributions'
    }
  ]

  if (!isOpen) return null

  return (
    <div className="visualization-panel" style={{
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
      <div className="visualization-modal" style={{
        background: 'var(--color-surface)',
        borderRadius: 20,
        boxShadow: 'var(--shadow-xl)',
        width: '100%',
        maxWidth: '1400px',
        height: '90vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div className="visualization-header" style={{
          padding: '1.5rem 2rem',
          borderBottom: '1px solid var(--color-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <h2 style={{ margin: 0, color: 'var(--color-text-primary)' }}>
            📊 Data Visualizations
          </h2>
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

        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Sidebar with buttons */}
          <div className="visualization-sidebar" style={{
            width: '300px',
            background: 'var(--gradient-surface)',
            padding: '1.5rem',
            overflowY: 'auto',
            borderRight: '1px solid var(--color-border)'
          }}>
            <div style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ margin: '0 0 1rem 0', color: 'var(--color-text-primary)' }}>
                Visualization Types
              </h3>
              
              {visualizationButtons.map((button) => (
                <button
                  key={button.id}
                  onClick={() => handleVisualizationClick(button.id)}
                  style={{
                    width: '100%',
                    padding: '1rem',
                    marginBottom: '0.75rem',
                    background: activeVisualization === button.id 
                      ? 'var(--gradient-primary)' 
                      : 'var(--color-surface)',
                    color: activeVisualization === button.id 
                      ? 'white' 
                      : 'var(--color-text-primary)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 12,
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all 0.2s ease',
                    fontSize: '0.9rem',
                    fontWeight: '500'
                  }}
                  onMouseOver={(e) => {
                    if (activeVisualization !== button.id) {
                      e.target.style.background = 'var(--color-surface-elevated)'
                      e.target.style.transform = 'translateY(-1px)'
                    }
                  }}
                  onMouseOut={(e) => {
                    if (activeVisualization !== button.id) {
                      e.target.style.background = 'var(--color-surface)'
                      e.target.style.transform = 'translateY(0)'
                    }
                  }}
                >
                  <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>
                    {button.label}
                  </div>
                  <div style={{ 
                    fontSize: '0.8rem', 
                    opacity: 0.8,
                    lineHeight: 1.3
                  }}>
                    {button.description}
                  </div>
                </button>
              ))}
            </div>

            {/* State selector for state analysis */}
            {activeVisualization === 'state-analysis' && availableStates.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  Select State:
                </label>
                <select
                  value={selectedState}
                  onChange={(e) => setSelectedState(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid var(--color-border)',
                    borderRadius: 8,
                    background: 'var(--color-surface)',
                    color: 'var(--color-text-primary)',
                    fontSize: '0.9rem'
                  }}
                >
                  <option value="">All States</option>
                  {availableStates.map((state) => (
                    <option key={state} value={state}>
                      {state}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Metric selector for geographical heatmap */}
            {activeVisualization === 'geographical-heatmap' && availableMetrics.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  Select Metric:
                </label>
                <select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid var(--color-border)',
                    borderRadius: 8,
                    background: 'var(--color-surface)',
                    color: 'var(--color-text-primary)',
                    fontSize: '0.9rem'
                  }}
                >
                  {availableMetrics.map((metric) => (
                    <option key={metric} value={metric}>
                      {metric}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Main content area */}
          <div className="visualization-content" style={{
            flex: 1,
            padding: '1.5rem',
            overflowY: 'auto',
            background: 'var(--color-background)'
          }}>
            {activeVisualization ? (
              <div>
                <h3 style={{ 
                  margin: '0 0 1rem 0', 
                  color: 'var(--color-text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  {getVisualizationTitle(activeVisualization)}
                  {loading && (
                    <div style={{
                      width: 20,
                      height: 20,
                      border: '2px solid var(--color-border)',
                      borderTop: '2px solid var(--color-primary)',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }} />
                  )}
                </h3>
                
                <PlotlyChart
                  plotData={plotData}
                  loading={loading}
                  error={error}
                  title={getVisualizationTitle(activeVisualization)}
                  height={600}
                />
              </div>
            ) : (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'var(--color-text-secondary)',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>📊</div>
                <h3 style={{ margin: '0 0 0.5rem 0' }}>Select a Visualization</h3>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>
                  Choose a visualization type from the sidebar to explore groundwater data
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default VisualizationPanel
