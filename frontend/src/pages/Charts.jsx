import React, { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import './Charts.css'

const Charts = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedState, setSelectedState] = useState('')
  const [availableStates, setAvailableStates] = useState([])
  const [visualizations, setVisualizations] = useState({
    overview: null,
    stateAnalysis: null,
    geographicalHeatmap: null,
    temporalAnalysis: null,
    correlationMatrix: null
  })

  // Load available states on component mount
  useEffect(() => {
    loadAvailableStates()
  }, [])

  const loadAvailableStates = async () => {
    try {
      const response = await fetch('http://localhost:8000/visualizations/available-states')
      const data = await response.json()
      if (data.success) {
        setAvailableStates(data.states.filter(state => state && state.trim() !== ''))
      }
    } catch (err) {
      console.error('Error loading states:', err)
    }
  }

  const loadVisualization = async (type, state = null) => {
    setLoading(true)
    setError(null)
    
    try {
      let url = `http://localhost:8000/visualizations/${type}`
      if (state) {
        url += `?state=${encodeURIComponent(state)}`
      }
      
      const response = await fetch(url)
      const data = await response.json()
      
      if (data.success && data.plot_json) {
        const plotData = JSON.parse(data.plot_json)
        console.log(`Loaded ${type} visualization:`, plotData)
        setVisualizations(prev => ({
          ...prev,
          [type]: plotData
        }))
      } else {
        throw new Error(data.detail || 'Failed to load visualization')
      }
    } catch (err) {
      console.error(`Error loading ${type}:`, err)
      setError(`Failed to load ${type}: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const loadAllVisualizations = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const promises = [
        loadVisualization('overview'),
        loadVisualization('geographical-heatmap'),
        loadVisualization('temporal-analysis'),
        loadVisualization('correlation-matrix')
      ]
      
      await Promise.all(promises)
    } catch (err) {
      console.error('Error loading visualizations:', err)
      setError('Failed to load some visualizations')
    } finally {
      setLoading(false)
    }
  }

  const loadStateVisualization = async () => {
    if (!selectedState) return
    await loadVisualization('state-analysis', selectedState)
  }

  const plotConfig = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    responsive: true
  }

  const plotLayout = {
    autosize: true,
    margin: { l: 50, r: 50, t: 50, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, system-ui, sans-serif',
      size: 12,
      color: '#FFFFFF'
    }
  }

  return (
    <div className="charts-container">
      <div className="charts-header">
        <h1>📊 Groundwater Data Visualizations</h1>
        <p>Interactive charts and graphs for comprehensive groundwater analysis across all states</p>
      </div>

      {/* Controls */}
      <div className="charts-controls">
        <div className="control-group">
          <button 
            className="btn-primary"
            onClick={loadAllVisualizations}
            disabled={loading}
          >
            {loading ? '⏳ Loading...' : '📊 Load All Visualizations'}
          </button>
        </div>
        
        <div className="control-group">
          <select 
            value={selectedState} 
            onChange={(e) => setSelectedState(e.target.value)}
            className="state-selector"
            style={{
              backgroundColor: '#ffffff',
              color: '#333333',
              border: '2px solid #667eea',
              borderRadius: '8px',
              padding: '12px 16px',
              fontSize: '16px',
              fontWeight: '500',
              minWidth: '300px',
              cursor: 'pointer'
            }}
          >
            <option value="" style={{ color: '#666666' }}>Select a State for Analysis</option>
            {availableStates.map(state => (
              <option key={state} value={state} style={{ color: '#333333', backgroundColor: '#ffffff' }}>
                {state}
              </option>
            ))}
          </select>
          <button 
            className="btn-secondary"
            onClick={loadStateVisualization}
            disabled={!selectedState || loading}
          >
            🔍 Analyze State
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Debug Info */}
      {process.env.NODE_ENV === 'development' && (
        <div className="debug-info" style={{ 
          background: 'rgba(0, 0, 0, 0.8)', 
          color: '#ffffff', 
          padding: '15px', 
          margin: '15px 0', 
          borderRadius: '8px',
          border: '2px solid #00ff00',
          fontSize: '14px',
          fontFamily: 'monospace'
        }}>
          <h3 style={{ color: '#00ff00', margin: '0 0 10px 0', fontSize: '16px' }}>🔍 Debug Info:</h3>
          <p style={{ margin: '5px 0', color: '#ffffff' }}><strong>Available visualizations:</strong> {Object.keys(visualizations).join(', ') || 'None'}</p>
          <p style={{ margin: '5px 0', color: '#ffffff' }}><strong>Selected state:</strong> {selectedState || 'None'}</p>
          <p style={{ margin: '5px 0', color: '#ffffff' }}><strong>Loading:</strong> {loading ? 'Yes' : 'No'}</p>
          <p style={{ margin: '5px 0', color: '#ffffff' }}><strong>State Analysis Data:</strong> {visualizations['state-analysis'] ? 'Available' : 'Not Available'}</p>
        </div>
      )}

      {/* Test Chart */}
      <div className="chart-section">
        <h2>🧪 Test Chart (Plotly Working?)</h2>
        <div className="chart-container">
          <Plot
            data={[{
              x: ['A', 'B', 'C', 'D'],
              y: [1, 2, 3, 4],
              type: 'bar',
              marker: { color: 'lightblue' }
            }]}
            layout={{
              title: 'Test Chart - If you see this, Plotly is working!',
              ...plotLayout
            }}
            config={plotConfig}
            style={{ width: '100%', height: '300px' }}
            useResizeHandler={true}
            onError={(err) => console.error('Test chart Plotly error:', err)}
          />
        </div>
      </div>

      {/* Overview Dashboard */}
      {visualizations.overview && (
        <div className="chart-section overview-chart">
          <h2>📈 National Overview Dashboard</h2>
          <div className="chart-container">
            <Plot
              data={visualizations.overview.data || []}
              layout={{
                ...plotLayout,
                ...visualizations.overview.layout,
                title: 'Groundwater Overview Dashboard',
                height: 600,
                margin: { l: 60, r: 60, t: 80, b: 60 }
              }}
              config={plotConfig}
              style={{ width: '100%', height: '600px' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}

      {/* State Analysis */}
      {visualizations['state-analysis'] && (
        <div className="chart-section state-analysis-chart">
          <h2>🗺️ State Analysis - {selectedState}</h2>
          <div className="chart-container">
            <Plot
              data={visualizations['state-analysis'].data || []}
              layout={{
                ...plotLayout,
                ...visualizations['state-analysis'].layout,
                title: {
                  text: `Groundwater Analysis - ${selectedState}`,
                  font: { size: 20, color: '#ffffff' }
                },
                height: 800,
                margin: { l: 80, r: 80, t: 100, b: 80 },
                font: { size: 14, color: '#ffffff' },
                xaxis: { 
                  ...visualizations['state-analysis'].layout?.xaxis,
                  tickangle: -45,
                  tickfont: { size: 12, color: '#ffffff' }
                },
                yaxis: { 
                  ...visualizations['state-analysis'].layout?.yaxis,
                  tickfont: { size: 12, color: '#ffffff' }
                }
              }}
              config={{
                ...plotConfig,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                displaylogo: false
              }}
              style={{ width: '100%', height: '800px' }}
              useResizeHandler={true}
              onError={(err) => console.error('Plotly error:', err)}
            />
          </div>
        </div>
      )}

      {/* Geographical Heatmap */}
      {visualizations.geographicalHeatmap && (
        <div className="chart-section geographical-chart">
          <h2>🌍 Geographical Distribution</h2>
          <div className="chart-container">
            <Plot
              data={visualizations.geographicalHeatmap.data || []}
              layout={{
                ...plotLayout,
                ...visualizations.geographicalHeatmap.layout,
                title: 'Geographical Distribution of Groundwater Metrics',
                height: 500,
                margin: { l: 60, r: 60, t: 80, b: 60 }
              }}
              config={plotConfig}
              style={{ width: '100%', height: '500px' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}

      {/* Temporal Analysis */}
      {visualizations.temporalAnalysis && (
        <div className="chart-section temporal-chart">
          <h2>📅 Temporal Trends</h2>
          <div className="chart-container">
            <Plot
              data={visualizations.temporalAnalysis.data || []}
              layout={{
                ...plotLayout,
                ...visualizations.temporalAnalysis.layout,
                title: 'Groundwater Trends Over Time',
                height: 500,
                margin: { l: 60, r: 60, t: 80, b: 60 }
              }}
              config={plotConfig}
              style={{ width: '100%', height: '500px' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}

      {/* Correlation Matrix */}
      {visualizations.correlationMatrix && (
        <div className="chart-section correlation-chart">
          <h2>🔗 Parameter Correlations</h2>
          <div className="chart-container">
            <Plot
              data={visualizations.correlationMatrix.data || []}
              layout={{
                ...plotLayout,
                ...visualizations.correlationMatrix.layout,
                title: 'Groundwater Parameter Correlations',
                height: 500,
                margin: { l: 60, r: 60, t: 80, b: 60 }
              }}
              config={plotConfig}
              style={{ width: '100%', height: '500px' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}

      {/* State Grid */}
      {availableStates.length > 0 && (
        <div className="chart-section">
          <h2>🏛️ Available States for Analysis</h2>
          <div className="states-grid">
            {availableStates.map(state => (
              <div 
                key={state} 
                className={`state-card ${selectedState === state ? 'selected' : ''}`}
                onClick={() => setSelectedState(state)}
              >
                <h3>{state}</h3>
                <p>Click to analyze</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Charts
