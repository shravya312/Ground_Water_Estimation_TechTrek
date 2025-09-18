// Visualization service for calling the main.py API endpoints
const API_BASE = import.meta.env?.VITE_API_URL || 'http://localhost:8000'

export const visualizationService = {
  // Get overview dashboard
  async getOverviewDashboard() {
    try {
      const response = await fetch(`${API_BASE}/visualizations/overview`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching overview dashboard:', error)
      throw error
    }
  },

  // Get state analysis
  async getStateAnalysis(state = null) {
    try {
      const url = state 
        ? `${API_BASE}/visualizations/state-analysis?state=${encodeURIComponent(state)}`
        : `${API_BASE}/visualizations/state-analysis`
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching state analysis:', error)
      throw error
    }
  },

  // Get temporal analysis
  async getTemporalAnalysis() {
    try {
      const response = await fetch(`${API_BASE}/visualizations/temporal-analysis`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching temporal analysis:', error)
      throw error
    }
  },

  // Get geographical heatmap
  async getGeographicalHeatmap(metric = 'Annual Ground water Recharge (ham) - Total - Total') {
    try {
      const url = `${API_BASE}/visualizations/geographical-heatmap?metric=${encodeURIComponent(metric)}`
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching geographical heatmap:', error)
      throw error
    }
  },

  // Get correlation matrix
  async getCorrelationMatrix() {
    try {
      const response = await fetch(`${API_BASE}/visualizations/correlation-matrix`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching correlation matrix:', error)
      throw error
    }
  },

  // Get statistical summary
  async getStatisticalSummary() {
    try {
      const response = await fetch(`${API_BASE}/visualizations/statistical-summary`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching statistical summary:', error)
      throw error
    }
  },

  // Get available states
  async getAvailableStates() {
    try {
      const response = await fetch(`${API_BASE}/visualizations/available-states`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching available states:', error)
      throw error
    }
  },

  // Get available metrics
  async getAvailableMetrics() {
    try {
      const response = await fetch(`${API_BASE}/visualizations/available-metrics`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching available metrics:', error)
      throw error
    }
  }
}

export default visualizationService
