const API_BASE_URL = import.meta.env?.VITE_API_URL || ''

export const ingresService = {
  // Query groundwater data with structured response
  async queryGroundwater(query, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/ingres/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          state: options.state,
          district: options.district,
          assessment_unit: options.assessment_unit,
          include_visualizations: options.include_visualizations !== false,
          language: options.language || 'en'
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error querying groundwater data:', error)
      throw error
    }
  },

  // Analyze location-based groundwater data
  async analyzeLocation(lat, lng, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/ingres/location-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat,
          lng,
          include_visualizations: options.include_visualizations !== false,
          language: options.language || 'en'
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error analyzing location:', error)
      throw error
    }
  },

  // Get national criticality summary
  async getCriticalitySummary() {
    try {
      const response = await fetch(`${API_BASE_URL}/ingres/criticality-summary`)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error getting criticality summary:', error)
      throw error
    }
  },

  // Get available states
  async getStates() {
    try {
      const response = await fetch(`${API_BASE_URL}/ingres/states`)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error getting states:', error)
      throw error
    }
  },

  // Get districts for a state
  async getDistricts(state) {
    try {
      const response = await fetch(`${API_BASE_URL}/ingres/districts/${encodeURIComponent(state)}`)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error getting districts:', error)
      throw error
    }
  }
}

export default ingresService
