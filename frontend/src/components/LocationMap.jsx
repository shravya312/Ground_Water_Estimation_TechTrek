import { useEffect, useRef, useState } from 'react'
import { loadGoogleMaps, isGoogleMapsReady } from '../utils/googleMapsLoader'
import { analyzeLocation } from '../services/locationAnalysisService'

const LocationMap = ({ location, onLocationChange, isGettingLocation }) => {
  const mapRef = useRef(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [mapError, setMapError] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [clickedLocation, setClickedLocation] = useState(null)
  const [hoveredState, setHoveredState] = useState(null)
  const [statePolygons, setStatePolygons] = useState([])

  // Initialize Google Map
  useEffect(() => {
    const initializeGoogleMaps = async () => {
      try {
        console.log('Loading Google Maps...')
        await loadGoogleMaps()
        console.log('Google Maps loaded successfully')
        setMapLoaded(true)
        initializeMap()
      } catch (error) {
        console.error('Failed to load Google Maps:', error)
        // Try direct loading as fallback
        console.log('Trying direct Google Maps loading...')
        const script = document.createElement('script')
        script.src = 'https://maps.googleapis.com/maps/api/js?key=AIzaSyBuSA6XXz7ZmSFonptlXs1ALyNTZfLrf8g&libraries=places'
        script.async = true
        script.defer = true
        script.onload = () => {
          console.log('Google Maps loaded via fallback')
          setMapLoaded(true)
          initializeMap()
        }
        script.onerror = (scriptError) => {
          console.error('Script loading error:', scriptError)
          setMapError(`Failed to load Google Maps script: ${scriptError.message || 'Unknown error'}`)
        }
        document.head.appendChild(script)
      }
    }

    if (isGoogleMapsReady()) {
      console.log('Google Maps already ready')
      setMapLoaded(true)
      initializeMap()
    } else {
      console.log('Initializing Google Maps...')
      initializeGoogleMaps()
    }

    // Add timeout to prevent hanging
    const timeout = setTimeout(() => {
      if (!mapLoaded && !mapError) {
        console.error('Google Maps loading timeout')
        setMapError('Google Maps loading timeout. Please check your internet connection and API key.')
      }
    }, 10000) // 10 second timeout

    return () => clearTimeout(timeout)
  }, [mapLoaded, mapError])

  // Add state boundaries with hover highlighting for specific states only
  const addStateBoundaries = (map) => {
    // Only highlight specific states with groundwater data
    const highlightedStates = [
      {
        name: 'Maharashtra',
        coords: [
          { lat: 19.7515, lng: 72.3954 },
          { lat: 19.7515, lng: 80.8889 },
          { lat: 15.6021, lng: 80.8889 },
          { lat: 15.6021, lng: 72.3954 },
          { lat: 19.7515, lng: 72.3954 }
        ],
        center: { lat: 19.7515, lng: 76.6421 },
        highlight: true
      },
      {
        name: 'Karnataka',
        coords: [
          { lat: 18.1124, lng: 74.1240 },
          { lat: 18.1124, lng: 78.5704 },
          { lat: 11.6705, lng: 78.5704 },
          { lat: 11.6705, lng: 74.1240 },
          { lat: 18.1124, lng: 74.1240 }
        ],
        center: { lat: 15.3173, lng: 75.7139 },
        highlight: true
      },
      {
        name: 'Tamil Nadu',
        coords: [
          { lat: 13.0839, lng: 76.5704 },
          { lat: 13.0839, lng: 80.8889 },
          { lat: 8.0883, lng: 80.8889 },
          { lat: 8.0883, lng: 76.5704 },
          { lat: 13.0839, lng: 76.5704 }
        ],
        center: { lat: 11.1271, lng: 78.6569 },
        highlight: true
      },
      {
        name: 'Gujarat',
        coords: [
          { lat: 24.7136, lng: 68.1801 },
          { lat: 24.7136, lng: 74.1240 },
          { lat: 20.4283, lng: 74.1240 },
          { lat: 20.4283, lng: 68.1801 },
          { lat: 24.7136, lng: 68.1801 }
        ],
        center: { lat: 23.0225, lng: 72.5714 },
        highlight: true
      },
      {
        name: 'Rajasthan',
        coords: [
          { lat: 30.0479, lng: 69.1960 },
          { lat: 30.0479, lng: 78.5704 },
          { lat: 23.0225, lng: 78.5704 },
          { lat: 23.0225, lng: 69.1960 },
          { lat: 30.0479, lng: 69.1960 }
        ],
        center: { lat: 27.0238, lng: 74.2179 },
        highlight: true
      },
      {
        name: 'Uttar Pradesh',
        coords: [
          { lat: 30.0479, lng: 77.5704 },
          { lat: 30.0479, lng: 84.1240 },
          { lat: 23.0225, lng: 84.1240 },
          { lat: 23.0225, lng: 77.5704 },
          { lat: 30.0479, lng: 77.5704 }
        ],
        center: { lat: 26.8467, lng: 80.9462 },
        highlight: false
      },
      {
        name: 'Madhya Pradesh',
        coords: [
          { lat: 26.8467, lng: 74.1240 },
          { lat: 26.8467, lng: 82.1240 },
          { lat: 21.1240, lng: 82.1240 },
          { lat: 21.1240, lng: 74.1240 },
          { lat: 26.8467, lng: 74.1240 }
        ],
        center: { lat: 22.9734, lng: 78.6569 },
        highlight: false
      },
      {
        name: 'West Bengal',
        coords: [
          { lat: 27.1240, lng: 85.1240 },
          { lat: 27.1240, lng: 89.1240 },
          { lat: 21.1240, lng: 89.1240 },
          { lat: 21.1240, lng: 85.1240 },
          { lat: 27.1240, lng: 85.1240 }
        ],
        center: { lat: 22.9868, lng: 87.6850 },
        highlight: false
      },
      {
        name: 'Andhra Pradesh',
        coords: [
          { lat: 19.1240, lng: 76.5704 },
          { lat: 19.1240, lng: 84.1240 },
          { lat: 12.1240, lng: 84.1240 },
          { lat: 12.1240, lng: 76.5704 },
          { lat: 19.1240, lng: 76.5704 }
        ],
        center: { lat: 15.9129, lng: 79.7400 },
        highlight: true
      },
      {
        name: 'Kerala',
        coords: [
          { lat: 12.1240, lng: 74.1240 },
          { lat: 12.1240, lng: 77.5704 },
          { lat: 8.1240, lng: 77.5704 },
          { lat: 8.1240, lng: 74.1240 },
          { lat: 12.1240, lng: 74.1240 }
        ],
        center: { lat: 10.8505, lng: 76.2711 },
        highlight: true
      }
    ]

    const polygons = highlightedStates.map(state => {
      const polygon = new window.google.maps.Polygon({
        paths: state.coords,
        strokeColor: state.highlight ? '#4285F4' : '#CCCCCC',
        strokeOpacity: state.highlight ? 0.8 : 0.3,
        strokeWeight: state.highlight ? 2 : 1,
        fillColor: 'transparent',
        fillOpacity: 0,
        map: map,
        title: state.name
      })

      // Add hover effects only for highlighted states
      if (state.highlight) {
        polygon.addListener('mouseover', () => {
          polygon.setOptions({
            fillColor: 'transparent',
            fillOpacity: 0,
            strokeColor: '#FF6B6B',
            strokeWeight: 4,
            strokeOpacity: 1
          })
          setHoveredState(state.name)
        })

        polygon.addListener('mouseout', () => {
          polygon.setOptions({
            fillColor: 'transparent',
            fillOpacity: 0,
            strokeColor: '#4285F4',
            strokeWeight: 2,
            strokeOpacity: 0.8
          })
          setHoveredState(null)
        })
      }

      // Add click listener for state analysis
      polygon.addListener('click', async (event) => {
        const lat = event.latLng.lat()
        const lng = event.latLng.lng()
        
        setClickedLocation({ lat, lng })
        setIsAnalyzing(true)
        setAnalysisResult(null)

        try {
          const result = await analyzeLocation(lat, lng)
          setAnalysisResult(result)
          
          if (onLocationChange) {
            onLocationChange({ lat, lng }, result)
          }
        } catch (error) {
          console.error('Error analyzing location:', error)
          setAnalysisResult({ error: error.message })
          
          if (onLocationChange) {
            onLocationChange({ lat, lng }, { error: error.message })
          }
        } finally {
          setIsAnalyzing(false)
        }
      })

      return polygon
    })

    setStatePolygons(polygons)
  }

  const initializeMap = () => {
    if (mapRef.current && window.google && window.google.maps) {
      console.log('Initializing map with Google Maps API')
      const map = new window.google.maps.Map(mapRef.current, {
        center: { lat: 20.5937, lng: 78.9629 }, // Center of India
        zoom: 6,
        mapTypeId: 'hybrid', // Better for seeing terrain
        mapTypeControl: true,
        streetViewControl: false,
        fullscreenControl: true,
        zoomControl: true
      })

      // Add click listener for location analysis
      map.addListener('click', async (event) => {
        const lat = event.latLng.lat()
        const lng = event.latLng.lng()
        
        setClickedLocation({ lat, lng })
        setIsAnalyzing(true)
        setAnalysisResult(null)

        try {
          const result = await analyzeLocation(lat, lng)
          setAnalysisResult(result)
          
          // Pass analysis result to parent component
          if (onLocationChange) {
            onLocationChange({ lat, lng }, result)
          }
          
          // Add marker for clicked location
          new window.google.maps.Marker({
            position: { lat, lng },
            map: map,
            title: `Analysis for ${result.state}`,
            icon: {
              url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="10" fill="#FF6B6B" stroke="#FFFFFF" stroke-width="2"/>
                  <circle cx="12" cy="12" r="4" fill="#FFFFFF"/>
                </svg>
              `),
              scaledSize: new window.google.maps.Size(24, 24)
            }
          })
        } catch (error) {
          console.error('Error analyzing location:', error)
          setAnalysisResult({ error: error.message })
          
          // Pass error to parent component
          if (onLocationChange) {
            onLocationChange({ lat, lng }, { error: error.message })
          }
        } finally {
          setIsAnalyzing(false)
        }
      })

      // Add a marker for current location if available
      if (location) {
        new window.google.maps.Marker({
          position: location,
          map: map,
          title: 'Your Current Location',
          icon: {
            url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" fill="#4CAF50" stroke="#FFFFFF" stroke-width="2"/>
                <circle cx="12" cy="12" r="4" fill="#FFFFFF"/>
              </svg>
            `),
            scaledSize: new window.google.maps.Size(24, 24)
          }
        })
      }

      // Add state boundaries with hover effects
      addStateBoundaries(map)
    }
  }

  // Update map when location changes
  useEffect(() => {
    if (mapLoaded && location) {
      initializeMap()
    }
  }, [location, mapLoaded])

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const newLocation = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
          }
          onLocationChange(newLocation)
        },
        (error) => {
          console.error('Error getting location:', error)
          alert('Unable to get your location. Please check your browser permissions.')
        }
      )
    } else {
      alert('Geolocation is not supported by this browser.')
    }
  }

  return (
    <>
      <style>
        {`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}
      </style>
      <div style={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-md)'
      }}>
      {/* Map Container */}
      <div style={{ flex: 1, position: 'relative', minHeight: '300px' }}>
        {mapError ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--color-error)',
            fontSize: '0.9rem',
            textAlign: 'center',
            padding: '2rem'
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚠️</div>
            <div style={{ marginBottom: '0.5rem', fontWeight: '600' }}>Map Loading Error</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
              {mapError}
            </div>
            <div style={{ 
              fontSize: '0.75rem', 
              color: 'var(--color-text-muted)', 
              marginTop: '1rem',
              padding: '0.5rem',
              background: 'var(--color-surface-elevated)',
              borderRadius: '4px'
            }}>
              Please check the GOOGLE_MAPS_SETUP.md file for configuration instructions.
            </div>
          </div>
        ) : mapLoaded ? (
          <div 
            ref={mapRef}
            style={{
              width: '100%',
              height: '100%'
            }}
          />
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--color-text-secondary)',
            fontSize: '0.9rem'
          }}>
            Loading map...
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{
        padding: '1rem',
        borderTop: '1px solid var(--color-border)',
        background: 'var(--color-surface-elevated)',
        display: 'flex',
        justifyContent: 'center'
      }}>
        <button 
          onClick={getCurrentLocation}
          disabled={isGettingLocation}
          style={{
            padding: '0.75rem 1.5rem',
            background: isGettingLocation ? 'var(--color-text-muted)' : 'var(--color-primary)',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: isGettingLocation ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
            fontWeight: '600',
            boxShadow: 'var(--shadow-sm)',
            transition: 'all 0.2s ease',
            opacity: isGettingLocation ? 0.7 : 1
          }}
          onMouseOver={(e) => {
            if (!isGettingLocation) {
              e.target.style.background = 'var(--color-primary-dark)'
              e.target.style.transform = 'translateY(-1px)'
            }
          }}
          onMouseOut={(e) => {
            if (!isGettingLocation) {
              e.target.style.background = 'var(--color-primary)'
              e.target.style.transform = 'translateY(0)'
            }
          }}
        >
          {isGettingLocation ? '🔄 Getting Location...' : '📍 Get Current Location'}
        </button>
      </div>

      {/* Selected Location Display */}
      {clickedLocation && (
        <div style={{
          padding: '1rem',
          background: 'rgba(59, 130, 246, 0.1)',
          borderTop: '1px solid rgba(59, 130, 246, 0.2)',
          fontSize: '0.9rem',
          color: 'var(--color-primary)',
          textAlign: 'center'
        }}>
          <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>
            📍 Selected Location
          </div>
          <div style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            <strong>Latitude:</strong> {clickedLocation.lat.toFixed(6)}<br />
            <strong>Longitude:</strong> {clickedLocation.lng.toFixed(6)}
          </div>
        </div>
      )}

      {/* Hover State Display */}
      {hoveredState && (
        <div style={{
          padding: '0.75rem 1rem',
          background: 'linear-gradient(135deg, #FF6B6B, #FF8E8E)',
          borderTop: '1px solid #FF6B6B',
          color: 'white',
          textAlign: 'center',
          fontSize: '0.9rem',
          fontWeight: '600',
          animation: 'fadeIn 0.3s ease-in-out'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
            <span>📍</span>
            <span>Hovering over: {hoveredState}</span>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div style={{
        padding: '1rem',
        borderTop: '1px solid var(--color-border)',
        background: 'var(--color-surface)',
        textAlign: 'center',
        fontSize: '0.8rem',
        color: 'var(--color-text-secondary)'
      }}>
        <div style={{ marginBottom: '0.5rem', fontWeight: '600' }}>
          🗺️ Interactive India Map
        </div>
        <div>
          Hover over highlighted states (blue borders) to see details, click to analyze groundwater data
        </div>
        <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
          Only states with groundwater data are highlighted
        </div>
      </div>

      {/* Analysis Results */}
      {analysisResult && (
        <div style={{
          padding: '1rem',
          borderTop: '1px solid var(--color-border)',
          background: 'var(--color-surface)',
          maxHeight: '400px',
          overflowY: 'auto',
          scrollbarWidth: 'thin',
          scrollbarColor: 'var(--color-primary) var(--color-surface-elevated)'
        }}>
          {analysisResult.error ? (
            <div style={{
              color: 'var(--color-error)',
              fontSize: '0.9rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>❌</div>
              <div>Error: {analysisResult.error}</div>
            </div>
          ) : (
            <div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '1rem',
                fontSize: '1.1rem',
                fontWeight: '600',
                color: 'var(--color-text-primary)'
              }}>
                <span>📍</span>
                <span>Analysis for {analysisResult.state}</span>
              </div>
              
              <div style={{
                fontSize: '0.8rem',
                color: 'var(--color-text-secondary)',
                marginBottom: '1rem',
                display: 'flex',
                gap: '1rem',
                flexWrap: 'wrap'
              }}>
                <span>📊 {analysisResult.data_points} data points</span>
                <span>🏛️ {analysisResult.summary?.districts_covered || 0} districts</span>
                <span>📅 {analysisResult.summary?.years_covered?.length || 0} years</span>
              </div>

              <div style={{
                background: 'var(--color-surface-elevated)',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid var(--color-border)',
                fontSize: '0.9rem',
                lineHeight: '1.5',
                color: 'var(--color-text-primary)',
                maxHeight: '250px',
                overflowY: 'auto',
                scrollbarWidth: 'thin',
                scrollbarColor: 'var(--color-primary) var(--color-surface)'
              }}>
                <div style={{ 
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'monospace'
                }}>
                  {analysisResult.analysis}
                </div>
              </div>

              {clickedLocation && (
                <div style={{
                  marginTop: '0.5rem',
                  fontSize: '0.75rem',
                  color: 'var(--color-text-muted)',
                  textAlign: 'center'
                }}>
                  Coordinates: {clickedLocation.lat.toFixed(4)}, {clickedLocation.lng.toFixed(4)}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Loading State for Analysis */}
      {isAnalyzing && (
        <div style={{
          padding: '1rem',
          borderTop: '1px solid var(--color-border)',
          background: 'var(--color-surface-elevated)',
          textAlign: 'center'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            color: 'var(--color-primary)',
            fontSize: '0.9rem'
          }}>
            <div style={{
              width: '16px',
              height: '16px',
              border: '2px solid var(--color-primary)',
              borderTop: '2px solid transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
            Analyzing groundwater data for this location...
          </div>
        </div>
      )}
      </div>
    </>
  )
}

export default LocationMap
