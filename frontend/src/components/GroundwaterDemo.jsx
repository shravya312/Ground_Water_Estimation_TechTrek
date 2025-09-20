import React, { useState } from 'react'
import ingresService from '../services/ingresService'
import GroundwaterAnalysisCard from './GroundwaterAnalysisCard'
import VisualizationModal from './VisualizationModal'

const GroundwaterDemo = () => {
  const [demoData, setDemoData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showVisualizationModal, setShowVisualizationModal] = useState(false)
  const [selectedVisualization, setSelectedVisualization] = useState(null)

  const demoQueries = [
    {
      title: '🏛️ State Analysis',
      query: 'groundwater analysis for chhattisgarh',
      description: 'Comprehensive groundwater analysis for Chhattisgarh state'
    },
    {
      title: '📍 Location Analysis',
      query: 'analyze groundwater at coordinates 28.7041, 77.1025',
      description: 'Location-based groundwater analysis for Delhi coordinates'
    },
    {
      title: '📊 National Overview',
      query: 'national groundwater criticality summary',
      description: 'Country-wide groundwater status and criticality distribution'
    },
    {
      title: '💧 Quality Analysis',
      query: 'water quality issues in rajasthan',
      description: 'Water quality analysis and contamination issues'
    },
    {
      title: '📈 Resource Analysis',
      query: 'groundwater recharge patterns in maharashtra',
      description: 'Groundwater recharge analysis and resource availability'
    }
  ]

  const handleDemoQuery = async (query) => {
    setLoading(true)
    setError(null)
    setDemoData(null)

    try {
      let response
      
      if (query.includes('coordinates') || query.includes('28.7041')) {
        // Location-based query
        response = await ingresService.analyzeLocation(28.7041, 77.1025, {
          include_visualizations: true
        })
      } else if (query.includes('national') || query.includes('summary')) {
        // National summary query
        response = await ingresService.getCriticalitySummary()
      } else {
        // Regular groundwater query
        response = await ingresService.queryGroundwater(query, {
          include_visualizations: true
        })
      }

      setDemoData(response)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleVisualizationClick = (visualization) => {
    setSelectedVisualization(visualization)
    setShowVisualizationModal(true)
  }

  return (
    <div style={{
      background: 'var(--color-background)',
      minHeight: '100vh',
      padding: '2rem'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{
          textAlign: 'center',
          marginBottom: '3rem'
        }}>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: '800',
            color: 'var(--color-text-primary)',
            marginBottom: '1rem',
            background: 'var(--gradient-primary)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            💧 INGRES Groundwater ChatBOT Demo
          </h1>
          <p style={{
            fontSize: '1.1rem',
            color: 'var(--color-text-secondary)',
            maxWidth: '600px',
            margin: '0 auto',
            lineHeight: '1.6'
          }}>
            Experience the power of AI-driven groundwater analysis with interactive visualizations, 
            color-coded criticality levels, and comprehensive quality assessments.
          </p>
        </div>

        {/* Demo Queries */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem',
          marginBottom: '3rem'
        }}>
          {demoQueries.map((demo, index) => (
            <div
              key={index}
              onClick={() => handleDemoQuery(demo.query)}
              style={{
                background: 'var(--color-surface)',
                padding: '1.5rem',
                borderRadius: '16px',
                border: '1px solid var(--color-border)',
                boxShadow: 'var(--shadow-lg)',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseOver={(e) => {
                e.target.style.transform = 'translateY(-4px)'
                e.target.style.boxShadow = 'var(--shadow-xl)'
                e.target.style.borderColor = 'var(--color-primary)'
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)'
                e.target.style.boxShadow = 'var(--shadow-lg)'
                e.target.style.borderColor = 'var(--color-border)'
              }}
            >
              <div style={{
                fontSize: '2rem',
                marginBottom: '1rem'
              }}>
                {demo.title.split(' ')[0]}
              </div>
              <h3 style={{
                margin: '0 0 0.5rem 0',
                fontSize: '1.1rem',
                fontWeight: '600',
                color: 'var(--color-text-primary)'
              }}>
                {demo.title}
              </h3>
              <p style={{
                margin: 0,
                fontSize: '0.9rem',
                color: 'var(--color-text-secondary)',
                lineHeight: '1.5'
              }}>
                {demo.description}
              </p>
              <div style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                fontSize: '1.5rem',
                opacity: 0.3
              }}>
                →
              </div>
            </div>
          ))}
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{
            textAlign: 'center',
            padding: '3rem',
            background: 'var(--color-surface)',
            borderRadius: '16px',
            border: '1px solid var(--color-border)',
            marginBottom: '2rem'
          }}>
            <div style={{
              width: '50px',
              height: '50px',
              border: '4px solid var(--color-border)',
              borderTop: '4px solid var(--color-primary)',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 1rem auto'
            }} />
            <h3 style={{
              margin: '0 0 0.5rem 0',
              color: 'var(--color-text-primary)'
            }}>
              Analyzing Groundwater Data...
            </h3>
            <p style={{
              margin: 0,
              color: 'var(--color-text-secondary)'
            }}>
              Processing your request with AI-powered analysis
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid #ef4444',
            borderRadius: '12px',
            padding: '1.5rem',
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            <div style={{
              fontSize: '2rem',
              marginBottom: '0.5rem'
            }}>
              ❌
            </div>
            <h3 style={{
              margin: '0 0 0.5rem 0',
              color: '#ef4444'
            }}>
              Analysis Failed
            </h3>
            <p style={{
              margin: 0,
              color: 'var(--color-text-secondary)'
            }}>
              {error}
            </p>
          </div>
        )}

        {/* Results */}
        {demoData && !loading && (
          <div>
            <h2 style={{
              fontSize: '1.5rem',
              fontWeight: '700',
              color: 'var(--color-text-primary)',
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              📊 Analysis Results
            </h2>
            <GroundwaterAnalysisCard
              analysisData={demoData}
              onVisualizationClick={handleVisualizationClick}
            />
          </div>
        )}

        {/* Features Showcase */}
        <div style={{
          marginTop: '4rem',
          background: 'var(--color-surface)',
          padding: '2rem',
          borderRadius: '16px',
          border: '1px solid var(--color-border)'
        }}>
          <h2 style={{
            fontSize: '1.5rem',
            fontWeight: '700',
            color: 'var(--color-text-primary)',
            marginBottom: '1.5rem',
            textAlign: 'center'
          }}>
            🎯 Key Features
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '1.5rem'
          }}>
            {[
              {
                icon: '🟢🟡🔴⚫',
                title: 'Color-coded Criticality',
                description: 'Visual status indicators for Safe, Semi-Critical, Critical, and Over-Exploited areas'
              },
              {
                icon: '📊',
                title: 'Interactive Charts',
                description: 'Real-time data visualizations with pie charts, bar charts, and gauge charts'
              },
              {
                icon: '💧',
                title: 'Quality Analysis',
                description: 'Detailed water quality assessments with health impact explanations'
              },
              {
                icon: '📍',
                title: 'Location-based Analysis',
                description: 'Coordinate-based groundwater analysis for any location in India'
              },
              {
                icon: '🤖',
                title: 'AI-powered Insights',
                description: 'Intelligent recommendations and comprehensive data analysis'
              },
              {
                icon: '📈',
                title: 'Real-time Data',
                description: 'Live data from INGRES database with up-to-date assessments'
              }
            ].map((feature, index) => (
              <div key={index} style={{
                textAlign: 'center',
                padding: '1rem'
              }}>
                <div style={{
                  fontSize: '2rem',
                  marginBottom: '0.75rem'
                }}>
                  {feature.icon}
                </div>
                <h3 style={{
                  margin: '0 0 0.5rem 0',
                  fontSize: '1rem',
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  {feature.title}
                </h3>
                <p style={{
                  margin: 0,
                  fontSize: '0.85rem',
                  color: 'var(--color-text-secondary)',
                  lineHeight: '1.4'
                }}>
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Visualization Modal */}
      <VisualizationModal
        isOpen={showVisualizationModal}
        onClose={() => setShowVisualizationModal(false)}
        visualizationData={selectedVisualization}
      />

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

export default GroundwaterDemo
