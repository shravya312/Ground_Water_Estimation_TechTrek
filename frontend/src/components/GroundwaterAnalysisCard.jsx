import React from 'react'
import PlotlyChart from './PlotlyChart'

const GroundwaterAnalysisCard = ({ analysisData, onVisualizationClick }) => {
  if (!analysisData) return null

  const {
    criticality_status,
    criticality_emoji,
    numerical_values,
    recommendations,
    quality_analysis,
    visualizations,
    comparison_data
  } = analysisData

  const getCriticalityColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'safe': return '#22c55e' // Green
      case 'semi-critical': return '#eab308' // Yellow
      case 'critical': return '#ef4444' // Red
      case 'over-exploited': return '#6b7280' // Gray
      default: return '#6b7280'
    }
  }

  const getCriticalityBgColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'safe': return 'rgba(34, 197, 94, 0.1)'
      case 'semi-critical': return 'rgba(234, 179, 8, 0.1)'
      case 'critical': return 'rgba(239, 68, 68, 0.1)'
      case 'over-exploited': return 'rgba(107, 114, 128, 0.1)'
      default: return 'rgba(107, 114, 128, 0.1)'
    }
  }

  const getQualitySeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'good': return '#22c55e'
      case 'moderate': return '#eab308'
      case 'poor': return '#ef4444'
      case 'unknown': return '#6b7280'
      default: return '#6b7280'
    }
  }

  return (
    <div className="groundwater-analysis-card" style={{
      background: 'var(--color-surface)',
      borderRadius: '16px',
      padding: '1.5rem',
      margin: '1rem 0',
      border: '1px solid var(--color-border)',
      boxShadow: 'var(--shadow-lg)'
    }}>
      {/* Criticality Status Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1.5rem',
        padding: '1rem',
        borderRadius: '12px',
        background: getCriticalityBgColor(criticality_status),
        border: `2px solid ${getCriticalityColor(criticality_status)}`
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '2rem' }}>{criticality_emoji}</span>
          <div>
            <h3 style={{ 
              margin: 0, 
              fontSize: '1.25rem', 
              fontWeight: '700',
              color: getCriticalityColor(criticality_status)
            }}>
              {criticality_status || 'Unknown Status'}
            </h3>
            <p style={{ 
              margin: 0, 
              fontSize: '0.9rem', 
              opacity: 0.8,
              color: 'var(--color-text-primary)'
            }}>
              Groundwater Criticality Level
            </p>
          </div>
        </div>
        {numerical_values?.extraction_stage && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: '700',
              color: getCriticalityColor(criticality_status)
            }}>
              {numerical_values.extraction_stage.toFixed(1)}%
            </div>
            <div style={{ 
              fontSize: '0.8rem', 
              opacity: 0.8,
              color: 'var(--color-text-primary)'
            }}>
              Extraction Stage
            </div>
          </div>
        )}
      </div>

      {/* Key Metrics Grid */}
      {numerical_values && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1rem',
          marginBottom: '1.5rem'
        }}>
          {Object.entries(numerical_values).map(([key, value]) => (
            <div key={key} style={{
              background: 'var(--color-surface-elevated)',
              padding: '1rem',
              borderRadius: '8px',
              border: '1px solid var(--color-border)'
            }}>
              <div style={{
                fontSize: '0.8rem',
                color: 'var(--color-text-secondary)',
                textTransform: 'uppercase',
                fontWeight: '600',
                marginBottom: '0.25rem'
              }}>
                {key.replace(/_/g, ' ')}
              </div>
              <div style={{
                fontSize: '1.1rem',
                fontWeight: '700',
                color: 'var(--color-text-primary)'
              }}>
                {typeof value === 'number' ? value.toLocaleString() : value}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Quality Analysis */}
      {quality_analysis && (
        <div style={{
          background: 'var(--color-surface-elevated)',
          padding: '1.25rem',
          borderRadius: '12px',
          marginBottom: '1.5rem',
          border: '1px solid var(--color-border)'
        }}>
          <h4 style={{
            margin: '0 0 1rem 0',
            fontSize: '1.1rem',
            fontWeight: '600',
            color: 'var(--color-text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            💧 Water Quality Analysis
            <span style={{
              padding: '0.25rem 0.5rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: '600',
              background: getQualitySeverityColor(quality_analysis.severity),
              color: 'white'
            }}>
              {quality_analysis.severity || 'Unknown'}
            </span>
          </h4>

          {/* Quality Issues */}
          {quality_analysis.issues && quality_analysis.issues.length > 0 && (
            <div style={{ marginBottom: '1rem' }}>
              <h5 style={{ 
                margin: '0 0 0.5rem 0', 
                fontSize: '0.9rem', 
                fontWeight: '600',
                color: 'var(--color-text-primary)'
              }}>
                🚨 Quality Issues:
              </h5>
              <ul style={{ margin: 0, paddingLeft: '1.25rem' }}>
                {quality_analysis.issues.map((issue, index) => (
                  <li key={index} style={{
                    marginBottom: '0.25rem',
                    color: 'var(--color-text-primary)',
                    fontSize: '0.9rem'
                  }}>
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Quality Parameters */}
          {(quality_analysis.major_parameters || quality_analysis.other_parameters) && (
            <div style={{ marginBottom: '1rem' }}>
              <h5 style={{ 
                margin: '0 0 0.5rem 0', 
                fontSize: '0.9rem', 
                fontWeight: '600',
                color: 'var(--color-text-primary)'
              }}>
                📋 Quality Parameters:
              </h5>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                {quality_analysis.major_parameters && quality_analysis.major_parameters !== 'None' && (
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: 'rgba(239, 68, 68, 0.1)',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    color: '#ef4444'
                  }}>
                    Major: {quality_analysis.major_parameters}
                  </div>
                )}
                {quality_analysis.other_parameters && quality_analysis.other_parameters !== 'None' && (
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: 'rgba(234, 179, 8, 0.1)',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    color: '#eab308'
                  }}>
                    Other: {quality_analysis.other_parameters}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Quality Explanations */}
          {quality_analysis.explanations && quality_analysis.explanations.length > 0 && (
            <div>
              <h5 style={{ 
                margin: '0 0 0.5rem 0', 
                fontSize: '0.9rem', 
                fontWeight: '600',
                color: 'var(--color-text-primary)'
              }}>
                📋 Detailed Explanations:
              </h5>
              {quality_analysis.explanations.map((explanation, index) => (
                <div key={index} style={{
                  background: 'var(--color-surface)',
                  padding: '0.75rem',
                  borderRadius: '8px',
                  marginBottom: '0.5rem',
                  border: '1px solid var(--color-border)'
                }}>
                  <div style={{
                    fontWeight: '600',
                    color: 'var(--color-text-primary)',
                    marginBottom: '0.25rem'
                  }}>
                    🔬 {explanation.parameter} ({explanation.level})
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                    <div><strong>Health Impact:</strong> {explanation.health_impact}</div>
                    <div><strong>Sources:</strong> {explanation.sources}</div>
                    <div><strong>Standards:</strong> {explanation.standards}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div style={{
          background: 'var(--color-surface-elevated)',
          padding: '1.25rem',
          borderRadius: '12px',
          marginBottom: '1.5rem',
          border: '1px solid var(--color-border)'
        }}>
          <h4 style={{
            margin: '0 0 1rem 0',
            fontSize: '1.1rem',
            fontWeight: '600',
            color: 'var(--color-text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            💡 Recommendations
          </h4>
          <ul style={{ margin: 0, paddingLeft: '1.25rem' }}>
            {recommendations.map((rec, index) => (
              <li key={index} style={{
                marginBottom: '0.5rem',
                color: 'var(--color-text-primary)',
                fontSize: '0.9rem',
                lineHeight: '1.5'
              }}>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Visualizations */}
      {visualizations && visualizations.length > 0 && (
        <div style={{
          background: 'var(--color-surface-elevated)',
          padding: '1.25rem',
          borderRadius: '12px',
          marginBottom: '1.5rem',
          border: '1px solid var(--color-border)'
        }}>
          <h4 style={{
            margin: '0 0 1rem 0',
            fontSize: '1.1rem',
            fontWeight: '600',
            color: 'var(--color-text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            📊 Interactive Visualizations
          </h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '1rem'
          }}>
            {visualizations.map((viz, index) => (
              <div key={index} style={{
                background: 'var(--color-surface)',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid var(--color-border)',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onClick={() => onVisualizationClick && onVisualizationClick(viz)}
              onMouseOver={(e) => {
                e.target.style.transform = 'translateY(-2px)'
                e.target.style.boxShadow = 'var(--shadow-lg)'
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)'
                e.target.style.boxShadow = 'none'
              }}
              >
                <div style={{
                  fontSize: '1.5rem',
                  marginBottom: '0.5rem'
                }}>
                  {viz.type === 'pie_chart' && '🥧'}
                  {viz.type === 'bar_chart' && '📊'}
                  {viz.type === 'gauge_chart' && '🎯'}
                </div>
                <div style={{
                  fontWeight: '600',
                  color: 'var(--color-text-primary)',
                  marginBottom: '0.25rem'
                }}>
                  {viz.title}
                </div>
                <div style={{
                  fontSize: '0.8rem',
                  color: 'var(--color-text-secondary)'
                }}>
                  Click to view interactive chart
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Data */}
      {comparison_data && (
        <div style={{
          background: 'var(--color-surface-elevated)',
          padding: '1.25rem',
          borderRadius: '12px',
          border: '1px solid var(--color-border)'
        }}>
          <h4 style={{
            margin: '0 0 1rem 0',
            fontSize: '1.1rem',
            fontWeight: '600',
            color: 'var(--color-text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            📈 Comparison with National Average
          </h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '1rem'
          }}>
            {Object.entries(comparison_data).map(([key, value]) => (
              <div key={key} style={{
                textAlign: 'center',
                padding: '0.75rem',
                background: 'var(--color-surface)',
                borderRadius: '8px',
                border: '1px solid var(--color-border)'
              }}>
                <div style={{
                  fontSize: '0.8rem',
                  color: 'var(--color-text-secondary)',
                  marginBottom: '0.25rem'
                }}>
                  {key.replace(/_/g, ' ')}
                </div>
                <div style={{
                  fontSize: '1.1rem',
                  fontWeight: '700',
                  color: 'var(--color-text-primary)'
                }}>
                  {typeof value === 'number' ? value.toFixed(1) : value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Resources Section */}
      {analysisData.additional_resources && (
        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ 
            margin: '0 0 1rem 0', 
            fontSize: '1.1rem', 
            fontWeight: '600',
            color: 'var(--color-text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            🌊 Additional Resources & Analysis
          </h4>
          
          <div style={{
            background: 'var(--color-surface)',
            borderRadius: '12px',
            padding: '1.5rem',
            border: '1px solid var(--color-border)'
          }}>
            {/* Coastal Areas */}
            {analysisData.additional_resources.coastal_areas && (
              <div style={{ marginBottom: '1rem' }}>
                <h5 style={{ 
                  margin: '0 0 0.5rem 0', 
                  fontSize: '0.9rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  🏖️ Coastal Areas:
                </h5>
                <p style={{ 
                  margin: '0', 
                  fontSize: '0.85rem', 
                  color: 'var(--color-text-secondary)',
                  lineHeight: '1.5'
                }}>
                  {analysisData.additional_resources.coastal_areas}
                </p>
              </div>
            )}

            {/* Aquifer Types */}
            {analysisData.additional_resources.aquifer_types && (
              <div style={{ marginBottom: '1rem' }}>
                <h5 style={{ 
                  margin: '0 0 0.5rem 0', 
                  fontSize: '0.9rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  🕳️ Aquifer Types:
                </h5>
                <p style={{ 
                  margin: '0', 
                  fontSize: '0.85rem', 
                  color: 'var(--color-text-secondary)',
                  lineHeight: '1.5'
                }}>
                  {analysisData.additional_resources.aquifer_types}
                </p>
              </div>
            )}

            {/* Additional Resources */}
            {analysisData.additional_resources.additional_resources && (
              <div style={{ marginBottom: '1rem' }}>
                <h5 style={{ 
                  margin: '0 0 0.5rem 0', 
                  fontSize: '0.9rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  💎 Additional Resources:
                </h5>
                <p style={{ 
                  margin: '0', 
                  fontSize: '0.85rem', 
                  color: 'var(--color-text-secondary)',
                  lineHeight: '1.5'
                }}>
                  {analysisData.additional_resources.additional_resources}
                </p>
              </div>
            )}

            {/* AI Analysis */}
            {analysisData.additional_resources.analysis && (
              <div>
                <h5 style={{ 
                  margin: '0 0 0.5rem 0', 
                  fontSize: '0.9rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  🤖 AI-Generated Analysis:
                </h5>
                <div style={{ 
                  background: 'rgba(59, 130, 246, 0.1)',
                  borderRadius: '8px',
                  padding: '1rem',
                  fontSize: '0.85rem', 
                  color: 'var(--color-text-primary)',
                  lineHeight: '1.6',
                  whiteSpace: 'pre-line'
                }}>
                  {analysisData.additional_resources.analysis}
                </div>
                {analysisData.additional_resources.generated_by && (
                  <div style={{ 
                    marginTop: '0.5rem',
                    fontSize: '0.75rem',
                    color: 'var(--color-text-muted)',
                    fontStyle: 'italic'
                  }}>
                    Generated by: {analysisData.additional_resources.generated_by}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Key Findings and Trends Section */}
      {analysisData.key_findings_trends && (
        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ 
            margin: '0 0 1rem 0', 
            fontSize: '1.1rem', 
            fontWeight: '600',
            color: 'var(--color-text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            📊 Key Findings and Trends
          </h4>
          
          <div style={{
            background: 'var(--color-surface)',
            borderRadius: '12px',
            padding: '1.5rem',
            border: '1px solid var(--color-border)'
          }}>
            {/* Data Summary */}
            {analysisData.key_findings_trends.data_summary && (
              <div style={{ marginBottom: '1rem' }}>
                <h5 style={{ 
                  margin: '0 0 0.5rem 0', 
                  fontSize: '0.9rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  📈 Quick Summary:
                </h5>
                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: analysisData.key_findings_trends.data_summary.over_extraction_risk === 'High' ? 'rgba(239, 68, 68, 0.1)' : 
                               analysisData.key_findings_trends.data_summary.over_extraction_risk === 'Moderate' ? 'rgba(234, 179, 8, 0.1)' : 'rgba(34, 197, 94, 0.1)',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    color: analysisData.key_findings_trends.data_summary.over_extraction_risk === 'High' ? '#ef4444' : 
                           analysisData.key_findings_trends.data_summary.over_extraction_risk === 'Moderate' ? '#eab308' : '#22c55e'
                  }}>
                    Over-extraction Risk: {analysisData.key_findings_trends.data_summary.over_extraction_risk}
                  </div>
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: 'rgba(59, 130, 246, 0.1)',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    color: '#3b82f6'
                  }}>
                    Rainfall Dependency: {analysisData.key_findings_trends.data_summary.rainfall_dependency}
                  </div>
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: analysisData.key_findings_trends.data_summary.sustainability_status === 'Critical' ? 'rgba(239, 68, 68, 0.1)' : 
                               analysisData.key_findings_trends.data_summary.sustainability_status === 'At Risk' ? 'rgba(234, 179, 8, 0.1)' : 'rgba(34, 197, 94, 0.1)',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    color: analysisData.key_findings_trends.data_summary.sustainability_status === 'Critical' ? '#ef4444' : 
                           analysisData.key_findings_trends.data_summary.sustainability_status === 'At Risk' ? '#eab308' : '#22c55e'
                  }}>
                    Sustainability: {analysisData.key_findings_trends.data_summary.sustainability_status}
                  </div>
                </div>
              </div>
            )}

            {/* AI Analysis */}
            {analysisData.key_findings_trends.analysis && (
              <div>
                <h5 style={{ 
                  margin: '0 0 0.5rem 0', 
                  fontSize: '0.9rem', 
                  fontWeight: '600',
                  color: 'var(--color-text-primary)'
                }}>
                  🔍 Comprehensive Analysis:
                </h5>
                <div style={{ 
                  background: 'rgba(16, 185, 129, 0.1)',
                  borderRadius: '8px',
                  padding: '1rem',
                  fontSize: '0.85rem', 
                  color: 'var(--color-text-primary)',
                  lineHeight: '1.6',
                  whiteSpace: 'pre-line'
                }}>
                  {analysisData.key_findings_trends.analysis}
                </div>
                {analysisData.key_findings_trends.generated_by && (
                  <div style={{ 
                    marginTop: '0.5rem',
                    fontSize: '0.75rem',
                    color: 'var(--color-text-muted)',
                    fontStyle: 'italic'
                  }}>
                    Generated by: {analysisData.key_findings_trends.generated_by}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default GroundwaterAnalysisCard
