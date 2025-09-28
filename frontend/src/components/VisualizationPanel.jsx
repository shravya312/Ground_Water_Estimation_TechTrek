import React, { useState, useEffect } from 'react';
import './VisualizationPanel.css';

const VisualizationPanel = ({ response, onDownload }) => {
  const [visualizations, setVisualizations] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (response && response.includes('## 📊 Interactive Visualizations')) {
      extractVisualizations(response);
    }
  }, [response]);

  const extractVisualizations = (response) => {
    try {
      const vizSection = response.split('## 📊 Interactive Visualizations')[1];
      if (!vizSection) return;

      const vizTypes = [
        'extraction_trends',
        'recharge_analysis',
        'district_comparison',
        'criticality_distribution',
        'rainfall_correlation',
        'state_overview'
      ];

      const extractedViz = {};
      
      vizTypes.forEach(vizType => {
        const regex = new RegExp(`### ${vizType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}\\s*<div class='visualization-container'>(.*?)</div>`, 's');
        const match = vizSection.match(regex);
        if (match && match[1]) {
          extractedViz[vizType] = match[1];
        }
      });

      setVisualizations(extractedViz);
    } catch (err) {
      console.error('Error extracting visualizations:', err);
      setError('Failed to extract visualizations');
    }
  };

  const handleDownload = async (vizType) => {
    try {
      setLoading(true);
      const response = await fetch(`/visualize/download/${vizType}`);
      const data = await response.json();
      
      if (data.html_content) {
        const blob = new Blob([data.html_content], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${vizType}_visualization.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Download failed:', err);
      setError('Download failed');
    } finally {
      setLoading(false);
    }
  };

  if (Object.keys(visualizations).length === 0) {
    return null;
  }

  return (
    <div className="visualization-panel">
      <div className="visualization-header">
        <h3>📊 Interactive Visualizations</h3>
        <p>Explore the data through interactive charts and graphs</p>
      </div>
      
      <div className="visualization-grid">
        {Object.entries(visualizations).map(([vizType, vizHtml]) => (
          <div key={vizType} className="visualization-card">
            <div className="visualization-card-header">
              <h4>{vizType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
              <button 
                className="download-btn"
                onClick={() => handleDownload(vizType)}
                disabled={loading}
              >
                {loading ? '⏳' : '⬇️'} Download
              </button>
            </div>
            <div 
              className="visualization-content"
              dangerouslySetInnerHTML={{ __html: vizHtml }}
            />
          </div>
        ))}
      </div>
      
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};

export default VisualizationPanel;