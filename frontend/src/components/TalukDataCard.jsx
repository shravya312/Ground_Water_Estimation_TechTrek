import React, { useState, useEffect } from 'react';

const TalukDataCard = ({ state, district, taluk }) => {
  const [talukData, setTalukData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (state && district && taluk) {
      loadTalukData();
    }
  }, [state, district, taluk]);

  const loadTalukData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`/dropdown/taluk-data/${encodeURIComponent(state)}/${encodeURIComponent(district)}/${encodeURIComponent(taluk)}`);
      const data = await response.json();
      
      if (data.success) {
        setTalukData(data.data);
      } else {
        setError(data.error || 'Failed to load taluk data');
      }
    } catch (err) {
      setError('Error loading taluk data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getCategorizationColor = (categorization) => {
    switch (categorization?.toLowerCase()) {
      case 'safe':
        return '#27ae60';
      case 'semi_critical':
        return '#f39c12';
      case 'critical':
        return '#e67e22';
      case 'over_exploited':
        return '#e74c3c';
      default:
        return '#95a5a6';
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend?.toLowerCase()) {
      case 'rising':
        return '📈';
      case 'falling':
        return '📉';
      case 'stable':
        return '➡️';
      default:
        return '❓';
    }
  };

  if (loading) {
    return (
      <div className="taluk-data-card loading">
        <h3>Loading Taluk Data...</h3>
        <p>Fetching detailed groundwater information for {taluk}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="taluk-data-card error">
        <h3>Error Loading Taluk Data</h3>
        <p style={{ color: 'red' }}>{error}</p>
        <button onClick={loadTalukData}>Retry</button>
      </div>
    );
  }

  if (!talukData) {
    return null;
  }

  return (
    <div className="taluk-data-card">
      <h3>📊 Taluk Groundwater Data: {taluk}</h3>
      
      <div className="taluk-data-grid">
        <div className="data-item">
          <label>Extraction Status:</label>
          <div 
            className="categorization-badge"
            style={{ backgroundColor: getCategorizationColor(talukData.categorization) }}
          >
            {talukData.categorization?.replace('_', ' ').toUpperCase() || 'N/A'}
          </div>
        </div>

        <div className="data-item">
          <label>Extraction Percentage:</label>
          <span className="extraction-percentage">
            {talukData.stage_of_ground_water_extraction || 'N/A'}%
          </span>
        </div>

        <div className="data-item">
          <label>Rainfall:</label>
          <span>{talukData.rainfall_mm || 'N/A'} mm</span>
        </div>

        <div className="data-item">
          <label>Groundwater Recharge:</label>
          <span>{talukData.ground_water_recharge_ham || 'N/A'} ham</span>
        </div>

        <div className="data-item">
          <label>Groundwater Extraction:</label>
          <span>{talukData.ground_water_extraction_ham || 'N/A'} ham</span>
        </div>

        <div className="data-item">
          <label>Pre-Monsoon Trend:</label>
          <span className="trend">
            {getTrendIcon(talukData.pre_monsoon_trend)} {talukData.pre_monsoon_trend || 'N/A'}
          </span>
        </div>

        <div className="data-item">
          <label>Post-Monsoon Trend:</label>
          <span className="trend">
            {getTrendIcon(talukData.post_monsoon_trend)} {talukData.post_monsoon_trend || 'N/A'}
          </span>
        </div>

        <div className="data-item">
          <label>Assessment Year:</label>
          <span>{talukData.year || 'N/A'}</span>
        </div>
      </div>

      <div className="taluk-insights">
        <h4>💡 Key Insights:</h4>
        <ul>
          {talukData.categorization === 'over_exploited' && (
            <li className="warning">⚠️ This taluk is over-exploited and requires immediate attention</li>
          )}
          {talukData.categorization === 'safe' && (
            <li className="success">✅ This taluk has sustainable groundwater levels</li>
          )}
          {talukData.pre_monsoon_trend === 'falling' && talukData.post_monsoon_trend === 'falling' && (
            <li className="warning">📉 Both pre and post-monsoon trends are falling - concerning pattern</li>
          )}
          {talukData.pre_monsoon_trend === 'rising' && talukData.post_monsoon_trend === 'rising' && (
            <li className="success">📈 Both pre and post-monsoon trends are rising - positive pattern</li>
          )}
          {parseFloat(talukData.stage_of_ground_water_extraction) > 100 && (
            <li className="warning">🚨 Extraction exceeds 100% - unsustainable usage</li>
          )}
          {parseFloat(talukData.stage_of_ground_water_extraction) < 70 && (
            <li className="success">💧 Extraction below 70% - good water management</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default TalukDataCard;
