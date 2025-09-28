import React, { useState } from 'react';
import LocationDropdown from '../components/LocationDropdown';
import TalukDataCard from '../components/TalukDataCard';

const DropdownDemo = () => {
  const [selectedState, setSelectedState] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [selectedTaluk, setSelectedTaluk] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [enableTaluk, setEnableTaluk] = useState(true);

  const handleLocationSelect = (state, district, taluk) => {
    setSelectedState(state);
    setSelectedDistrict(district);
    setSelectedTaluk(taluk);
    
    if (state && district) {
      analyzeLocation(state, district, taluk);
    }
  };

  const analyzeLocation = async (state, district, taluk) => {
    try {
      setLoading(true);
      setAnalysisResult(null);
      
      // Create a query for the selected location
      let query;
      let location;
      
      if (taluk) {
        query = `groundwater estimation in ${taluk}, ${district}, ${state}`;
        location = { state, district, taluk };
      } else {
        query = `groundwater estimation in ${district}, ${state}`;
        location = { state, district };
      }
      
      // Call the groundwater analysis API
      const response = await fetch('/ingres/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          limit: 10
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setAnalysisResult({
          query: query,
          response: data.response,
          location: location
        });
      } else {
        setAnalysisResult({
          query: query,
          error: data.error || 'Analysis failed',
          location: location
        });
      }
    } catch (error) {
      setAnalysisResult({
        query: taluk 
          ? `groundwater estimation in ${taluk}, ${district}, ${state}`
          : `groundwater estimation in ${district}, ${state}`,
        error: 'Error: ' + error.message,
        location: { state, district, taluk }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dropdown-demo">
      <div className="container">
        <h1>Groundwater Location Dropdown Demo</h1>
        <p>Select a state, district, and optionally a taluk to analyze groundwater data:</p>
        
        <div className="demo-section">
          <div style={{ marginBottom: '20px' }}>
            <label>
              <input
                type="checkbox"
                checked={enableTaluk}
                onChange={(e) => setEnableTaluk(e.target.checked)}
                style={{ marginRight: '8px' }}
              />
              Enable Taluk Selection (1,203 taluks available)
            </label>
          </div>
          
          <LocationDropdown
            onLocationSelect={handleLocationSelect}
            selectedState={selectedState}
            selectedDistrict={selectedDistrict}
            selectedTaluk={selectedTaluk}
            enableTaluk={enableTaluk}
          />
        </div>

        {loading && (
          <div className="loading">
            <p>Analyzing groundwater data for {selectedTaluk ? `${selectedTaluk}, ` : ''}{selectedDistrict}, {selectedState}...</p>
          </div>
        )}

        {analysisResult && (
          <div className="analysis-result">
            <h3>Analysis Result</h3>
            <div className="result-header">
              <p><strong>Location:</strong> 
                {analysisResult.location.taluk ? `${analysisResult.location.taluk}, ` : ''}
                {analysisResult.location.district}, {analysisResult.location.state}
              </p>
              <p><strong>Query:</strong> {analysisResult.query}</p>
            </div>
            
            {analysisResult.error ? (
              <div className="error">
                <p style={{ color: 'red' }}>{analysisResult.error}</p>
              </div>
            ) : (
              <div className="response">
                <h4>Groundwater Analysis:</h4>
                <div className="response-content">
                  {analysisResult.response}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Show Taluk Data Card when a taluk is selected */}
        {selectedState && selectedDistrict && selectedTaluk && enableTaluk && (
          <TalukDataCard 
            state={selectedState}
            district={selectedDistrict}
            taluk={selectedTaluk}
          />
        )}

        <div className="info-section">
          <h3>Available Data</h3>
          <ul>
            <li><strong>37 States</strong> across India</li>
            <li><strong>751 Districts</strong> with groundwater data</li>
            <li><strong>1,203 Taluks</strong> with detailed groundwater data (NEW!)</li>
            <li>Comprehensive coverage including major cities and regions</li>
            <li>Real-time groundwater analysis and recommendations</li>
          </ul>
          
          <h4>Special Features:</h4>
          <ul>
            <li>✅ <strong>Ooty Support:</strong> "THE NILGIRIS" district in Tamil Nadu</li>
            <li>✅ <strong>Smart Location Detection:</strong> Handles various location name formats</li>
            <li>✅ <strong>Hierarchical Selection:</strong> State → District → Taluk selection</li>
            <li>✅ <strong>Taluk-level Analysis:</strong> Detailed groundwater data at taluk level</li>
            <li>✅ <strong>Real-time Analysis:</strong> Instant groundwater data analysis</li>
          </ul>
          
          <h4>Example Taluk Data (Chikkamagaluru District):</h4>
          <ul>
            <li><strong>Ajjampura:</strong> Over-exploited (146.97% extraction)</li>
            <li><strong>Koppa:</strong> Safe (27.73% extraction)</li>
            <li><strong>Narasimharajapura:</strong> Safe (19.81% extraction)</li>
            <li><strong>Chikmagalur:</strong> Safe (42.19% extraction)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DropdownDemo;
