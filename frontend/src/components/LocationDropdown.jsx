import React, { useState, useEffect } from 'react';
import './LocationDropdown.css';

const LocationDropdown = ({ onLocationSelect, selectedState, selectedDistrict, selectedTaluk, enableTaluk = true }) => {
  const [states, setStates] = useState([]);
  const [districts, setDistricts] = useState([]);
  const [taluks, setTaluks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load states on component mount
  useEffect(() => {
    loadStates();
  }, []);

  // Load districts when state changes
  useEffect(() => {
    if (selectedState) {
      loadDistricts(selectedState);
    } else {
      setDistricts([]);
      setTaluks([]);
    }
  }, [selectedState]);

  // Load taluks when district changes
  useEffect(() => {
    if (selectedState && selectedDistrict && enableTaluk) {
      loadTaluks(selectedState, selectedDistrict);
    } else {
      setTaluks([]);
    }
  }, [selectedState, selectedDistrict, enableTaluk]);

  const loadStates = async () => {
    try {
      setLoading(true);
      const response = await fetch('/dropdown/states');
      const data = await response.json();
      
      if (data.success) {
        setStates(data.states);
      } else {
        setError('Failed to load states');
      }
    } catch (err) {
      setError('Error loading states: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadDistricts = async (state) => {
    try {
      setLoading(true);
      const response = await fetch(`/dropdown/districts/${encodeURIComponent(state)}`);
      const data = await response.json();
      
      if (data.success) {
        setDistricts(data.districts);
      } else {
        setError('Failed to load districts');
        setDistricts([]);
      }
    } catch (err) {
      setError('Error loading districts: ' + err.message);
      setDistricts([]);
    } finally {
      setLoading(false);
    }
  };

  const loadTaluks = async (state, district) => {
    try {
      setLoading(true);
      const response = await fetch(`/dropdown/taluks/${encodeURIComponent(state)}/${encodeURIComponent(district)}`);
      const data = await response.json();
      
      if (data.success) {
        setTaluks(data.taluks);
      } else {
        setError('Failed to load taluks');
        setTaluks([]);
      }
    } catch (err) {
      setError('Error loading taluks: ' + err.message);
      setTaluks([]);
    } finally {
      setLoading(false);
    }
  };

  const handleStateChange = (event) => {
    const newState = event.target.value;
    onLocationSelect(newState, null, null);
  };

  const handleDistrictChange = (event) => {
    const newDistrict = event.target.value;
    onLocationSelect(selectedState, newDistrict, null);
  };

  const handleTalukChange = (event) => {
    const newTaluk = event.target.value;
    onLocationSelect(selectedState, selectedDistrict, newTaluk);
  };

  if (error) {
    return (
      <div className="location-dropdown error">
        <p style={{ color: 'red' }}>{error}</p>
        <button onClick={loadStates}>Retry</button>
      </div>
    );
  }

  return (
    <div className="location-dropdown">
      <h3>Select Location</h3>
      
      <div className="dropdown-group">
        <label htmlFor="state-select">State:</label>
        <select
          id="state-select"
          value={selectedState || ''}
          onChange={handleStateChange}
          disabled={loading}
        >
          <option value="">Select a state...</option>
          {states.map((state) => (
            <option key={state} value={state}>
              {state}
            </option>
          ))}
        </select>
      </div>

      <div className="dropdown-group">
        <label htmlFor="district-select">District:</label>
        <select
          id="district-select"
          value={selectedDistrict || ''}
          onChange={handleDistrictChange}
          disabled={loading || !selectedState}
        >
          <option value="">Select a district...</option>
          {districts.map((district) => (
            <option key={district} value={district}>
              {district}
            </option>
          ))}
        </select>
      </div>

      {enableTaluk && (
        <div className="dropdown-group taluk-group">
          <label htmlFor="taluk-select">Taluk (Optional):</label>
          <select
            id="taluk-select"
            value={selectedTaluk || ''}
            onChange={handleTalukChange}
            disabled={loading || !selectedState || !selectedDistrict}
          >
            <option value="">Select a taluk...</option>
            {taluks.map((taluk) => (
              <option key={taluk} value={taluk}>
                {taluk}
              </option>
            ))}
          </select>
        </div>
      )}

      {loading && <p>Loading...</p>}
      
      {selectedState && selectedDistrict && (
        <div className={`selected-location ${selectedTaluk && enableTaluk ? 'taluk-selected' : ''}`}>
          <p><strong>Selected:</strong> 
            {selectedTaluk && enableTaluk 
              ? `${selectedTaluk}, ${selectedDistrict}, ${selectedState}`
              : `${selectedDistrict}, ${selectedState}`
            }
          </p>
          <button 
            onClick={() => onLocationSelect(selectedState, selectedDistrict, selectedTaluk)}
            className="analyze-btn"
          >
            {enableTaluk && selectedTaluk 
              ? 'Analyze Taluk Groundwater Data'
              : 'Analyze District Groundwater Data'
            }
          </button>
        </div>
      )}
    </div>
  );
};

export default LocationDropdown;
