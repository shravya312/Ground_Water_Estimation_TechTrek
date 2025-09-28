import React, { useState } from 'react';
import './LocationDropdown.css';

const AdvancedRAGTest = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [config, setConfig] = useState({
    hybrid_alpha: 0.6,
    rerank_top_k: 10,
    query_expansion_terms: 3,
    rerank_min_similarity: 0.1,
    min_similarity_score: 0.1
  });
  const [showConfig, setShowConfig] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setResponse('');

    try {
      const response = await fetch('/ingres/advanced-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          language: 'en',
          user_id: 'test_user'
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        setResponse(data.response);
      } else {
        setResponse(`Error: ${data.response}`);
      }
    } catch (error) {
      setResponse(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const updateConfig = async () => {
    try {
      const response = await fetch('/rag/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      const data = await response.json();
      if (data.message) {
        alert('Configuration updated successfully!');
      }
    } catch (error) {
      alert(`Error updating configuration: ${error.message}`);
    }
  };

  const loadConfig = async () => {
    try {
      const response = await fetch('/rag/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Error loading configuration:', error);
    }
  };

  const exampleQueries = [
    "groundwater estimation in Chikkamagaluru",
    "over-exploited areas in Karnataka",
    "rainfall data for Tamil Nadu",
    "water quality issues in Maharashtra",
    "sustainable groundwater extraction rates"
  ];

  return (
    <div className="advanced-rag-test">
      <h2>🚀 Advanced RAG Testing</h2>
      <p>Test hybrid search, reranking, and query expansion capabilities</p>

      <div className="test-section">
        <h3>Configuration</h3>
        <button 
          onClick={() => setShowConfig(!showConfig)}
          className="config-toggle-btn"
        >
          {showConfig ? 'Hide' : 'Show'} Configuration
        </button>
        
        {showConfig && (
          <div className="config-panel">
            <div className="config-group">
              <label>Hybrid Alpha (Dense vs Sparse):</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.hybrid_alpha}
                onChange={(e) => setConfig({...config, hybrid_alpha: parseFloat(e.target.value)})}
              />
              <span>{config.hybrid_alpha}</span>
            </div>

            <div className="config-group">
              <label>Rerank Top K:</label>
              <input
                type="number"
                min="1"
                max="50"
                value={config.rerank_top_k}
                onChange={(e) => setConfig({...config, rerank_top_k: parseInt(e.target.value)})}
              />
            </div>

            <div className="config-group">
              <label>Query Expansion Terms:</label>
              <input
                type="number"
                min="0"
                max="10"
                value={config.query_expansion_terms}
                onChange={(e) => setConfig({...config, query_expansion_terms: parseInt(e.target.value)})}
              />
            </div>

            <div className="config-group">
              <label>Rerank Min Similarity:</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.rerank_min_similarity}
                onChange={(e) => setConfig({...config, rerank_min_similarity: parseFloat(e.target.value)})}
              />
              <span>{config.rerank_min_similarity}</span>
            </div>

            <div className="config-group">
              <label>Min Similarity Score:</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.min_similarity_score}
                onChange={(e) => setConfig({...config, min_similarity_score: parseFloat(e.target.value)})}
              />
              <span>{config.min_similarity_score}</span>
            </div>

            <div className="config-actions">
              <button onClick={loadConfig} className="load-config-btn">
                Load Current Config
              </button>
              <button onClick={updateConfig} className="update-config-btn">
                Update Configuration
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="test-section">
        <h3>Query Testing</h3>
        
        <div className="query-input-group">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your groundwater query..."
            className="query-input"
            onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
          />
          <button 
            onClick={handleQuery} 
            disabled={loading || !query.trim()}
            className="query-btn"
          >
            {loading ? 'Processing...' : 'Test Advanced RAG'}
          </button>
        </div>

        <div className="example-queries">
          <h4>Example Queries:</h4>
          <div className="query-examples">
            {exampleQueries.map((example, index) => (
              <button
                key={index}
                onClick={() => setQuery(example)}
                className="example-query-btn"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </div>

      {response && (
        <div className="test-section">
          <h3>Response</h3>
          <div className="response-container">
            <pre className="response-text">{response}</pre>
          </div>
        </div>
      )}

      <div className="test-section">
        <h3>RAG Features</h3>
        <div className="features-grid">
          <div className="feature-card">
            <h4>🔍 Hybrid Search</h4>
            <p>Combines dense (vector) and sparse (BM25) retrieval for comprehensive coverage</p>
          </div>
          <div className="feature-card">
            <h4>📝 Query Expansion</h4>
            <p>Enhances queries with domain-specific groundwater terminology</p>
          </div>
          <div className="feature-card">
            <h4>🔄 Advanced Reranking</h4>
            <p>Improves result relevance through semantic similarity scoring</p>
          </div>
          <div className="feature-card">
            <h4>⚙️ Configurable</h4>
            <p>Adjustable thresholds and parameters for fine-tuning performance</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedRAGTest;
