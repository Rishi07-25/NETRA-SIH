import React from 'react';

/**
 * ForecastCard Component
 * Displays forecasted threat probability across a forward time horizon.
 * Placeholder for future implementation.
 */
export default function ForecastCard({ horizon = '5m', risk = 0, probability = 0 }) {
  return (
    <div className="forecast-card">
      <h4>Horizon: {horizon}</h4>
      <p>Risk: {risk}%</p>
      <p>Probability: {(probability * 100).toFixed(1)}%</p>
    </div>
  );
}
