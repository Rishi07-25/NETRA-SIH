import React from 'react';

/**
 * ThreatScore Component
 * Visual gauge displaying current composite threat score (0-100).
 * Placeholder for future implementation.
 */
export default function ThreatScore({ score = 0 }) {
  return (
    <div className="threat-score-card">
      <h3>Composite Threat Score</h3>
      <div className="score-display">{score} / 100</div>
    </div>
  );
}
