import React from 'react';

/**
 * ThreatTimeline Component
 * Chronological line chart tracking threat score trajectory.
 * Placeholder for future implementation.
 */
export default function ThreatTimeline({ history = [] }) {
  return (
    <div className="threat-timeline">
      <h3>Temporal Threat Trajectory</h3>
      <p>Data points tracked: {history.length}</p>
    </div>
  );
}
