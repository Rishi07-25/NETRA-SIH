import React from 'react';

/**
 * NetworkStats Component
 * Displays live bandwidth, flow count, and packet metrics.
 * Placeholder for future implementation.
 */
export default function NetworkStats({ stats = {} }) {
  return (
    <div className="network-stats">
      <h3>Network Flow Telemetry</h3>
      <div>Active Flows: {stats.active_flows || 0}</div>
      <div>Packets/sec: {stats.packets_per_second || 0}</div>
    </div>
  );
}
