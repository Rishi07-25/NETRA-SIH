import React from 'react';
import ThreatScore from '../components/ThreatScore';
import ForecastCard from '../components/ForecastCard';
import NetworkStats from '../components/NetworkStats';
import ThreatTimeline from '../components/ThreatTimeline';
import AttackTable from '../components/AttackTable';
import EventFeed from '../components/EventFeed';

/**
 * Dashboard Page
 * Primary Security Operations Center (SOC) operational dashboard.
 * Placeholder for future implementation.
 */
export default function Dashboard() {
  return (
    <div className="dashboard-container">
      <header>
        <h1>NETRA SOC Dashboard</h1>
        <p>Network Threat Early-warning & Risk Analytics</p>
      </header>
      <div className="grid">
        <ThreatScore score={12} />
        <ForecastCard horizon="5m" risk={15} probability={0.15} />
        <NetworkStats stats={{ active_flows: 142, packets_per_second: 1200 }} />
      </div>
      <ThreatTimeline history={[]} />
      <div className="grid">
        <AttackTable attacks={[]} />
        <EventFeed events={[]} />
      </div>
    </div>
  );
}
