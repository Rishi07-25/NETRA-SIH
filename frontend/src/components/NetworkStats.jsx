import React from 'react';

/**
 * NetworkStats Component
 * Displays live bandwidth, flow count, and packet metrics.
 */
export default function NetworkStats({ stats = {} }) {
  const activeFlows = stats.active_flows || 0;
  const pps = stats.packets_per_second ? stats.packets_per_second.toLocaleString() : '0';
  const bps = stats.bytes_per_second ? (stats.bytes_per_second / 1024).toFixed(1) + ' KB/s' : '0 KB/s';
  const anomalyRate = stats.anomaly_rate !== undefined ? (stats.anomaly_rate * 100).toFixed(1) + '%' : '0.0%';

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg">
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
        Network Flow Telemetry
      </h3>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/30">
          <div className="text-xs text-slate-400">Active Flows</div>
          <div className="text-lg font-bold text-slate-100">{activeFlows}</div>
        </div>
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/30">
          <div className="text-xs text-slate-400">Throughput</div>
          <div className="text-lg font-bold text-slate-100">{pps} pps</div>
        </div>
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/30">
          <div className="text-xs text-slate-400">Bandwidth</div>
          <div className="text-lg font-bold text-slate-100">{bps}</div>
        </div>
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/30">
          <div className="text-xs text-slate-400">Anomaly Rate</div>
          <div className="text-lg font-bold text-amber-400">{anomalyRate}</div>
        </div>
      </div>
    </div>
  );
}
