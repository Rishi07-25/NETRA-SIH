import React from 'react';

/**
 * ThreatTimeline Component
 * Chronological trajectory tracking threat score evolution and trend line.
 */
export default function ThreatTimeline({ history = [], trajectory = 'STABLE' }) {
  const getTrajectoryBadge = (traj) => {
    switch (traj) {
      case 'ESCALATING':
        return 'bg-red-500/20 text-red-300 border-red-500/40';
      case 'DECLINING':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'STABLE':
      default:
        return 'bg-slate-700 text-slate-300 border-slate-600';
    }
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg my-6">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Temporal Threat Trajectory
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">Sequential Killchain Projection Over Time</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Trajectory:</span>
          <span className={`text-xs px-2.5 py-1 rounded-md border font-semibold tracking-wide ${getTrajectoryBadge(trajectory)}`}>
            {trajectory}
          </span>
        </div>
      </div>
      
      {/* Simple responsive CSS bar chart representing historical window progression */}
      <div className="h-32 flex items-end gap-2 pt-4 px-2 bg-slate-900/40 rounded-lg border border-slate-700/30">
        {history.length === 0 ? (
          <div className="w-full text-center text-slate-500 text-xs py-8">
            Awaiting rolling sequence telemetry...
          </div>
        ) : (
          history.map((val, idx) => (
            <div key={idx} className="flex-1 flex flex-col items-center gap-1 group">
              <span className="text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
                {Math.round(val)}
              </span>
              <div
                className={`w-full rounded-t transition-all duration-300 ${
                  val > 65 ? 'bg-red-500' : val > 35 ? 'bg-amber-400' : 'bg-emerald-400'
                }`}
                style={{ height: `${Math.max(val, 8)}%` }}
              ></div>
              <span className="text-[9px] text-slate-500 font-mono">T-{history.length - idx}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
