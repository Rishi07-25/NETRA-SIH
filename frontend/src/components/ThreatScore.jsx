import React from 'react';

/**
 * ThreatScore Component
 * Visual gauge displaying current composite threat score (0-100) and threat severity.
 */
export default function ThreatScore({ score = 0, level = 'LOW' }) {
  const getBadgeColor = (lvl) => {
    switch (lvl) {
      case 'CRITICAL':
      case 'HIGH':
        return 'text-red-400 border-red-500/30 bg-red-500/10';
      case 'MEDIUM':
        return 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10';
      case 'LOW':
      default:
        return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
    }
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg flex flex-col justify-between">
      <div className="flex justify-between items-start">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Unified Threat Score
        </h3>
        <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${getBadgeColor(level)}`}>
          {level}
        </span>
      </div>
      <div className="my-4">
        <div className="text-4xl font-bold tracking-tight text-white flex items-baseline gap-2">
          <span>{Math.round(score)}</span>
          <span className="text-slate-500 text-lg font-normal">/ 100</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2.5 mt-3 overflow-hidden">
          <div
            className={`h-2.5 rounded-full transition-all duration-500 ${
              score > 65 ? 'bg-red-500' : score > 35 ? 'bg-yellow-500' : 'bg-emerald-500'
            }`}
            style={{ width: `${Math.min(Math.max(score, 5), 100)}%` }}
          ></div>
        </div>
      </div>
      <p className="text-xs text-slate-400">Real-time fusion of anomaly signals & classification</p>
    </div>
  );
}
