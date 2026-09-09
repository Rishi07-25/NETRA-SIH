import React from 'react';

/**
 * ForecastCard Component
 * Displays forecasted threat probability across a forward time horizon.
 */
export default function ForecastCard({ horizon = '5m', risk = 0, probability = 0, stage = 'Normal' }) {
  const probPct = Math.round(probability * 100);
  const isElevated = probPct >= 50;

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg flex flex-col justify-between">
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-sm font-semibold text-slate-300">Horizon: +{horizon}</h4>
        <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono">
          {stage}
        </span>
      </div>
      <div className="my-2">
        <div className="text-3xl font-bold text-white flex items-baseline gap-1">
          <span>{probPct}%</span>
          <span className="text-xs text-slate-400 font-normal">P(Attack)</span>
        </div>
        <div className="flex items-center gap-2 mt-2 text-xs text-slate-400">
          <span>Traffic Risk:</span>
          <span className="font-semibold text-slate-200">{Math.round(risk)}/100</span>
        </div>
      </div>
      <div className="mt-3">
        <span className={`text-xs px-2 py-1 rounded inline-block font-medium ${
          isElevated ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 'bg-slate-700/40 text-slate-400'
        }`}>
          {isElevated ? 'Imminent Surge Warning' : 'Normal Fluctuations'}
        </span>
      </div>
    </div>
  );
}
