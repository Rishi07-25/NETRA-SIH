import React from 'react';

/**
 * EventFeed Component
 * Real-time alert stream for early warnings and threshold alerts.
 */
export default function EventFeed({ events = [] }) {
  const getSeverityBadge = (sev) => {
    switch (sev) {
      case 'CRITICAL':
      case 'HIGH':
        return 'bg-red-500/20 text-red-400 border border-red-500/30';
      case 'MEDIUM':
        return 'bg-amber-500/20 text-amber-400 border border-amber-500/30';
      case 'LOW':
      default:
        return 'bg-blue-500/20 text-blue-400 border border-blue-500/30';
    }
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg flex flex-col justify-between">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Early-Warning Alert Feed
        </h3>
        <span className="text-xs text-slate-400">{events.length} Events</span>
      </div>
      <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
        {events.length === 0 ? (
          <p className="text-sm text-slate-500 py-4 text-center">Awaiting telemetry alerts...</p>
        ) : (
          events.map((evt, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-slate-900/60 border border-slate-700/40 hover:border-slate-600/60 transition-all"
            >
              <div className="flex justify-between items-start gap-2 mb-1">
                <span className="font-semibold text-sm text-slate-100">{evt.title}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium ${getSeverityBadge(evt.severity)}`}>
                  {evt.severity}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed mb-1.5">{evt.description}</p>
              <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono">
                <span>{evt.event_id || `EVT-${idx + 1}`}</span>
                <span>{evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : 'Live'}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
