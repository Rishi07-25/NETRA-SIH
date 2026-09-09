import React from 'react';

/**
 * AttackTable Component
 * Tabular representation of flagged attacks and confidence scores.
 */
export default function AttackTable({ attacks = [] }) {
  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Active Threat Classifications
        </h3>
        <span className="text-xs text-slate-400">Total Classified: {attacks.length}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-slate-900/60 text-xs text-slate-400 uppercase">
            <tr>
              <th className="py-2.5 px-3">Timestamp</th>
              <th className="py-2.5 px-3">Attack Family</th>
              <th className="py-2.5 px-3">Confidence</th>
              <th className="py-2.5 px-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {attacks.length === 0 ? (
              <tr>
                <td colSpan="4" className="py-4 text-center text-slate-500">
                  No active threat records
                </td>
              </tr>
            ) : (
              attacks.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-700/30 transition-colors">
                  <td className="py-2.5 px-3 font-mono text-xs text-slate-400">
                    {row.timestamp || 'Just now'}
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-white">
                    {row.attack_class}
                  </td>
                  <td className="py-2.5 px-3">
                    <span className="font-mono">{Math.round((row.confidence || 0) * 100)}%</span>
                  </td>
                  <td className="py-2.5 px-3">
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                      row.attack_class === 'BENIGN'
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-red-500/20 text-red-300'
                    }`}>
                      {row.attack_class === 'BENIGN' ? 'Normal' : 'Mitigation Active'}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
