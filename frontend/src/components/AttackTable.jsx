import React from 'react';

/**
 * AttackTable Component
 * Tabular representation of flagged attacks and confidence scores.
 * Placeholder for future implementation.
 */
export default function AttackTable({ attacks = [] }) {
  return (
    <div className="attack-table-container">
      <h3>Active Threat Classifications</h3>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Attack Family</th>
            <th>Confidence</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colSpan="4">No active threat records</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
