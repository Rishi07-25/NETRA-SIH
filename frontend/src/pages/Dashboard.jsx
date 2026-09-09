import React, { useEffect, useState } from 'react';
import ThreatScore from '../components/ThreatScore';
import ForecastCard from '../components/ForecastCard';
import NetworkStats from '../components/NetworkStats';
import ThreatTimeline from '../components/ThreatTimeline';
import AttackTable from '../components/AttackTable';
import EventFeed from '../components/EventFeed';
import { fetchHealth, fetchNetworkStats, fetchEvents, postForecast, postPrediction } from '../services/api';

/**
 * Dashboard Page
 * Primary Security Operations Center (SOC) operational dashboard.
 * Connects directly to NETRA FastAPI backend services.
 */
export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [backendHealth, setBackendHealth] = useState(null);
  const [networkStats, setNetworkStats] = useState({
    active_flows: 0,
    packets_per_second: 0,
    bytes_per_second: 0,
    anomaly_rate: 0,
  });
  const [threatScore, setThreatScore] = useState(15);
  const [threatLevel, setThreatLevel] = useState('LOW');
  const [forecasts, setForecasts] = useState([
    { horizon: '1m', risk_score: 12, attack_probability: 0.08, predicted_stage: 'Normal' },
    { horizon: '5m', risk_score: 18, attack_probability: 0.14, predicted_stage: 'Normal' },
    { horizon: '15m', risk_score: 25, attack_probability: 0.20, predicted_stage: 'Reconnaissance' },
  ]);
  const [trajectory, setTrajectory] = useState('STABLE');
  const [timelineHistory, setTimelineHistory] = useState([12, 14, 15, 18, 22, 28, 32]);
  const [attacks, setAttacks] = useState([]);
  const [events, setEvents] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');

  // Sample sequence window fixture for live forecasting demonstration
  const sampleHistoryWindows = [
    { flow_count: 10, tot_fwd_pkts: 40, tot_bwd_pkts: 35, tot_fwd_bytes: 4000, tot_bwd_bytes: 8000, avg_pkt_size: 160, flow_duration: 50000, flow_pkt_rate: 110, flow_byte_rate: 14000, syn_flag_cnt: 1, rst_flag_cnt: 0, ack_flag_cnt: 35, fwd_bwd_pkt_ratio: 1.1, fwd_bwd_byte_ratio: 0.5, unique_src_ports: 5, unique_dst_ports: 2, dst_port_entropy: 1.1, tcp_ratio: 0.9, udp_ratio: 0.1 },
    { flow_count: 14, tot_fwd_pkts: 55, tot_bwd_pkts: 45, tot_fwd_bytes: 5200, tot_bwd_bytes: 9500, avg_pkt_size: 165, flow_duration: 55000, flow_pkt_rate: 125, flow_byte_rate: 15500, syn_flag_cnt: 2, rst_flag_cnt: 0, ack_flag_cnt: 42, fwd_bwd_pkt_ratio: 1.2, fwd_bwd_byte_ratio: 0.54, unique_src_ports: 6, unique_dst_ports: 3, dst_port_entropy: 1.2, tcp_ratio: 0.88, udp_ratio: 0.12 },
    { flow_count: 18, tot_fwd_pkts: 70, tot_bwd_pkts: 55, tot_fwd_bytes: 6800, tot_bwd_bytes: 11000, avg_pkt_size: 170, flow_duration: 60000, flow_pkt_rate: 140, flow_byte_rate: 17000, syn_flag_cnt: 4, rst_flag_cnt: 1, ack_flag_cnt: 50, fwd_bwd_pkt_ratio: 1.27, fwd_bwd_byte_ratio: 0.61, unique_src_ports: 9, unique_dst_ports: 5, dst_port_entropy: 1.45, tcp_ratio: 0.92, udp_ratio: 0.08 },
    { flow_count: 24, tot_fwd_pkts: 95, tot_bwd_pkts: 65, tot_fwd_bytes: 8900, tot_bwd_bytes: 13000, avg_pkt_size: 175, flow_duration: 65000, flow_pkt_rate: 160, flow_byte_rate: 19500, syn_flag_cnt: 8, rst_flag_cnt: 3, ack_flag_cnt: 60, fwd_bwd_pkt_ratio: 1.46, fwd_bwd_byte_ratio: 0.68, unique_src_ports: 14, unique_dst_ports: 8, dst_port_entropy: 1.82, tcp_ratio: 0.95, udp_ratio: 0.05 },
    { flow_count: 32, tot_fwd_pkts: 130, tot_bwd_pkts: 80, tot_fwd_bytes: 12000, tot_bwd_bytes: 16000, avg_pkt_size: 180, flow_duration: 70000, flow_pkt_rate: 190, flow_byte_rate: 23000, syn_flag_cnt: 15, rst_flag_cnt: 6, ack_flag_cnt: 75, fwd_bwd_pkt_ratio: 1.62, fwd_bwd_byte_ratio: 0.75, unique_src_ports: 20, unique_dst_ports: 12, dst_port_entropy: 2.15, tcp_ratio: 0.98, udp_ratio: 0.02 },
  ];

  useEffect(() => {
    async function initDashboard() {
      try {
        setLoading(true);
        // 1. Fetch Backend Health
        const health = await fetchHealth().catch(() => null);
        setBackendHealth(health);

        // 2. Fetch Network Stats
        const stats = await fetchNetworkStats().catch(() => null);
        if (stats) setNetworkStats(stats);

        // 3. Fetch Events Feed
        const evtsData = await fetchEvents().catch(() => null);
        if (evtsData && evtsData.events) setEvents(evtsData.events);

        // 4. Request Risk Forecast
        const forecastRes = await postForecast(sampleHistoryWindows).catch(() => null);
        if (forecastRes) {
          setThreatScore(forecastRes.current_threat_score || 25);
          if (forecastRes.forecasts) setForecasts(forecastRes.forecasts);
          if (forecastRes.trajectory) setTrajectory(forecastRes.trajectory);
          setThreatLevel(forecastRes.current_threat_score > 60 ? 'HIGH' : forecastRes.current_threat_score > 30 ? 'MEDIUM' : 'LOW');
        }

        // 5. Test Live Classification
        const predRes = await postPrediction({
          dst_port: 22,
          protocol: 6,
          flow_duration: 95000,
          tot_fwd_pkts: 8,
          tot_bwd_pkts: 7,
          tot_fwd_bytes: 650,
          tot_bwd_bytes: 820,
          flow_pkt_rate: 157.8,
          flow_byte_rate: 15473.0,
          syn_flag_cnt: 1,
          rst_flag_cnt: 0,
          ack_flag_cnt: 7,
          fwd_bwd_pkt_ratio: 1.14,
          fwd_bwd_byte_ratio: 0.79,
          avg_pkt_size: 98.0,
        }).catch(() => null);

        if (predRes) {
          setAttacks([
            {
              timestamp: '10:01:05 UTC',
              attack_class: predRes.attack_class,
              confidence: predRes.confidence,
            },
            {
              timestamp: '10:00:45 UTC',
              attack_class: 'Reconnaissance',
              confidence: 0.89,
            },
          ]);
        }
      } catch (err) {
        setErrorMsg('Failed to connect to NETRA backend services.');
      } finally {
        setLoading(false);
      }
    }

    initDashboard();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Top Banner & Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 mb-8 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
              NETRA SOC DASHBOARD
            </h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 font-mono border border-blue-500/30">
              SIH26153
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Network Threat Early-warning & Risk Analytics — Multi-Horizon Attack Forecasting
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
            <span className={`h-2.5 w-2.5 rounded-full ${backendHealth ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`}></span>
            <span className="text-slate-300">{backendHealth ? 'API CONNECTED' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* Main Top Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <ThreatScore score={threatScore} level={threatLevel} />
        {forecasts.length > 0 ? (
          <ForecastCard
            horizon={forecasts[1]?.horizon || '5m'}
            risk={forecasts[1]?.risk_score || 0}
            probability={forecasts[1]?.attack_probability || 0}
            stage={forecasts[1]?.predicted_stage || 'Normal'}
          />
        ) : (
          <ForecastCard horizon="5m" risk={0} probability={0} />
        )}
        <NetworkStats stats={networkStats} />
      </div>

      {/* Temporal Killchain Progression Timeline */}
      <ThreatTimeline history={timelineHistory} trajectory={trajectory} />

      {/* Multi-Horizon Projections Grid */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Forward Time-Horizon Attack Probability
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {forecasts.map((fc, i) => (
            <ForecastCard
              key={i}
              horizon={fc.horizon}
              risk={fc.risk_score}
              probability={fc.attack_probability}
              stage={fc.predicted_stage}
            />
          ))}
        </div>
      </div>

      {/* Active Incident Classifications & Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AttackTable attacks={attacks} />
        <EventFeed events={events} />
      </div>
    </div>
  );
}
