"""Demonstration runner executing the scripted reconnaissance-to-escalation sequence.

Executes the four scenario steps from demo/demo-scenario.md:
1. Baseline normal traffic (low threat score, low attack probability)
2. Precursor reconnaissance injection (anomalies flagged, Reconnaissance classified)
3. Risk horizon forecasting (surging probability, ESCALATING trajectory, early warning)
4. Full-scale brute force execution (high threat score, Brute Force classified)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from backend.services.anomaly_service import AnomalyService
from backend.services.forecasting_service import ForecastingService
from backend.services.prediction_service import PredictionService
from backend.services.traffic_service import TrafficService
from network.traffic_generator.brute_force_simulation import simulate_brute_force
from network.traffic_generator.normal_traffic import generate_normal_traffic
from network.traffic_generator.scan_simulation import simulate_port_scan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NETRA-Demo")


def run_demo_pipeline() -> Dict[str, Any]:
    """Execute the end-to-end P2 demonstration pipeline."""
    logger.info("==================================================")
    logger.info("NETRA — Network Threat Early-warning & Risk Analytics")
    logger.info("P2 Demonstration: Reconnaissance to Escalation")
    logger.info("==================================================")

    pred_service = PredictionService()
    anomaly_service = AnomalyService()
    forecast_service = ForecastingService()
    traffic_service = TrafficService()

    demo_results = {}

    # STEP 1: Baseline Network Health (T = 0m)
    logger.info("\n--- STEP 1: Baseline Network Health (T = 0m) ---")
    normal_flows = generate_normal_traffic(target_host="10.0.0.1", flow_count=5, random_seed=42)
    step1_sample = normal_flows[0]
    
    # Feature vector for ML
    features_step1 = {
        "dst_port": float(step1_sample["dst_port"]),
        "protocol": float(step1_sample["protocol"]),
        "flow_duration": float(step1_sample["flow_duration"]),
        "tot_fwd_pkts": float(step1_sample["tot_fwd_pkts"]),
        "tot_bwd_pkts": float(step1_sample["tot_bwd_pkts"]),
        "tot_fwd_bytes": float(step1_sample["tot_fwd_bytes"]),
        "tot_bwd_bytes": float(step1_sample["tot_bwd_bytes"]),
        "flow_pkt_rate": float(step1_sample["flow_pkt_rate"]),
        "flow_byte_rate": float(step1_sample["flow_byte_rate"]),
        "syn_flag_cnt": float(step1_sample["syn_flag_cnt"]),
        "rst_flag_cnt": float(step1_sample["rst_flag_cnt"]),
        "ack_flag_cnt": float(step1_sample["ack_flag_cnt"]),
        "fwd_bwd_pkt_ratio": float(step1_sample["tot_fwd_pkts"] / max(step1_sample["tot_bwd_pkts"], 1)),
        "fwd_bwd_byte_ratio": float(step1_sample["tot_fwd_bytes"] / max(step1_sample["tot_bwd_bytes"], 1)),
        "avg_pkt_size": float((step1_sample["tot_fwd_bytes"] + step1_sample["tot_bwd_bytes"]) / max(step1_sample["tot_fwd_pkts"] + step1_sample["tot_bwd_pkts"], 1)),
    }
    
    pred1 = pred_service.classify_threat(features_step1)
    anom1 = anomaly_service.evaluate_anomaly(features_step1)
    logger.info(f"Traffic: Normal | Classification: {pred1['attack_class']} (Conf: {pred1['confidence']:.2f}) | Anomaly: {anom1['is_anomalous']} (Score: {anom1['anomaly_score']:.2f})")
    demo_results["step1"] = {"flows": len(normal_flows), "prediction": pred1, "anomaly": anom1}

    # STEP 2: Precursor Reconnaissance Injected (T = 2m)
    logger.info("\n--- STEP 2: Precursor Reconnaissance Injected (T = 2m) ---")
    scan_flows = simulate_port_scan(target_subnet="10.0.0.1", intensity="stealth", random_seed=42)
    step2_sample = scan_flows[0]
    features_step2 = {
        "dst_port": float(step2_sample["dst_port"]),
        "protocol": float(step2_sample["protocol"]),
        "flow_duration": float(step2_sample["flow_duration"]),
        "tot_fwd_pkts": float(step2_sample["tot_fwd_pkts"]),
        "tot_bwd_pkts": float(step2_sample["tot_bwd_pkts"]),
        "tot_fwd_bytes": float(step2_sample["tot_fwd_bytes"]),
        "tot_bwd_bytes": float(step2_sample["tot_bwd_bytes"]),
        "flow_pkt_rate": float(step2_sample["flow_pkt_rate"]),
        "flow_byte_rate": float(step2_sample["flow_byte_rate"]),
        "syn_flag_cnt": float(step2_sample["syn_flag_cnt"]),
        "rst_flag_cnt": float(step2_sample["rst_flag_cnt"]),
        "ack_flag_cnt": float(step2_sample["ack_flag_cnt"]),
        "fwd_bwd_pkt_ratio": float(step2_sample["tot_fwd_pkts"] / max(step2_sample["tot_bwd_pkts"], 1)),
        "fwd_bwd_byte_ratio": float(step2_sample["tot_fwd_bytes"] / max(step2_sample["tot_bwd_bytes"], 1)),
        "avg_pkt_size": 40.0,
    }
    pred2 = pred_service.classify_threat(features_step2)
    anom2 = anomaly_service.evaluate_anomaly(features_step2)
    threat2 = pred_service.assess_full_threat(features_step2)
    logger.info(f"Traffic: Port Scan | Classification: {pred2['attack_class']} (Conf: {pred2['confidence']:.2f}) | Threat Level: {threat2['threat_level']}")
    demo_results["step2"] = {"flows": len(scan_flows), "prediction": pred2, "anomaly": anom2, "threat_level": threat2["threat_level"]}

    # STEP 3: Risk Horizon Forecasting (T = 4m)
    logger.info("\n--- STEP 3: Risk Horizon Forecasting (T = 4m) ---")
    # Simulate escalating window sequence
    escalating_windows = [
        {"flow_count": 10 + i * 8, "tot_fwd_pkts": 40 + i * 30, "tot_bwd_pkts": 35 + i * 15, "tot_fwd_bytes": 4000 + i * 3000, "tot_bwd_bytes": 8000 + i * 3000, "avg_pkt_size": 160 + i * 10, "flow_duration": 50000 + i * 5000, "flow_pkt_rate": 110 + i * 25, "flow_byte_rate": 14000 + i * 3000, "syn_flag_cnt": 1 + i * 4, "rst_flag_cnt": i, "ack_flag_cnt": 35 + i * 10, "fwd_bwd_pkt_ratio": 1.1 + i * 0.15, "fwd_bwd_byte_ratio": 0.5 + i * 0.1, "unique_src_ports": 5 + i * 4, "unique_dst_ports": 2 + i * 3, "dst_port_entropy": 1.1 + i * 0.3, "tcp_ratio": 0.9, "udp_ratio": 0.1}
        for i in range(5)
    ]
    forecast_res = forecast_service.generate_risk_forecast(escalating_windows)
    logger.info(f"Threat Score: {forecast_res['current_threat_score']:.1f} | Trajectory: {forecast_res['trajectory']}")
    for fc in forecast_res["forecasts"]:
        logger.info(f"  Horizon +{fc['horizon']}: P(Attack) = {fc['attack_probability'] * 100:.1f}%, Risk Score = {fc['risk_score']:.1f}, Stage = {fc['predicted_stage']}")
    demo_results["step3"] = forecast_res

    # STEP 4: Full-Scale Brute Force Attempted (T = 8m)
    logger.info("\n--- STEP 4: Full-Scale Brute Force Attempted (T = 8m) ---")
    bf_flows = simulate_brute_force(target_service="10.0.0.1", attempts=10, random_seed=42)
    step4_sample = bf_flows[0]
    features_step4 = {
        "dst_port": float(step4_sample["dst_port"]),
        "protocol": float(step4_sample["protocol"]),
        "flow_duration": float(step4_sample["flow_duration"]),
        "tot_fwd_pkts": float(step4_sample["tot_fwd_pkts"]),
        "tot_bwd_pkts": float(step4_sample["tot_bwd_pkts"]),
        "tot_fwd_bytes": float(step4_sample["tot_fwd_bytes"]),
        "tot_bwd_bytes": float(step4_sample["tot_bwd_bytes"]),
        "flow_pkt_rate": float(step4_sample["flow_pkt_rate"]),
        "flow_byte_rate": float(step4_sample["flow_byte_rate"]),
        "syn_flag_cnt": float(step4_sample["syn_flag_cnt"]),
        "rst_flag_cnt": float(step4_sample["rst_flag_cnt"]),
        "ack_flag_cnt": float(step4_sample["ack_flag_cnt"]),
        "fwd_bwd_pkt_ratio": float(step4_sample["tot_fwd_pkts"] / max(step4_sample["tot_bwd_pkts"], 1)),
        "fwd_bwd_byte_ratio": float(step4_sample["tot_fwd_bytes"] / max(step4_sample["tot_bwd_bytes"], 1)),
        "avg_pkt_size": float((step4_sample["tot_fwd_bytes"] + step4_sample["tot_bwd_bytes"]) / max(step4_sample["tot_fwd_pkts"] + step4_sample["tot_bwd_pkts"], 1)),
    }
    pred4 = pred_service.classify_threat(features_step4)
    threat4 = pred_service.assess_full_threat(features_step4)
    logger.info(f"Traffic: Brute Force | Classification: {pred4['attack_class']} (Conf: {pred4['confidence']:.2f}) | Threat Level: {threat4['threat_level']}")
    demo_results["step4"] = {"flows": len(bf_flows), "prediction": pred4, "threat_level": threat4["threat_level"]}

    logger.info("\n==================================================")
    logger.info("P2 Demonstration Execution Completed Successfully")
    logger.info("==================================================")
    return demo_results


if __name__ == "__main__":
    run_demo_pipeline()
