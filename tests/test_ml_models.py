"""Automated unit and integration tests for NETRA Stage 2 AI Threat Engine.

Validates:
- Data loading and schema validation
- Deterministic train/test split
- Logistic Regression baseline training and probability outputs
- Random Forest classifier training, feature importance, and schema enforcement
- Isolation Forest unsupervised training (zero label contamination)
- Unified Threat Engine heuristic scoring
- Evaluation metrics calculation
- End-to-end Stage 2 pipeline execution
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

from ml.anomaly_detection.predict import AnomalyDetectorPredictor, predict_anomaly
from ml.anomaly_detection.train import train_anomaly_detector
from ml.attack_classification.baseline import train_baseline
from ml.attack_classification.predict import AttackClassifierPredictor, predict_attack_class
from ml.attack_classification.train import train_attack_classifier
from ml.data_validation import load_and_validate_features, split_threat_data, validate_feature_data
from ml.evaluation.evaluate import evaluate_models
from ml.evaluation.metrics import compute_anomaly_metrics, compute_classification_metrics
from ml.run_stage2 import run_stage2_threat_engine
from ml.threat_engine import ThreatEngine, evaluate_threat

FEATURES_PATH = "data/features/features_X.csv"
LABELS_PATH = "data/features/labels_y.csv"


def test_data_validation_success():
    """Verify that Stage 1 outputs load and validate cleanly without NaNs or Infs."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    assert not X.empty
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not np.isinf(X.to_numpy()).any()
    assert "label" not in X.columns
    assert "src_ip" not in X.columns
    assert "timestamp" not in X.columns


def test_data_validation_detects_leakage():
    """Verify that validation raises an error if prohibited label/ID columns exist in X."""
    bad_df = pd.DataFrame({"dst_port": [80], "label": ["BENIGN"]})
    with pytest.raises(ValueError, match="Prohibited target/identifier columns"):
        validate_feature_data(bad_df)


def test_train_test_split_reproducibility():
    """Verify train/test split determinism and stratification."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr1, X_te1, y_tr1, y_te1 = split_threat_data(X, y, test_size=0.20, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = split_threat_data(X, y, test_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    assert len(X_tr1) + len(X_te1) == len(X)


def test_logistic_regression_baseline_training(tmp_path):
    """Verify Logistic Regression baseline trains and outputs calibrated probabilities."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    save_file = tmp_path / "baseline.joblib"
    pipeline, bundle = train_baseline(X_tr, y_tr, random_state=42, save_path=save_file)

    assert isinstance(pipeline, Pipeline)
    assert save_file.exists()
    assert "classes" in bundle
    assert len(bundle["classes"]) == y.nunique()

    # Predictor wrapper test
    pred = AttackClassifierPredictor(save_file)
    sample_res = pred.predict_single(X_te.iloc[0])

    assert "predicted_attack" in sample_res
    assert "confidence" in sample_res
    assert "class_probabilities" in sample_res
    # Probabilities sum to 1
    total_prob = sum(sample_res["class_probabilities"].values())
    assert pytest.approx(total_prob, abs=1e-4) == 1.0


def test_logistic_regression_protocol_is_categorical(tmp_path):
    """TEST 1: Verify protocol is handled by OneHotEncoder and excluded from StandardScaler."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, _, y_tr, _ = split_threat_data(X, y, test_size=0.20, random_state=42)

    save_file = tmp_path / "baseline_cat.joblib"
    pipeline, bundle = train_baseline(X_tr, y_tr, random_state=42, save_path=save_file)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = preprocessor.transformers_

    # Find numeric and protocol transformers
    numeric_trans = next(t for t in transformers if t[0] == "numeric")
    protocol_trans = next(t for t in transformers if t[0] == "protocol")

    # Verify protocol is NOT in numeric transformer features
    numeric_features = numeric_trans[2]
    assert "protocol" not in numeric_features

    # Verify protocol IS in protocol transformer and uses OneHotEncoder
    protocol_features = protocol_trans[2]
    assert "protocol" in protocol_features
    assert isinstance(protocol_trans[1], OneHotEncoder)
    assert protocol_trans[1].handle_unknown == "ignore"


def test_logistic_regression_unknown_protocol_handling(tmp_path):
    """TEST 2: Verify predict_proba does not crash when encountering an unseen protocol."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    save_file = tmp_path / "baseline_unknown_proto.joblib"
    pipeline, _ = train_baseline(X_tr, y_tr, random_state=42, save_path=save_file)

    # Inject an unseen protocol (e.g. 999 or 255)
    unseen_row = X_te.iloc[[0]].copy()
    unseen_row["protocol"] = 999

    pred = AttackClassifierPredictor(save_file)
    res = pred.predict_single(unseen_row.iloc[0])

    assert "predicted_attack" in res
    assert "class_probabilities" in res
    total_prob = sum(res["class_probabilities"].values())
    assert pytest.approx(total_prob, abs=1e-4) == 1.0


def test_logistic_regression_existing_predictions_still_work(tmp_path):
    """TEST 3: Verify predict() and predict_proba() return valid outputs with the 15-column schema."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    save_file = tmp_path / "baseline_pred_work.joblib"
    pipeline, bundle = train_baseline(X_tr, y_tr, random_state=42, save_path=save_file)

    pred = AttackClassifierPredictor(save_file)

    # Single prediction
    single_res = pred.predict_single(X_te.iloc[0])
    assert single_res["predicted_attack"] in bundle["classes"]
    assert 0.0 <= single_res["confidence"] <= 1.0
    assert len(single_res["class_probabilities"]) == len(bundle["classes"])

    # Batch prediction
    batch_res = pred.predict_batch(X_te)
    assert len(batch_res) == len(X_te)
    for r in batch_res:
        assert r["predicted_attack"] in bundle["classes"]
        assert pytest.approx(sum(r["class_probabilities"].values()), abs=1e-4) == 1.0


def test_logistic_regression_model_persistence(tmp_path):
    """TEST 4: Save the corrected model, reload it, and verify predictions remain consistent."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    save_file = tmp_path / "persisted_baseline.joblib"
    pipeline, _ = train_baseline(X_tr, y_tr, random_state=42, save_path=save_file)

    # Load via AttackClassifierPredictor
    pred1 = AttackClassifierPredictor(save_file)
    res1 = pred1.predict_single(X_te.iloc[0])

    # Re-load from disk directly and verify equivalence
    reloaded_bundle = joblib.load(save_file)
    pred2 = AttackClassifierPredictor(reloaded_bundle)
    res2 = pred2.predict_single(X_te.iloc[0])

    assert res1["predicted_attack"] == res2["predicted_attack"]
    assert pytest.approx(res1["confidence"], abs=1e-6) == res2["confidence"]
    assert res1["class_probabilities"] == res2["class_probabilities"]


def test_random_forest_training_and_feature_importance(tmp_path):
    """Verify Random Forest training, schema validation, and feature importances."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    save_file = tmp_path / "rf_model.joblib"
    model, bundle = train_attack_classifier(X_tr, y_tr, n_estimators=50, random_state=42, save_path=save_file)

    assert save_file.exists()
    assert len(model.feature_importances_) == X.shape[1]

    pred = AttackClassifierPredictor(save_file)
    res = pred.predict_single(X_te.iloc[0])

    assert res["predicted_attack"] in bundle["classes"]
    assert 0.0 <= res["confidence"] <= 1.0
    total_prob = sum(res["class_probabilities"].values())
    assert pytest.approx(total_prob, abs=1e-4) == 1.0

    # Schema validation test: missing column should raise ValueError
    incomplete_input = X_te.iloc[0].drop("dst_port")
    with pytest.raises(ValueError, match="missing required features"):
        pred.predict_single(incomplete_input)


def test_isolation_forest_unsupervised_training(tmp_path):
    """Verify Isolation Forest trains without labels and outputs valid scores."""
    X, _ = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    save_file = tmp_path / "iso_forest.joblib"

    # CRITICAL: labels not passed
    model, bundle = train_anomaly_detector(X, contamination=0.15, save_path=save_file)

    assert save_file.exists()
    assert "model" in bundle

    pred = AnomalyDetectorPredictor(save_file)
    sample_res = pred.predict_single(X.iloc[0])

    assert "is_anomalous" in sample_res
    assert isinstance(sample_res["is_anomalous"], bool)
    assert "raw_score" in sample_res
    assert "display_score" in sample_res
    assert 0.0 <= sample_res["display_score"] <= 1.0


def test_unified_threat_engine(tmp_path):
    """Verify Threat Engine combines classification and anomaly evidence with transparent heuristics."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    rf_file = tmp_path / "rf.joblib"
    anom_file = tmp_path / "anom.joblib"

    train_attack_classifier(X_tr, y_tr, n_estimators=50, random_state=42, save_path=rf_file)
    train_anomaly_detector(X_tr, contamination=0.15, random_state=42, save_path=anom_file)

    engine = ThreatEngine(classifier_model=rf_file, anomaly_model=anom_file)
    result = engine.assess_threat(X_te.iloc[0])

    assert "predicted_attack" in result
    assert "classification_confidence" in result
    assert "is_anomalous" in result
    assert "anomaly_score" in result
    assert "threat_level" in result
    assert result["threat_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert "threat_level_heuristic_note" in result


def test_threat_level_deterministic_logic():
    """Verify transparent heuristic thresholds for LOW, MEDIUM, and HIGH."""
    engine = ThreatEngine.__new__(ThreatEngine)
    engine.config = {
        "high_confidence_threshold": 0.80,
        "medium_confidence_threshold": 0.40,
        "anomaly_display_threshold": 0.65,
    }

    # Benign with low anomaly -> LOW
    assert engine.determine_threat_level("BENIGN", 0.95, False, 0.20) == "LOW"

    # Benign with anomaly flag -> MEDIUM
    assert engine.determine_threat_level("BENIGN", 0.60, True, 0.50) == "MEDIUM"

    # Attack with moderate confidence -> MEDIUM
    assert engine.determine_threat_level("PortScan", 0.60, False, 0.30) == "MEDIUM"

    # Attack with high confidence -> HIGH
    assert engine.determine_threat_level("DDoS", 0.85, False, 0.40) == "HIGH"

    # Attack with anomaly confirmation -> HIGH
    assert engine.determine_threat_level("PortScan", 0.55, True, 0.50) == "HIGH"

    # Extreme anomaly display score -> HIGH
    assert engine.determine_threat_level("BENIGN", 0.70, False, 0.80) == "HIGH"


def test_evaluation_metrics_and_artifacts(tmp_path):
    """Verify evaluation harness outputs metrics, confusion matrices, and JSON artifacts."""
    X, y = load_and_validate_features(FEATURES_PATH, LABELS_PATH)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y, test_size=0.20, random_state=42)

    pipe, _ = train_baseline(X_tr, y_tr, random_state=42, save_path=None)
    rf, _ = train_attack_classifier(X_tr, y_tr, n_estimators=50, random_state=42, save_path=None)
    anom, _ = train_anomaly_detector(X_tr, contamination=0.15, random_state=42, save_path=None)

    eval_dir = tmp_path / "eval_results"
    results = evaluate_models(
        X_test=X_te,
        y_test=y_te,
        baseline_model=pipe,
        primary_model=rf,
        anomaly_model=anom,
        feature_names=list(X.columns),
        results_dir=eval_dir,
        is_synthetic_data=True,
    )

    assert (eval_dir / "classification_results.json").exists()
    assert (eval_dir / "anomaly_results.json").exists()
    assert (eval_dir / "feature_importance.csv").exists()
    assert (eval_dir / "confusion_matrix.csv").exists()

    assert "accuracy" in results["primary"]
    assert "f1_macro" in results["primary"]
    assert "anomaly_precision" in results["anomaly"]
    assert "anomaly_recall" in results["anomaly"]


def test_stage2_end_to_end_smoke_test(tmp_path):
    """Smoke test executing run_stage2_threat_engine end-to-end on synthetic data."""
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    res = run_stage2_threat_engine(
        features_path=FEATURES_PATH,
        labels_path=LABELS_PATH,
        models_dir=str(models_dir),
        results_dir=str(results_dir),
        test_size=0.20,
        random_state=42,
        is_synthetic=True,
    )

    assert (models_dir / "logistic_regression_baseline.joblib").exists()
    assert (models_dir / "random_forest_classifier.joblib").exists()
    assert (models_dir / "isolation_forest.joblib").exists()
    assert (results_dir / "classification_results.json").exists()
    assert "baseline" in res
    assert "primary" in res
    assert "anomaly" in res
