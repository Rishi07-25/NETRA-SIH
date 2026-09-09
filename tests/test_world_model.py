"""Unit and integration tests for NETRA Stage 3B.1 and Stage 3B.2 Temporal World Model.

Verifies:
1. Model accepts (N, 5, 19) tensor & output shapes are correct across all 5 prediction heads
2. Rejection of invalid input dimensions
3. Forward pass contains no NaN/Inf
4. Binary probabilities and risk predictions bounded in [0, 1]
5. Deterministic initialization with random seed
6. Training loss decreases on a controlled fixture
7. Checkpoint save/load preserves predictions exactly (v2 checkpoint)
8. Target label encoders informative error on unknown label
9. Multi-head evaluation metrics calculation
10. Stage 3A contract preservation
11. State head output shape = (N, 19) and finite outputs
12. Multi-task loss includes state loss and respects configurable weights (default risk=2.0)
13. Benign attribution masking
14. Inverse scaling of predicted states
15. Autoregressive rollout shape = (N, K, 19) and recursive state feedback without future ground truth
16. Rollout K=1 equivalence with single-step state prediction
17. Rollout input validation and safety (K >= 1, finite outputs, valid shape)
18. Deterministic rollout reproducibility
19. Persistence baseline calculation
20. Horizon degradation metric table calculation
21. Latent embedding extraction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from ml.forecasting.predict_world_model import WorldModelPredictor
from ml.forecasting.rollout import (
    StateRolloutEngine,
    evaluate_horizon_degradation,
    persistence_baseline,
    validate_rollout_input,
)
from ml.forecasting.schema import (
    STATE_PREDICTIVE_FEATURES,
    TAXONOMY_DISCLAIMER,
    VALID_ATTACK_STAGES,
)
from ml.forecasting.train_world_model import (
    compute_multitask_loss,
    create_label_encoders,
    encode_targets,
    evaluate_world_model,
    set_seed,
    train_world_model,
)
from ml.forecasting.world_model import CANONICAL_ATTACK_TYPES, TemporalWorldModel


@pytest.fixture
def synthetic_sequence_batch() -> torch.Tensor:
    """Deterministic (N=4, L=5, D=19) input tensor."""
    set_seed(42)
    return torch.randn(4, 5, 19, dtype=torch.float32)


@pytest.fixture
def default_world_model() -> TemporalWorldModel:
    """Standard TemporalWorldModel instance."""
    set_seed(42)
    return TemporalWorldModel(
        input_size=19,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )


# ---------------------------------------------------------------------------
# 1. Model Accepts (N, 5, 19) & Output Shapes Are Correct (5 Heads)
# ---------------------------------------------------------------------------
def test_model_accepts_correct_shape_and_output_shapes(default_world_model, synthetic_sequence_batch):
    default_world_model.eval()
    with torch.no_grad():
        outputs = default_world_model(synthetic_sequence_batch)

    assert "embedding" in outputs
    assert "binary_logits" in outputs
    assert "type_logits" in outputs
    assert "stage_logits" in outputs
    assert "risk_score" in outputs
    assert "next_state" in outputs

    assert outputs["embedding"].shape == (4, 64)
    assert outputs["binary_logits"].shape == (4, 2)
    assert outputs["type_logits"].shape == (4, len(CANONICAL_ATTACK_TYPES))
    assert outputs["stage_logits"].shape == (4, len(VALID_ATTACK_STAGES))
    assert outputs["risk_score"].shape == (4,)
    assert outputs["next_state"].shape == (4, 19)


# ---------------------------------------------------------------------------
# 2. Rejection of Invalid Input Dimensions
# ---------------------------------------------------------------------------
def test_model_rejects_invalid_dimensions(default_world_model):
    # 2D instead of 3D
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        default_world_model(torch.randn(4, 19))

    # Feature dimension mismatch (18 instead of 19)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        default_world_model(torch.randn(4, 5, 18))


# ---------------------------------------------------------------------------
# 3. Forward Pass Contains No NaN or Inf
# ---------------------------------------------------------------------------
def test_forward_pass_finite_outputs(default_world_model, synthetic_sequence_batch):
    default_world_model.eval()
    with torch.no_grad():
        outputs = default_world_model(synthetic_sequence_batch)

    for k, v in outputs.items():
        assert torch.isfinite(v).all(), f"Output tensor '{k}' contains non-finite values."


# ---------------------------------------------------------------------------
# 4. Binary Probabilities and Risk Predictions Bounded in [0, 1]
# ---------------------------------------------------------------------------
def test_prediction_bounds_and_interface(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    predictor = WorldModelPredictor(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    X = np.random.randn(8, 5, 19).astype(np.float32)
    preds = predictor.predict(X)

    # Future attack probability in [0, 1]
    prob = preds["future_attack_probability"]
    assert (prob >= 0.0).all() and (prob <= 1.0).all()

    # Binary in {0, 1}
    assert set(np.unique(preds["predicted_attack_binary"])).issubset({0, 1})

    # Risk score in [0, 1]
    risk = preds["predicted_risk_score"]
    assert (risk >= 0.0).all() and (risk <= 1.0).all()

    # Next state prediction present and shaped (8, 19)
    assert "predicted_next_state" in preds
    assert preds["predicted_next_state"].shape == (8, 19)

    # Attack types and stages belong to valid vocabularies
    for t_pred in preds["predicted_attack_type"]:
        assert t_pred in CANONICAL_ATTACK_TYPES

    for s_pred in preds["predicted_attack_stage"]:
        assert s_pred in VALID_ATTACK_STAGES

    # Disclaimer present
    assert preds["taxonomy_disclaimer"] == TAXONOMY_DISCLAIMER


# ---------------------------------------------------------------------------
# 5. Deterministic Initialization with Random Seed
# ---------------------------------------------------------------------------
def test_deterministic_initialization():
    set_seed(123)
    m1 = TemporalWorldModel(input_size=19, hidden_size=32, num_layers=1)

    set_seed(123)
    m2 = TemporalWorldModel(input_size=19, hidden_size=32, num_layers=1)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Model parameters initialized non-deterministically."


# ---------------------------------------------------------------------------
# 6. Training Loss Decreases on Controlled Fixture
# ---------------------------------------------------------------------------
def test_training_loss_decreases_over_epochs():
    set_seed(42)
    N = 16
    L = 5
    D = 19
    X_train = np.random.randn(N, L, D).astype(np.float32)
    y_train = pd.DataFrame({
        "future_attack_binary": [0, 1] * (N // 2),
        "future_attack_type": ["BENIGN", "DDoS"] * (N // 2),
        "future_attack_stage": ["Normal", "Impact"] * (N // 2),
        "future_attack_risk_score": [0.0, 0.95] * (N // 2),
    })
    # Add standardized state targets
    for i, feat in enumerate(STATE_PREDICTIVE_FEATURES):
        y_train[f"state_{feat}"] = np.linspace(-1.0, 1.0, N)

    bundle = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": np.empty((0, L, D), dtype=np.float32),
        "y_val": pd.DataFrame(),
        "X_test": np.empty((0, L, D), dtype=np.float32),
        "y_test": pd.DataFrame(),
        "features": list(STATE_PREDICTIVE_FEATURES),
    }

    res = train_world_model(
        dataset_bundle=bundle,
        hidden_size=32,
        num_layers=1,
        epochs=15,
        learning_rate=0.01,
        batch_size=8,
        patience=15,
        seed=42,
    )

    history = res["history"]
    assert len(history) > 1
    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


# ---------------------------------------------------------------------------
# 7. Checkpoint Save/Load Preserves Predictions Exactly (v2 Checkpoint)
# ---------------------------------------------------------------------------
def test_checkpoint_save_load_preserves_predictions(tmp_path):
    set_seed(42)
    N = 8
    L = 5
    D = 19
    X = np.random.randn(N, L, D).astype(np.float32)
    y = pd.DataFrame({
        "future_attack_binary": [1] * N,
        "future_attack_type": ["PortScan"] * N,
        "future_attack_stage": ["Reconnaissance"] * N,
        "future_attack_risk_score": [0.75] * N,
    })

    bundle = {
        "X_train": X,
        "y_train": y,
        "X_val": X,
        "y_val": y,
        "X_test": X,
        "y_test": y,
        "features": list(STATE_PREDICTIVE_FEATURES),
    }

    res = train_world_model(
        dataset_bundle=bundle,
        hidden_size=32,
        num_layers=1,
        epochs=3,
        save_dir=tmp_path,
        seed=42,
    )

    model_original = res["model"]
    type_to_idx = res["checkpoint"]["type_to_idx"]
    stage_to_idx = res["checkpoint"]["stage_to_idx"]

    pred_original = WorldModelPredictor(model_original, type_to_idx, stage_to_idx).predict(X)

    # Load from saved v2 checkpoint
    ckpt_file = tmp_path / "temporal_world_model_v2.pt"
    assert ckpt_file.exists()
    pred_loaded = WorldModelPredictor.load_from_checkpoint(ckpt_file).predict(X)

    np.testing.assert_allclose(
        pred_original["future_attack_probability"],
        pred_loaded["future_attack_probability"],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        pred_original["predicted_risk_score"],
        pred_loaded["predicted_risk_score"],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        pred_original["predicted_next_state"],
        pred_loaded["predicted_next_state"],
        rtol=1e-5,
    )
    assert pred_original["predicted_attack_type"] == pred_loaded["predicted_attack_type"]
    assert pred_original["predicted_attack_stage"] == pred_loaded["predicted_attack_stage"]


# ---------------------------------------------------------------------------
# 8. Target Label Encoders Informative Error on Unknown Label
# ---------------------------------------------------------------------------
def test_unknown_target_label_fails_cleanly():
    type_to_idx, stage_to_idx = create_label_encoders()
    bad_df = pd.DataFrame({
        "future_attack_binary": [1],
        "future_attack_type": ["NonExistentAlienAttack"],
        "future_attack_stage": ["Normal"],
        "future_attack_risk_score": [0.5],
    })
    with pytest.raises(KeyError, match="Unknown attack type"):
        encode_targets(bad_df, type_to_idx, stage_to_idx)


# ---------------------------------------------------------------------------
# 9. Multi-head Evaluation Metrics Calculation
# ---------------------------------------------------------------------------
def test_evaluate_world_model_metrics():
    set_seed(42)
    N = 10
    L = 5
    D = 19
    X = np.random.randn(N, L, D).astype(np.float32)
    y = pd.DataFrame({
        "future_attack_binary": [0, 1] * 5,
        "future_attack_type": ["BENIGN", "DoS"] * 5,
        "future_attack_stage": ["Normal", "Impact"] * 5,
        "future_attack_risk_score": [0.0, 0.8] * 5,
    })
    for feat in STATE_PREDICTIVE_FEATURES:
        y[f"state_{feat}"] = np.zeros(N, dtype=np.float32)

    type_to_idx, stage_to_idx = create_label_encoders()
    model = TemporalWorldModel(input_size=19, hidden_size=32, num_layers=1)
    metrics = evaluate_world_model(model, X, y, type_to_idx, stage_to_idx)

    assert "binary_attack" in metrics
    assert "attack_type" in metrics
    assert "attack_stage" in metrics
    assert "risk_regression" in metrics
    assert "state_prediction" in metrics

    assert "f1" in metrics["binary_attack"]
    assert "f1_macro" in metrics["attack_type"]
    assert "f1_macro" in metrics["attack_stage"]
    assert "mae" in metrics["risk_regression"]
    assert "rmse" in metrics["risk_regression"]
    assert "mae" in metrics["state_prediction"]


# ---------------------------------------------------------------------------
# 10. Stage 3A Contract Remains Unchanged and Compatible
# ---------------------------------------------------------------------------
def test_stage3a_contract_preservation():
    from ml.forecasting.dataset_loader import RealDatasetLoader
    from ml.forecasting.temporal_split import (
        build_partitioned_temporal_dataset,
        partition_states_chronologically,
    )
    from ml.forecasting.temporal_state import build_temporal_states

    loader = RealDatasetLoader()
    flows_df, _ = loader.load_dataset("data/samples/synthetic_network_flows.csv")
    states_df = build_temporal_states(flows_df, window_size_sec=30, stride_sec=10)
    train_s, val_s, test_s = partition_states_chronologically(states_df, 0.60, 0.20, 0.20)

    bundle = build_partitioned_temporal_dataset(
        train_states=train_s,
        val_states=val_s,
        test_states=test_s,
        sequence_length=3,
        forecast_horizon=1,
        scale_features=True,
    )

    assert "X_train" in bundle
    assert "y_train" in bundle
    assert "meta_train" in bundle
    assert "scaler" in bundle
    assert bundle["X_train"].ndim == 3
    assert bundle["X_train"].shape[2] == 19
    assert bundle["X_train"].dtype == np.float32

    # Verify model trains directly on this Stage 3A bundle
    res = train_world_model(
        dataset_bundle=bundle,
        epochs=3,
        seed=42,
    )
    assert res["model"] is not None
    assert "test_metrics" in res


# ---------------------------------------------------------------------------
# 11. State Head Output Shape = (N, 19) and Finite Outputs
# ---------------------------------------------------------------------------
def test_state_head_output_shape_and_finite(default_world_model, synthetic_sequence_batch):
    default_world_model.eval()
    with torch.no_grad():
        out = default_world_model(synthetic_sequence_batch)
    state = out["next_state"]
    assert state.shape == (4, 19)
    assert torch.isfinite(state).all()


# ---------------------------------------------------------------------------
# 12. Multi-Task Loss Includes State Loss & Configurable Weights (Default Risk=2.0)
# ---------------------------------------------------------------------------
def test_multitask_loss_components_and_weights(default_world_model):
    batch_size = 4
    x = torch.randn(batch_size, 5, 19)
    outputs = default_world_model(x)

    targets = (
        torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        torch.tensor([0, 2, 0, 3], dtype=torch.int64),
        torch.tensor([0, 1, 0, 4], dtype=torch.int64),
        torch.tensor([0.1, 0.9, 0.05, 0.85], dtype=torch.float32),
        torch.randn(batch_size, 19),
    )

    # Default weights: state=1.0, binary=1.0, type=1.0, stage=1.0, risk=2.0
    total_loss, loss_dict = compute_multitask_loss(outputs, targets)
    assert "loss_state" in loss_dict
    assert "loss_risk" in loss_dict
    assert "loss_binary" in loss_dict
    assert "loss_type" in loss_dict
    assert "loss_stage" in loss_dict
    assert loss_dict["loss_state"] >= 0.0

    # Custom weights test
    custom_weights = {"state": 5.0, "attack_binary": 0.0, "attack_type": 0.0, "attack_stage": 0.0, "risk": 0.0}
    total_custom, _ = compute_multitask_loss(outputs, targets, loss_weights=custom_weights)
    expected = 5.0 * loss_dict["loss_state"]
    assert np.isclose(total_custom.item(), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# 13. Benign Attribution Masking
# ---------------------------------------------------------------------------
def test_benign_attribution_masking(default_world_model):
    x = torch.randn(4, 5, 19)
    outputs = default_world_model(x)

    # All benign batch (binary_target = 0)
    targets_all_benign = (
        torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.randn(4, 19),
    )

    # With masking enabled: type and stage losses over zero attack samples should be 0.0
    _, loss_dict_masked = compute_multitask_loss(
        outputs, targets_all_benign, mask_benign_attribution=True
    )
    assert np.isclose(loss_dict_masked["loss_type"], 0.0)
    assert np.isclose(loss_dict_masked["loss_stage"], 0.0)

    # With masking disabled: type and stage compute cross-entropy on class 0 (BENIGN/Normal)
    _, loss_dict_unmasked = compute_multitask_loss(
        outputs, targets_all_benign, mask_benign_attribution=False
    )
    assert loss_dict_unmasked["loss_type"] > 0.0
    assert loss_dict_unmasked["loss_stage"] > 0.0


# ---------------------------------------------------------------------------
# 14. Inverse Scaling of Predicted States
# ---------------------------------------------------------------------------
def test_inverse_scaling_of_predicted_states(default_world_model):
    scaler = StandardScaler()
    raw_data = np.random.uniform(10.0, 100.0, size=(100, 19)).astype(np.float32)
    scaler.fit(raw_data)

    type_to_idx, stage_to_idx = create_label_encoders()
    predictor = WorldModelPredictor(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
        scaler=scaler,
    )

    std_state = np.random.randn(5, 19).astype(np.float32)
    phys_state = predictor.inverse_scale_state(std_state)

    assert phys_state.shape == (5, 19)
    # Applying scaler.transform back should recover std_state
    recovered = scaler.transform(phys_state)
    np.testing.assert_allclose(std_state, recovered, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 15. Autoregressive Rollout Shape = (N, K, 19) & Recursive Feedback
# ---------------------------------------------------------------------------
def test_autoregressive_rollout_execution(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    engine = StateRolloutEngine(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    N = 3
    L = 5
    D = 19
    K = 4
    init_seq = np.random.randn(N, L, D).astype(np.float32)

    res = engine.rollout(init_seq, steps=K)

    assert res["predicted_states"].shape == (N, K, D)
    assert res["future_attack_probability"].shape == (N, K)
    assert res["predicted_attack_binary"].shape == (N, K)
    assert res["predicted_risk_score"].shape == (N, K)
    assert len(res["predicted_attack_type"]) == N
    assert len(res["predicted_attack_type"][0]) == K
    assert len(res["predicted_attack_stage"]) == N
    assert len(res["predicted_attack_stage"][0]) == K


# ---------------------------------------------------------------------------
# 16. Rollout K=1 Equivalence with Single-Step State Prediction
# ---------------------------------------------------------------------------
def test_rollout_k1_equivalence(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    predictor = WorldModelPredictor(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    X = np.random.randn(2, 5, 19).astype(np.float32)
    single_pred = predictor.predict(X)
    rollout_pred = predictor.rollout(X, steps=1)

    # State prediction at K=1 must be identical
    np.testing.assert_allclose(
        single_pred["predicted_next_state"],
        rollout_pred["predicted_states"][:, 0, :],
        rtol=1e-5,
    )
    # Risk prediction at K=1 must be identical
    np.testing.assert_allclose(
        single_pred["predicted_risk_score"],
        rollout_pred["predicted_risk_score"][:, 0],
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# 17. Rollout Safety & Validation (K >= 1, Shape, Non-finite values)
# ---------------------------------------------------------------------------
def test_rollout_safety_and_validation(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    engine = StateRolloutEngine(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    valid_x = np.random.randn(2, 5, 19).astype(np.float32)

    # K < 1 must raise
    with pytest.raises(ValueError, match="steps.*must be >= 1"):
        engine.rollout(valid_x, steps=0)

    # Non-finite input must raise
    bad_x = valid_x.copy()
    bad_x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Infinite"):
        engine.rollout(bad_x, steps=2)

    # Wrong shape
    with pytest.raises(ValueError, match="Expected sequence length L=5"):
        engine.rollout(np.random.randn(2, 4, 19), steps=2)


# ---------------------------------------------------------------------------
# 18. Deterministic Rollout Reproducibility
# ---------------------------------------------------------------------------
def test_deterministic_rollout_reproducibility(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    engine = StateRolloutEngine(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    X = np.random.randn(3, 5, 19).astype(np.float32)
    res1 = engine.rollout(X, steps=3)
    res2 = engine.rollout(X, steps=3)

    np.testing.assert_equal(res1["predicted_states"], res2["predicted_states"])
    np.testing.assert_equal(res1["future_attack_probability"], res2["future_attack_probability"])


# ---------------------------------------------------------------------------
# 19. Persistence Baseline Calculation
# ---------------------------------------------------------------------------
def test_persistence_baseline_calculation():
    N, L, D = 4, 5, 19
    K = 3
    seq = np.random.randn(N, L, D).astype(np.float32)

    pers = persistence_baseline(seq, steps=K)
    assert pers.shape == (N, K, D)

    # For every step k in [0, K-1], pers[:, k, :] should equal seq[:, -1, :] (S_t)
    expected_s_t = seq[:, -1, :]
    for k in range(K):
        np.testing.assert_allclose(pers[:, k, :], expected_s_t, rtol=1e-5)


# ---------------------------------------------------------------------------
# 20. Horizon Degradation Metric Table Calculation
# ---------------------------------------------------------------------------
def test_horizon_degradation_evaluation(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    engine = StateRolloutEngine(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    N, L, D = 6, 5, 19
    K = 3
    X_test = np.random.randn(N, L, D).astype(np.float32)
    ground_truth_states = np.random.randn(N, K, D).astype(np.float32)
    threats = {
        "binary": np.random.randint(0, 2, size=(N, K)),
        "risk": np.random.uniform(0.0, 1.0, size=(N, K)).astype(np.float32),
        "type": np.array([["BENIGN"] * K] * N),
        "stage": np.array([["Normal"] * K] * N),
    }

    report = evaluate_horizon_degradation(
        rollout_engine=engine,
        X_test=X_test,
        ground_truth_trajectories=ground_truth_states,
        ground_truth_threats=threats,
        horizons=[1, 2, 3],
    )

    assert "horizon_metrics" in report
    assert len(report["horizon_metrics"]) == 3
    for entry in report["horizon_metrics"]:
        assert "horizon" in entry
        assert "state_mae" in entry
        assert "state_rmse" in entry
        assert "persistence_mae" in entry
        assert "attack_f1" in entry
        assert "risk_mae" in entry


# ---------------------------------------------------------------------------
# 21. Latent Embedding Extraction
# ---------------------------------------------------------------------------
def test_latent_embedding_extraction(default_world_model):
    type_to_idx, stage_to_idx = create_label_encoders()
    predictor = WorldModelPredictor(
        model=default_world_model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
    )

    X = np.random.randn(5, 5, 19).astype(np.float32)
    emb = predictor.get_latent_embedding(X)

    assert emb.shape == (5, 64)
    assert np.isfinite(emb).all()
