"""Reproducible Training & Evaluation Pipeline for NETRA Stage 3B.2 Temporal World Model.

Trains LSTM multi-head world model using Stage 3A partitioned data:
- Deterministic random seeding
- Scaler strictly fitted on training sequences
- Next-state prediction target (S_(t+1) in R^19)
- Multi-task loss with configurable weights (defaults: state=1.0, binary=1.0, type=1.0, stage=1.0, risk=2.0)
- Benign target masking (mask_benign_attribution=True)
- Early stopping & best model checkpointing (v2 checkpoint)
- Comprehensive evaluation metrics for binary, multi-class, continuous risk, and state predictions
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.forecasting.dataset_loader import RealDatasetLoader
from ml.forecasting.predict_world_model import WorldModelPredictor
from ml.forecasting.schema import (
    LABEL_NORMALIZATION_MAP,
    STATE_PREDICTIVE_FEATURES,
    TAXONOMY_DISCLAIMER,
    VALID_ATTACK_STAGES,
)
from ml.forecasting.temporal_split import (
    build_partitioned_temporal_dataset,
    partition_states_chronologically,
)
from ml.forecasting.temporal_state import build_temporal_states
from ml.forecasting.world_model import CANONICAL_ATTACK_TYPES, TemporalWorldModel

logger = logging.getLogger(__name__)

SYNTHETIC_WARNING: str = (
    "Synthetic fixture result — not representative of CIC-IDS2017 benchmark performance."
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for determinism."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_label_encoders(
    attack_types: Optional[List[str]] = None,
    attack_stages: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Generate deterministic string-to-index mappings."""
    types = attack_types or list(CANONICAL_ATTACK_TYPES)
    stages = attack_stages or list(VALID_ATTACK_STAGES)

    type_to_idx = {name: idx for idx, name in enumerate(types)}
    stage_to_idx = {name: idx for idx, name in enumerate(stages)}
    return type_to_idx, stage_to_idx


def encode_targets(
    y_df: pd.DataFrame,
    type_to_idx: Dict[str, int],
    stage_to_idx: Dict[str, int],
    feature_names: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert pandas target DataFrame into PyTorch tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        (binary_tensor, type_tensor, stage_tensor, risk_tensor, next_state_tensor)
    """
    # 1. Binary target
    binary_arr = y_df["future_attack_binary"].to_numpy(dtype=np.int64).copy()
    binary_tensor = torch.from_numpy(binary_arr)

    # 2. Attack type
    type_arr = np.zeros(len(y_df), dtype=np.int64)
    for i, t_val in enumerate(y_df["future_attack_type"]):
        clean_t = str(t_val).strip()
        norm_t = LABEL_NORMALIZATION_MAP.get(clean_t.lower(), clean_t)
        if norm_t not in type_to_idx:
            raise KeyError(f"Unknown attack type: '{t_val}'. Registered types: {list(type_to_idx.keys())}")
        type_arr[i] = type_to_idx[norm_t]
    type_tensor = torch.from_numpy(type_arr.copy())

    # 3. Attack stage
    stage_arr = np.zeros(len(y_df), dtype=np.int64)
    for i, s_val in enumerate(y_df["future_attack_stage"]):
        clean_s = str(s_val).strip()
        if clean_s not in stage_to_idx:
            raise KeyError(f"Unknown attack stage: '{s_val}'. Registered stages: {list(stage_to_idx.keys())}")
        stage_arr[i] = stage_to_idx[clean_s]
    stage_tensor = torch.from_numpy(stage_arr.copy())

    # 4. Risk score
    risk_arr = y_df["future_attack_risk_score"].to_numpy(dtype=np.float32).copy()
    risk_tensor = torch.from_numpy(risk_arr)

    # 5. Next state target (if state columns exist in y_df)
    features = feature_names or list(STATE_PREDICTIVE_FEATURES)
    matching_state_cols = [f"state_{feat}" for feat in features if f"state_{feat}" in y_df.columns]
    if len(matching_state_cols) == len(features):
        state_arr = y_df[matching_state_cols].to_numpy(dtype=np.float32).copy()
        next_state_tensor = torch.from_numpy(state_arr)
    elif all(feat in y_df.columns for feat in features):
        state_arr = y_df[features].to_numpy(dtype=np.float32).copy()
        next_state_tensor = torch.from_numpy(state_arr)
    else:
        next_state_tensor = None

    return binary_tensor, type_tensor, stage_tensor, risk_tensor, next_state_tensor


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    loss_weights: Optional[Dict[str, float]] = None,
    criterion_risk: Optional[nn.Module] = None,
    criterion_state: Optional[nn.Module] = None,
    mask_benign_attribution: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculate multi-task loss with next-state loss and configurable benign masking.

    Default loss weights:
      state = 1.0, binary = 1.0, type = 1.0, stage = 1.0, risk = 2.0
    (Risk defaults to 2.0 because MSE on [0, 1] produces smaller gradient magnitude
    relative to CrossEntropy on classification heads).

    When mask_benign_attribution=True:
      - Binary loss: calculated for all samples
      - Risk loss: calculated for all samples
      - State loss: calculated for all samples
      - Type loss: calculated only where binary_target == 1
      - Stage loss: calculated only where binary_target == 1
    """
    weights = loss_weights or {
        "state": 1.0,
        "attack_binary": 1.0,
        "attack_type": 1.0,
        "attack_stage": 1.0,
        "risk": 2.0,
    }
    crit_risk = criterion_risk or nn.MSELoss()
    crit_state = criterion_state or nn.SmoothL1Loss()
    crit_ce = nn.CrossEntropyLoss()

    binary_target, type_target, stage_target, risk_target, state_target = targets

    # 1. Binary loss (all samples)
    l_binary = crit_ce(outputs["binary_logits"], binary_target)

    # 2 & 3. Type & Stage losses (optionally masked for benign windows)
    if mask_benign_attribution:
        attack_mask = binary_target == 1
        if attack_mask.any():
            l_type = crit_ce(outputs["type_logits"][attack_mask], type_target[attack_mask])
            l_stage = crit_ce(outputs["stage_logits"][attack_mask], stage_target[attack_mask])
        else:
            # If batch has zero attacks, fall back to unweighted 0-loss tensor with grad graph intact
            l_type = 0.0 * crit_ce(outputs["type_logits"], type_target)
            l_stage = 0.0 * crit_ce(outputs["stage_logits"], stage_target)
    else:
        l_type = crit_ce(outputs["type_logits"], type_target)
        l_stage = crit_ce(outputs["stage_logits"], stage_target)

    # 4. Risk loss (all samples)
    l_risk = crit_risk(outputs["risk_score"], risk_target)

    # 5. Next-State regression loss (all samples)
    if state_target is not None and "next_state" in outputs:
        l_state = crit_state(outputs["next_state"], state_target)
    else:
        l_state = torch.tensor(0.0, device=outputs["binary_logits"].device)

    w_state = weights.get("state", 1.0)
    w_bin = weights.get("attack_binary", 1.0)
    w_type = weights.get("attack_type", 1.0)
    w_stage = weights.get("attack_stage", 1.0)
    w_risk = weights.get("risk", 2.0)

    total_loss = (
        w_state * l_state
        + w_bin * l_binary
        + w_type * l_type
        + w_stage * l_stage
        + w_risk * l_risk
    )

    losses = {
        "total_loss": float(total_loss.item()),
        "loss_state": float(l_state.item()),
        "loss_binary": float(l_binary.item()),
        "loss_type": float(l_type.item()),
        "loss_stage": float(l_stage.item()),
        "loss_risk": float(l_risk.item()),
    }
    return total_loss, losses


def evaluate_world_model(
    model: TemporalWorldModel,
    X: np.ndarray,
    y: pd.DataFrame,
    type_to_idx: Dict[str, int],
    stage_to_idx: Dict[str, int],
    batch_size: int = 32,
    device: Optional[str] = None,
    is_synthetic: bool = True,
    feature_names: Optional[List[str]] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Calculate performance metrics across all five forecasting heads."""
    if len(X) == 0 or len(y) == 0:
        return {
            "status": "EMPTY_EVALUATION",
            "samples": 0,
            "disclaimer": SYNTHETIC_WARNING if is_synthetic else TAXONOMY_DISCLAIMER,
        }

    predictor = WorldModelPredictor(
        model=model,
        type_to_idx=type_to_idx,
        stage_to_idx=stage_to_idx,
        device=device,
        feature_names=feature_names,
        scaler=scaler,
    )
    preds = predictor.predict(X)

    # 1. Binary Attack Metrics
    y_true_binary = y["future_attack_binary"].to_numpy(dtype=int)
    y_pred_binary = preds["predicted_attack_binary"]
    y_prob_binary = preds["future_attack_probability"]

    acc_bin = float(accuracy_score(y_true_binary, y_pred_binary))
    prec_bin = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    rec_bin = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1_bin = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    roc_auc: Optional[float] = None
    if len(np.unique(y_true_binary)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_binary, y_prob_binary))
        except Exception:
            roc_auc = None

    # 2. Attack Type Metrics
    y_true_type = y["future_attack_type"].astype(str).tolist()
    y_pred_type = preds["predicted_attack_type"]
    acc_type = float(accuracy_score(y_true_type, y_pred_type))
    f1_type_macro = float(f1_score(y_true_type, y_pred_type, average="macro", zero_division=0))
    f1_type_weighted = float(f1_score(y_true_type, y_pred_type, average="weighted", zero_division=0))

    # 3. Attack Stage Metrics
    y_true_stage = y["future_attack_stage"].astype(str).tolist()
    y_pred_stage = preds["predicted_attack_stage"]
    acc_stage = float(accuracy_score(y_true_stage, y_pred_stage))
    f1_stage_macro = float(f1_score(y_true_stage, y_pred_stage, average="macro", zero_division=0))
    f1_stage_weighted = float(f1_score(y_true_stage, y_pred_stage, average="weighted", zero_division=0))

    # 4. Risk Score Regression Metrics
    y_true_risk = y["future_attack_risk_score"].to_numpy(dtype=float)
    y_pred_risk = preds["predicted_risk_score"]
    mae_risk = float(mean_absolute_error(y_true_risk, y_pred_risk))
    rmse_risk = float(np.sqrt(mean_squared_error(y_true_risk, y_pred_risk)))

    r2_risk: Optional[float] = None
    if len(y_true_risk) > 1 and np.var(y_true_risk) > 1e-6:
        try:
            r2_risk = float(r2_score(y_true_risk, y_pred_risk))
        except Exception:
            r2_risk = None

    # 5. State Prediction Metrics (if available in target)
    state_metrics = None
    features = feature_names or list(STATE_PREDICTIVE_FEATURES)
    matching_state_cols = [f"state_{feat}" for feat in features if f"state_{feat}" in y.columns]
    if len(matching_state_cols) == len(features) and "predicted_next_state" in preds:
        y_true_state = y[matching_state_cols].to_numpy(dtype=float)
        y_pred_state = preds["predicted_next_state"]
        state_mae = float(mean_absolute_error(y_true_state, y_pred_state))
        state_rmse = float(np.sqrt(mean_squared_error(y_true_state, y_pred_state)))
        state_metrics = {
            "mae": round(state_mae, 4),
            "rmse": round(state_rmse, 4),
        }

    return {
        "num_samples": len(X),
        "synthetic_warning": SYNTHETIC_WARNING if is_synthetic else None,
        "taxonomy_disclaimer": TAXONOMY_DISCLAIMER,
        "binary_attack": {
            "accuracy": round(acc_bin, 4),
            "precision": round(prec_bin, 4),
            "recall": round(rec_bin, 4),
            "f1": round(f1_bin, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "class_distribution": pd.Series(y_true_binary).value_counts().to_dict(),
        },
        "attack_type": {
            "accuracy": round(acc_type, 4),
            "f1_macro": round(f1_type_macro, 4),
            "f1_weighted": round(f1_type_weighted, 4),
            "class_distribution": pd.Series(y_true_type).value_counts().to_dict(),
            "prediction_distribution": pd.Series(y_pred_type).value_counts().to_dict(),
        },
        "attack_stage": {
            "accuracy": round(acc_stage, 4),
            "f1_macro": round(f1_stage_macro, 4),
            "f1_weighted": round(f1_stage_weighted, 4),
            "class_distribution": pd.Series(y_true_stage).value_counts().to_dict(),
            "prediction_distribution": pd.Series(y_pred_stage).value_counts().to_dict(),
        },
        "risk_regression": {
            "mae": round(mae_risk, 4),
            "rmse": round(rmse_risk, 4),
            "r2": round(r2_risk, 4) if r2_risk is not None else None,
        },
        "state_prediction": state_metrics,
    }


def train_world_model(
    dataset_bundle: Dict[str, Any],
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 7,
    seed: int = 42,
    loss_weights: Optional[Dict[str, float]] = None,
    mask_benign_attribution: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    is_synthetic: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate TemporalWorldModel with early stopping and v2 checkpointing."""
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    X_train = dataset_bundle["X_train"]
    y_train = dataset_bundle["y_train"]
    X_val = dataset_bundle["X_val"]
    y_val = dataset_bundle["y_val"]
    X_test = dataset_bundle["X_test"]
    y_test = dataset_bundle["y_test"]

    if len(X_train) == 0:
        raise ValueError("X_train has 0 sequences. Cannot train world model.")

    feature_names = dataset_bundle.get("features", list(STATE_PREDICTIVE_FEATURES))
    input_size = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Encoders
    type_to_idx, stage_to_idx = create_label_encoders()

    # Build model (now with head_state in R^19)
    model = TemporalWorldModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_attack_types=len(type_to_idx),
        num_attack_stages=len(stage_to_idx),
        attack_types=list(type_to_idx.keys()),
        attack_stages=list(stage_to_idx.keys()),
        feature_names=feature_names,
    ).to(dev)

    # Convert Training Data
    tr_bin, tr_type, tr_stage, tr_risk, tr_state = encode_targets(
        y_train, type_to_idx, stage_to_idx, feature_names=feature_names
    )
    train_x = torch.from_numpy(X_train.astype(np.float32))
    tensor_list = [train_x, tr_bin, tr_type, tr_stage, tr_risk]
    if tr_state is not None:
        tensor_list.append(tr_state)

    train_dataset = TensorDataset(*tensor_list)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    # Convert Validation Data if present
    has_val = len(X_val) > 0 and len(y_val) > 0
    val_loader = None
    if has_val:
        val_bin, val_type, val_stage, val_risk, val_state = encode_targets(
            y_val, type_to_idx, stage_to_idx, feature_names=feature_names
        )
        val_x = torch.from_numpy(X_val.astype(np.float32))
        val_tensors = [val_x, val_bin, val_type, val_stage, val_risk]
        if val_state is not None:
            val_tensors.append(val_state)
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_risk = nn.MSELoss()
    criterion_state = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    logger.info(
        f"Initiating World Model v2 training: samples={len(X_train)}, "
        f"L={seq_len}, D={input_size}, epochs={epochs}, lr={learning_rate}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            bx = batch[0].to(dev)
            b_bin = batch[1].to(dev)
            b_type = batch[2].to(dev)
            b_stage = batch[3].to(dev)
            b_risk = batch[4].to(dev)
            b_state = batch[5].to(dev) if len(batch) > 5 else None

            btargets = (b_bin, b_type, b_stage, b_risk, b_state)

            optimizer.zero_grad()
            outputs = model(bx)
            loss, _ = compute_multitask_loss(
                outputs=outputs,
                targets=btargets,
                loss_weights=loss_weights,
                criterion_risk=criterion_risk,
                criterion_state=criterion_state,
                mask_benign_attribution=mask_benign_attribution,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        avg_train_loss = epoch_loss / max(1, batch_count)

        # Validation Step
        avg_val_loss = avg_train_loss
        if has_val:
            model.eval()
            val_loss_acc = 0.0
            val_batches = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx = vbatch[0].to(dev)
                    v_bin = vbatch[1].to(dev)
                    v_type = vbatch[2].to(dev)
                    v_stage = vbatch[3].to(dev)
                    v_risk = vbatch[4].to(dev)
                    v_state = vbatch[5].to(dev) if len(vbatch) > 5 else None

                    vtargets = (v_bin, v_type, v_stage, v_risk, v_state)
                    voutputs = model(vx)
                    vloss, _ = compute_multitask_loss(
                        outputs=voutputs,
                        targets=vtargets,
                        loss_weights=loss_weights,
                        criterion_risk=criterion_risk,
                        criterion_state=criterion_state,
                        mask_benign_attribution=mask_benign_attribution,
                    )
                    val_loss_acc += float(vloss.item())
                    val_batches += 1
            avg_val_loss = val_loss_acc / max(1, val_batches)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
        })

        # Early Stopping & Best Checkpoint Tracking
        monitor_loss = avg_val_loss if has_val else avg_train_loss
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final Evaluation
    val_metrics = (
        evaluate_world_model(
            model,
            X_val,
            y_val,
            type_to_idx,
            stage_to_idx,
            device=device,
            is_synthetic=is_synthetic,
            feature_names=feature_names,
            scaler=dataset_bundle.get("scaler"),
        )
        if has_val
        else {"status": "NO_VALIDATION_SPLIT"}
    )
    test_metrics = (
        evaluate_world_model(
            model,
            X_test,
            y_test,
            type_to_idx,
            stage_to_idx,
            device=device,
            is_synthetic=is_synthetic,
            feature_names=feature_names,
            scaler=dataset_bundle.get("scaler"),
        )
        if len(X_test) > 0
        else {"status": "NO_TEST_SPLIT"}
    )

    # Model Checkpoint Dictionary (v2 bundle)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "architecture": model.get_config(),
        "type_to_idx": type_to_idx,
        "stage_to_idx": stage_to_idx,
        "attack_types": list(type_to_idx.keys()),
        "attack_stages": list(stage_to_idx.keys()),
        "feature_names": feature_names,
        "scaler": dataset_bundle.get("scaler"),
        "training_config": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "seed": seed,
            "loss_weights": loss_weights or {
                "state": 1.0,
                "attack_binary": 1.0,
                "attack_type": 1.0,
                "attack_stage": 1.0,
                "risk": 2.0,
            },
            "mask_benign_attribution": mask_benign_attribution,
            "best_loss": round(best_val_loss, 4),
        },
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "synthetic_warning": SYNTHETIC_WARNING if is_synthetic else None,
    }

    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "temporal_world_model_v2.pt"
        torch.save(checkpoint_payload, ckpt_path)

        # Export metrics JSON
        metrics_export = {
            "training_config": checkpoint_payload["training_config"],
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "synthetic_warning": SYNTHETIC_WARNING if is_synthetic else None,
        }
        with open(out_dir / "world_model_metrics_v2.json", "w") as f:
            json.dump(metrics_export, f, indent=2)

        logger.info(f"Temporal World Model v2 checkpoint saved to: {ckpt_path}")

    return {
        "model": model,
        "checkpoint": checkpoint_payload,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def run_stage3b_pipeline(
    raw_path: Union[str, Path] = "data/samples/synthetic_network_flows.csv",
    save_dir: Union[str, Path] = "ml/models",
    epochs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 3B.2 data-to-world-model pipeline."""
    loader = RealDatasetLoader()
    flows_df, _ = loader.load_dataset(raw_path)

    states_df = build_temporal_states(flows_df, window_size_sec=30, stride_sec=10)
    train_s, val_s, test_s = partition_states_chronologically(states_df, 0.60, 0.20, 0.20)

    bundle = build_partitioned_temporal_dataset(
        train_states=train_s,
        val_states=val_s,
        test_states=test_s,
        sequence_length=3,  # Fits within small test samples
        forecast_horizon=1,
        scale_features=True,
    )

    results = train_world_model(
        dataset_bundle=bundle,
        epochs=epochs,
        seed=seed,
        save_dir=save_dir,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NETRA Stage 3B.2 Temporal World Model")
    parser.add_argument("--input", default="data/samples/synthetic_network_flows.csv", help="Input dataset path")
    parser.add_argument("--save_dir", default="ml/models", help="Directory to save model checkpoint")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    res = run_stage3b_pipeline(
        raw_path=args.input,
        save_dir=args.save_dir,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps({
        "status": "STAGE 3B.2 TRAINING COMPLETE",
        "best_loss": res["checkpoint"]["training_config"]["best_loss"],
        "epochs_trained": len(res["history"]),
        "val_metrics": res["val_metrics"],
    }, indent=2))
