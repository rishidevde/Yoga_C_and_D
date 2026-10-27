"""Local training and dry-run validation script for Yoga Pose Classifier.

Trains both Scikit-Learn ensemble and PyTorch MLP on the Combined Dataset,
evaluates cross-validation performance, and exports model artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

from pose_classifier.dataset import load_dataset, ID_TO_LABEL, LABEL_TO_ID
from pose_classifier.model import YogaPoseMLP, build_sklearn_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_local")


def train_sklearn_local(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> float:
    """Train Scikit-Learn ensemble with Stratified 5-Fold Cross Validation."""
    logger.info("=" * 65)
    logger.info("TRAINING SCIKIT-LEARN ENSEMBLE CLASSIFIER")
    logger.info("=" * 65)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf = build_sklearn_classifier()
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], y_pred)
        fold_accs.append(acc)
        logger.info("  Fold %d Accuracy: %.2f%%", fold, acc * 100)

    mean_acc = float(np.mean(fold_accs))
    logger.info("5-Fold Mean Accuracy: %.2f%% (std: %.2f%%)", mean_acc * 100, float(np.std(fold_accs)) * 100)

    # Train final model on full dataset
    logger.info("Fitting final production model on full dataset (%d samples)...", len(X))
    final_clf = build_sklearn_classifier()
    final_clf.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, output_path)
    logger.info("Saved Scikit-Learn model artifact -> %s", output_path)

    # Print Full Evaluation Report
    y_full_pred = final_clf.predict(X)
    target_names = [ID_TO_LABEL[i] for i in range(len(LABEL_TO_ID))]
    logger.info("\nClassification Report (Full Dataset Fit):\n%s", classification_report(y, y_full_pred, target_names=target_names))

    return mean_acc


def train_pytorch_local(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.005,
) -> float:
    """Train PyTorch YogaPoseMLP model locally."""
    logger.info("=" * 65)
    logger.info("TRAINING PYTORCH YOGAPOSEMLP MODEL")
    logger.info("=" * 65)

    device = torch.device("cpu")
    model = YogaPoseMLP(in_features=24, hidden_dim=64, num_classes=9, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train-test split
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_pt = int(0.85 * len(X))
    train_idx, val_idx = indices[:split_pt], indices[split_pt:]

    X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.long)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), batch_size):
            batch_indices = permutation[i:i + batch_size]
            b_x, b_y = X_train_t[batch_indices], y_train_t[batch_indices]

            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_x)

        epoch_loss /= len(X_train_t)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_preds = torch.argmax(val_out, dim=-1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "in_features": 24,
                "hidden_dim": 64,
                "num_classes": 9,
            }, output_path)

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("  Epoch %2d/%2d | Train Loss: %.4f | Val Acc: %.2f%%", epoch, epochs, epoch_loss, val_acc * 100)

    logger.info("Best PyTorch Validation Accuracy: %.2f%%", best_val_acc * 100)
    logger.info("Saved PyTorch checkpoint artifact -> %s", output_path)
    return best_val_acc


def main():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir.parent / "Yoga_Poses-Dataset" / "Results" / "Combined Dataset"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data from %s...", dataset_dir)
    X, y = load_dataset(dataset_dir=dataset_dir, include_idle=True, augment_factor=2)
    logger.info("Dataset loaded successfully: %d samples across %d classes.", len(X), len(LABEL_TO_ID))

    # 1. Train Scikit-Learn
    skl_out = models_dir / "pose_classifier.pkl"
    train_sklearn_local(X, y, skl_out)

    # 2. Train PyTorch
    torch_out = models_dir / "pose_classifier.pt"
    train_pytorch_local(X, y, torch_out)

    logger.info("\nALL LOCAL TRAINING & DRY-RUNS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
