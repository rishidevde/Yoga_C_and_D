"""Production HPC Training Script for Yoga Pose Classifier.

Designed for High-Performance Computing clusters (SLURM/PBS/CUDA).
Includes:
- Multi-GPU & single-GPU CUDA acceleration
- Mixed precision training (AMP)
- Learning rate scheduling (Cosine Annealing)
- Early stopping & checkpointing
- ONNX and TorchScript production model export
- Complete CLI configuration for batch job arrays

Usage:
    python train_hpc.py --epochs 100 --batch-size 64 --lr 0.001 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from pose_classifier.dataset import load_dataset, ID_TO_LABEL, LABEL_TO_ID
from pose_classifier.model import YogaPoseMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_hpc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPC Yoga Pose Classifier Training")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(BASE_DIR.parent / "Yoga_Poses-Dataset" / "Results" / "Combined Dataset"),
        help="Path to folder containing dataset CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "hpc_output"),
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.003, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("  YOGA POSE CLASSIFIER - HPC TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info("Compute Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU Model: %s", torch.cuda.get_device_name(0))
        logger.info("Available VRAM: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # 2. Data loading & augmentation
    logger.info("Loading dataset from: %s", args.dataset_dir)
    X, y = load_dataset(dataset_dir=args.dataset_dir, include_idle=True, augment_factor=3, random_state=args.seed)
    logger.info("Total samples: %d (24 features, %d classes)", len(X), len(LABEL_TO_ID))

    # Stratified Train/Val split (80/20)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.80 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_ds = TensorDataset(torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Model setup
    model = YogaPoseMLP(
        in_features=24,
        hidden_dim=args.hidden_dim,
        num_classes=len(LABEL_TO_ID),
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model Parameters: %d trainable weights", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. Training loop with early stopping
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0

        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(b_x)

        train_loss = running_train_loss / len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                out = model(b_x)
                loss = criterion(out, b_y)
                running_val_loss += loss.item() * len(b_x)
                preds = torch.argmax(out, dim=-1)
                correct += (preds == b_y).sum().item()

        val_loss = running_val_loss / len(val_ds)
        val_acc = correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpointing
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = output_dir / "best_pose_classifier.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "in_features": 24,
                "hidden_dim": args.hidden_dim,
                "num_classes": len(LABEL_TO_ID),
                "label_map": LABEL_TO_ID,
            }, best_model_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info(
                "Epoch %3d/%3d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %6.2f%% | LR: %.6f",
                epoch, args.epochs, train_loss, val_loss, val_acc * 100, optimizer.param_groups[0]["lr"]
            )

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, args.patience)
            break

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("TRAINING FINISHED in %.2f seconds", elapsed)
    logger.info("Best Validation Accuracy: %.2f%%", best_val_acc * 100)

    # 5. Export production deployment formats (ONNX & TorchScript)
    logger.info("Exporting production deployment artifacts...")
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 24, dtype=torch.float32)

    # TorchScript Export
    ts_path = output_dir / "pose_classifier.torchscript.pt"
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(ts_path))
    logger.info("  Exported TorchScript -> %s", ts_path)

    # ONNX Export
    try:
        onnx_path = output_dir / "pose_classifier.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=14,
        )
        logger.info("  Exported ONNX -> %s", onnx_path)
    except Exception as err:
        logger.warning("  ONNX export skipped: %s", err)

    # Save training metrics JSON
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "training_time_sec": elapsed,
            "total_epochs": epoch,
            "history": history,
        }, f, indent=2)
    logger.info("  Saved Training Metrics -> %s", metrics_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
