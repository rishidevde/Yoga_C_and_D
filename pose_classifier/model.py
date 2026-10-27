"""Model architectures for Yoga Pose Classification.

Provides:
1. `YogaPoseMLP`: PyTorch lightweight Neural Network (ideal for HPC CUDA training & TorchScript/ONNX export).
2. `SklearnPoseClassifier`: Scikit-Learn Random Forest model (zero-dependency CPU runtime).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class YogaPoseMLP(nn.Module):
    """Ultra-lightweight Multi-Layer Perceptron for 9-class Yoga Pose Classification.

    Input dimension: 24 features
    Output dimension: 9 classes (8 yoga poses + 1 idle)
    Model Size: ~4,800 parameters (< 25 KB).
    """

    def __init__(
        self,
        in_features: int = 24,
        hidden_dim: int = 64,
        num_classes: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax probability distribution."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


def build_sklearn_classifier(n_estimators: int = 120, random_state: int = 42) -> Pipeline:
    """Build an optimized Scikit-Learn ensemble classifier pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )),
    ])
