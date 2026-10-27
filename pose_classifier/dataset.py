"""Dataset loader and synthetic transition generator for yoga pose classification.

Loads annotated landmarks from Yoga_Poses-Dataset/Results/Combined Dataset,
extracts invariant features, adds neutral/idle stance samples, and provides
training and validation batches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pose_classifier.features import extract_from_dataframe_row, extract_feature_vector

logger = logging.getLogger(__name__)

# Canonical Label Mapping (9 Classes: 8 poses + 1 idle/transition)
LABEL_TO_ID = {
    "idle": 0,
    "triangle": 1,
    "ardhachandrasana": 2,
    "baddhakonasana": 3,
    "downward_dog": 4,
    "natarajasana": 5,
    "utkatakonasana": 6,
    "veerabhadrasana": 7,
    "vrukshasana": 8,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Dataset CSV file mapping
DATASET_CSV_MAP = {
    "triangle": "Triangle_Combined.csv",
    "ardhachandrasana": "ArdhaChandrasana_Combined.csv",
    "baddhakonasana": "BaddhaKonasana_Combined.csv",
    "downward_dog": "DownwardDog_Combined.csv",
    "natarajasana": "Natarajasana_Combined.csv",
    "utkatakonasana": "UtkataKonasana_Combined.csv",
    "veerabhadrasana": "Veerabhadrasana_Combined.csv",
    "vrukshasana": "Vrukshana_Combined.csv",
}


def generate_synthetic_idle_samples(num_samples: int = 70, rng_seed: int = 42) -> np.ndarray:
    """Generate synthetic neutral/idle standing postures (Mountain Pose / standing / walking).

    In an idle stance:
    - User is upright (shoulders above hips, hips above knees above ankles)
    - Arms hang naturally at sides (elbows straight, wrists near hips)
    - Legs are straight, feet together or hip-width apart
    """
    rng = np.random.RandomState(rng_seed)
    idle_features = []

    for _ in range(num_samples):
        # Base neutral points
        pts = np.zeros((33, 2), dtype=np.float32)

        # Center x around 0.5 with slight shift
        cx = 0.5 + rng.normal(0, 0.05)
        # Head
        pts[0] = [cx, 0.20 + rng.normal(0, 0.02)]
        # Shoulders
        pts[11] = [cx - 0.10 + rng.normal(0, 0.01), 0.32 + rng.normal(0, 0.01)]
        pts[12] = [cx + 0.10 + rng.normal(0, 0.01), 0.32 + rng.normal(0, 0.01)]
        # Elbows (at waist)
        pts[13] = [cx - 0.12 + rng.normal(0, 0.015), 0.45 + rng.normal(0, 0.015)]
        pts[14] = [cx + 0.12 + rng.normal(0, 0.015), 0.45 + rng.normal(0, 0.015)]
        # Wrists (at mid thighs)
        pts[15] = [cx - 0.13 + rng.normal(0, 0.02), 0.58 + rng.normal(0, 0.02)]
        pts[16] = [cx + 0.13 + rng.normal(0, 0.02), 0.58 + rng.normal(0, 0.02)]
        # Hips
        pts[23] = [cx - 0.07 + rng.normal(0, 0.01), 0.52 + rng.normal(0, 0.01)]
        pts[24] = [cx + 0.07 + rng.normal(0, 0.01), 0.52 + rng.normal(0, 0.01)]
        # Knees
        pts[25] = [cx - 0.07 + rng.normal(0, 0.01), 0.70 + rng.normal(0, 0.015)]
        pts[26] = [cx + 0.07 + rng.normal(0, 0.01), 0.70 + rng.normal(0, 0.015)]
        # Ankles
        pts[27] = [cx - 0.07 + rng.normal(0, 0.015), 0.88 + rng.normal(0, 0.015)]
        pts[28] = [cx + 0.07 + rng.normal(0, 0.015), 0.88 + rng.normal(0, 0.015)]

        feat = extract_feature_vector(pts)
        idle_features.append(feat)

    return np.array(idle_features, dtype=np.float32)


def load_dataset(
    dataset_dir: str | Path = r"C:\Yoga_F\Yoga_Poses-Dataset\Results\Combined Dataset",
    include_idle: bool = True,
    augment_factor: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and integer labels from CSV files with data augmentation.

    Args:
        dataset_dir: Path to Combined Dataset directory.
        include_idle: Whether to synthesize idle/transition class.
        augment_factor: Number of perturbed augmented versions per sample.
        random_state: Random seed for reproducibility.

    Returns:
        X: (N, 24) float32 feature matrix.
        y: (N,) int64 class labels.
    """
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Combined Dataset directory not found: {path}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    rng = np.random.RandomState(random_state)

    for label_str, csv_name in DATASET_CSV_MAP.items():
        fpath = path / csv_name
        if not fpath.exists():
            logger.warning("Dataset CSV not found: %s", fpath)
            continue

        df = pd.read_csv(fpath)
        class_id = LABEL_TO_ID[label_str]

        for _, row in df.iterrows():
            feat = extract_from_dataframe_row(row)
            X_list.append(feat)
            y_list.append(class_id)

            # Data Augmentation: slight Gaussian jitter on features
            for _ in range(augment_factor):
                jitter = rng.normal(0.0, 0.015, size=feat.shape)
                # Keep angles in reasonable [0, 1] range
                aug_feat = feat + jitter
                aug_feat[:10] = np.clip(aug_feat[:10], 0.0, 1.0)
                X_list.append(aug_feat)
                y_list.append(class_id)

    # Add synthetic idle / transition class
    if include_idle:
        idle_feats = generate_synthetic_idle_samples(num_samples=70, rng_seed=random_state)
        for feat in idle_feats:
            X_list.append(feat)
            y_list.append(LABEL_TO_ID["idle"])

            for _ in range(augment_factor):
                jitter = rng.normal(0.0, 0.015, size=feat.shape)
                aug_feat = np.clip(feat + jitter, 0.0, 1.0)
                X_list.append(aug_feat)
                y_list.append(LABEL_TO_ID["idle"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


def get_train_val_test_splits(
    dataset_dir: str | Path = r"C:\Yoga_F\Yoga_Poses-Dataset\Results\Combined Dataset",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and return stratified Train and Test splits."""
    X, y = load_dataset(dataset_dir=dataset_dir, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
