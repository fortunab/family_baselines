"""
SVM Training Pipeline with RBF Kernel, Hyperparameter Optimization, and Cross-Validation.
Matches the baseline methodology of Kather et al. (2016).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def create_svm_pipeline(
    C: float = 10.0,
    gamma: Any = "scale",
    probability: bool = True,
    random_state: int = 45
) -> Pipeline:
    """
    Creates a scikit-learn Pipeline with StandardScaler and RBF-kernel SVC.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=C,
            gamma=gamma,
            probability=probability,
            class_weight='balanced',
            random_state=random_state
        ))
    ])


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Performs grid search cross-validation to find the optimal C and gamma parameters for RBF-SVM.
    """
    if param_grid is None:
        param_grid = {
            'svm__C': [0.1, 1.0, 10.0, 100.0],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }

    base_pipe = create_svm_pipeline()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print(f"[SVM-Train] Running Grid Search CV ({cv_folds} folds) over hyperparameter grid: {param_grid}...")
    grid_search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    print(f"[SVM-Train] Best Cross-Validation Accuracy: {grid_search.best_score_*100:.2f}%")
    print(f"[SVM-Train] Optimal Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_and_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    C: float = 10.0,
    gamma: Any = "scale",
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    """
    Runs full Stratified K-Fold Cross-Validation on the entire dataset.
    Returns:
    - y_pred_cv: Out-of-fold predicted class labels (N,)
    - y_proba_cv: Out-of-fold predicted class probabilities (N, n_classes)
    - final_pipeline: Pipeline fitted on all data
    """
    pipeline = create_svm_pipeline(C=C, gamma=gamma, probability=True, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"[SVM-Train] Generating out-of-fold predictions using {n_splits}-fold Stratified CV...")
    y_proba_cv = cross_val_predict(
        pipeline, X, y, cv=cv, method='predict_proba', n_jobs=n_jobs
    )
    y_pred_cv = np.argmax(y_proba_cv, axis=1)

    # Fit final pipeline on entire dataset
    print("[SVM-Train] Fitting final model on all 5,000 samples...")
    pipeline.fit(X, y)

    return y_pred_cv, y_proba_cv, pipeline


def save_model(pipeline: Pipeline, output_path: Path):
    """
    Saves trained pipeline to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"[SVM-Train] Model saved successfully to: {output_path}")


def load_model(model_path: Path) -> Pipeline:
    """
    Loads saved pipeline from disk.
    """
    return joblib.load(model_path)
