"""Machine Learning (ML) Family Training and Evaluation Engine.

Evaluates 7 Classical and Ensemble Machine Learning algorithms:
1. Support Vector Machine (SVM RBF)
2. Random Forest Classifier
3. Extra Trees Classifier
4. Histogram Gradient Boosting
5. Multi-Layer Perceptron (MLP)
6. Logistic Regression
7. Soft Voting Ensemble
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluator import evaluate_clinical_metrics


def build_ml_model_catalog(seed: int = 42) -> Dict[str, Any]:
    """Instantiates the 7 ML family classifiers with tuned hyperparameters."""
    svm_clf = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )

    et_clf = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=25,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )

    gb_clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        random_state=seed,
    )

    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=0.001,
        max_iter=300,
        random_state=seed,
    )

    lr_clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )

    voting_clf = VotingClassifier(
        estimators=[
            ("extra_trees", et_clf),
            ("svm", svm_clf),
            ("grad_boost", gb_clf),
        ],
        voting="soft",
        weights=[2, 2, 1],
    )

    return {
        "svm_rbf": svm_clf,
        "random_forest": rf_clf,
        "extra_trees": et_clf,
        "gradient_boosting": gb_clf,
        "mlp": mlp_clf,
        "logistic_regression": lr_clf,
        "voting_ensemble": voting_clf,
    }


def train_and_eval_single_ml_model(
    model_name: str,
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    scale_features: bool = True,
) -> Dict[str, Any]:
    """Trains an individual ML model, measures timing, and computes clinical metrics."""
    if scale_features:
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    else:
        pipeline = clf

    # Train and measure duration
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    t_train = time.time() - t_start

    # Predict on test set
    t_inf_start = time.time()
    y_pred = pipeline.predict(X_test)
    t_inf = time.time() - t_inf_start
    inf_latency_ms = (t_inf / len(X_test)) * 1000.0

    # Probabilities
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
        except Exception:
            y_prob = None

    metrics = evaluate_clinical_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        threshold_target=0.80,
    )

    metrics["model_name"] = model_name
    metrics["training_time_sec"] = round(t_train, 3)
    metrics["inference_latency_ms_per_slice"] = round(inf_latency_ms, 3)

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }


def run_ml_family_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    models_to_run: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes full benchmark suite across all classifiers in the ML Family."""
    catalog = build_ml_model_catalog(seed=seed)
    if models_to_run is None:
        models_to_run = list(catalog.keys())

    results = {}
    for name in models_to_run:
        if name in catalog:
            res = train_and_eval_single_ml_model(
                model_name=name,
                clf=catalog[name],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
                scale_features=True,
            )
            results[name] = res["metrics"]

    return results
