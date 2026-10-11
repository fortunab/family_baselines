"""Machine Learning Family Benchmark Engine.

Implements 9 specialized machine learning algorithms:
1. Support Vector Classifier (RBF Kernel)
2. Support Vector Classifier (Linear Kernel, Calibrated)
3. Random Forest Classifier (150 trees)
4. Extra Trees Classifier (150 trees)
5. Histogram Gradient Boosting Classifier (LightGBM-style)
6. Multi-Layer Perceptron (MLP Neural Classifier)
7. Multinomial Logistic Regression (L2 / ElasticNet)
8. K-Nearest Neighbors (KNN, distance-weighted)
9. Soft Voting Ensemble (Ensemble consensus)

Evaluates cross-validation, feature scaling, inference timing, and domain shift.
"""

import time
from typing import Any, Dict, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from src.evaluator import evaluate_clinical_dr_metrics


def build_ml_model_pipeline(
    algo_name: str,
    scaler_type: str = "robust",
    seed: int = 42,
) -> Any:
    """Constructs a scikit-learn pipeline with preprocessing and the specified ML algorithm."""
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()

    if algo_name == "svm_rbf":
        clf = SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    elif algo_name == "svm_linear":
        base_svc = SVC(C=1.0, kernel="linear", random_state=seed)
        clf = CalibratedClassifierCV(estimator=base_svc, cv=3)
    elif algo_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
    elif algo_name == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
    elif algo_name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
    elif algo_name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, alpha=0.001, random_state=seed)
    elif algo_name == "logistic_regression":
        clf = LogisticRegression(C=2.0, max_iter=500, random_state=seed)
    elif algo_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    elif algo_name == "soft_voting_ensemble":
        e1 = SVC(C=10.0, kernel="rbf", probability=True, random_state=seed)
        e2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        e3 = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        e4 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=seed)
        e5 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, random_state=seed)
        clf = VotingClassifier(
            estimators=[("svm", e1), ("rf", e2), ("et", e3), ("hgb", e4), ("mlp", e5)],
            voting="soft",
        )
    else:
        raise ValueError(f"Unknown ML algorithm: {algo_name}")

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("classifier", clf),
        ]
    )

    return pipeline


class MLFamilyBenchmarkEngine:
    """Manages training, cross-validation, and multi-paradigm evaluation across ML algorithms."""

    def __init__(
        self,
        class_names: List[str],
        scaler_type: str = "robust",
        seed: int = 42,
    ):
        self.class_names = class_names
        self.scaler_type = scaler_type
        self.seed = seed

    def evaluate_model(
        self,
        algo_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Trains a single algorithm and evaluates performance and inference timing."""
        pipeline = build_ml_model_pipeline(algo_name, scaler_type=self.scaler_type, seed=self.seed)

        t_train_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time_sec = time.perf_counter() - t_train_start

        t_inf_start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)
        else:
            y_prob = None
        inf_time_sec = time.perf_counter() - t_inf_start

        latency_ms_per_sample = (inf_time_sec / len(y_test)) * 1000.0

        metrics = evaluate_clinical_dr_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names,
            referral_threshold=2,
        )

        metrics["algorithm"] = algo_name
        metrics["training_time_sec"] = round(train_time_sec, 3)
        metrics["inference_latency_ms"] = round(latency_ms_per_sample, 3)
        metrics["trained_pipeline"] = pipeline

        return metrics

    def run_stratified_cv(
        self,
        algo_name: str,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Runs Stratified 5-Fold Cross Validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        accs, qwks, f1s = [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            pipe = build_ml_model_pipeline(algo_name, scaler_type=self.scaler_type, seed=self.seed)
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)

            res = evaluate_clinical_dr_metrics(y_va, preds, class_names=self.class_names)
            accs.append(res["accuracy"])
            qwks.append(res["quadratic_weighted_kappa"])
            f1s.append(res["macro_f1"])

        return {
            "cv_accuracy_mean": round(float(np.mean(accs)), 4),
            "cv_accuracy_std": round(float(np.std(accs)), 4),
            "cv_qwk_mean": round(float(np.mean(qwks)), 4),
            "cv_qwk_std": round(float(np.std(qwks)), 4),
            "cv_macro_f1_mean": round(float(np.mean(f1s)), 4),
            "cv_macro_f1_std": round(float(np.std(f1s)), 4),
        }

    def run_full_suite(
        self,
        algorithms: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluates all algorithms in the ML family on train/test split."""
        results = {}
        for algo in algorithms:
            res = self.evaluate_model(algo, X_train, y_train, X_test, y_test)
            # Remove non-serializable pipeline from summary dict
            res.pop("trained_pipeline", None)
            results[algo] = res

        return results
