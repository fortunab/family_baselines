"""Machine Learning Families Taxonomy Benchmark Engine for Diabetic Retinopathy.

Organizes machine learning algorithms into 6 distinct clinical families:
1. Kernel & Margin Family (SVM RBF, SVM Linear, SVM Poly)
2. Bagging & Random Forest Family (Random Forest, Extra Trees, Balanced RF)
3. Boosting & Sequential Optimization Family (HistGBM, Gradient Boosting, AdaBoost)
4. Linear & Probabilistic GLM Family (Logistic Regression L2, ElasticNet, Gaussian NB)
5. Neural & Multi-Layer Representation Family (Deep MLP, Shallow Wide MLP)
6. Meta-Learning & Stacking Consensus Family (Soft Voting Ensemble, Stacking Classifier)

Evaluates intra-family dynamics, best-in-family models, and cross-family clinical comparisons.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from src.evaluator import evaluate_clinical_dr_metrics


@dataclass
class MLFamilyDefinition:
    """Represents an ML family with its descriptive clinical metadata."""

    family_id: str
    display_name: str
    clinical_rationale: str
    algorithms: Dict[str, str]  # model_key -> display_name


# Clinical Taxonomy Definitions
ML_TAXONOMY: Dict[str, MLFamilyDefinition] = {
    "kernel_margin": MLFamilyDefinition(
        family_id="kernel_margin",
        display_name="Kernel & Margin Family",
        clinical_rationale="Maximizes decision boundary margin in Hilbert space; resilient to high-dimensional radiomic feature noise.",
        algorithms={
            "svm_rbf": "SVM (RBF Kernel, C=10.0)",
            "svm_linear": "SVM (Linear Kernel, Calibrated)",
            "svm_poly": "SVM (Polynomial Kernel, Degree 3)",
        },
    ),
    "bagging_forests": MLFamilyDefinition(
        family_id="bagging_forests",
        display_name="Bagging & Random Forest Family",
        clinical_rationale="Averages decorrelated decision trees; immune to collinear color and texture moments.",
        algorithms={
            "random_forest": "Random Forest (150 Trees)",
            "extra_trees": "Extra Trees (Extremely Randomized)",
            "balanced_rf": "Subsample Balanced Random Forest",
        },
    ),
    "boosting_sequential": MLFamilyDefinition(
        family_id="boosting_sequential",
        display_name="Boosting & Optimization Family",
        clinical_rationale="Sequentially fits pseudo-residuals; excels at non-linear microaneurysm and vessel interactions.",
        algorithms={
            "hist_gradient_boosting": "Histogram Gradient Boosting (LightGBM-style)",
            "gradient_boosting": "Friedman Gradient Tree Boosting",
            "adaboost": "AdaBoost on Decision Stumps",
        },
    ),
    "linear_glm": MLFamilyDefinition(
        family_id="linear_glm",
        display_name="Linear & Probabilistic GLM Family",
        clinical_rationale="Yields directly interpretable log-odds ratios and calibrated clinical posterior probabilities.",
        algorithms={
            "logistic_regression_l2": "Multinomial Logistic Regression (L2 Ridge)",
            "logistic_regression_elasticnet": "Logistic Regression (ElasticNet, L1/L2)",
            "gaussian_nb": "Gaussian Naive Bayes (Likelihood Ratio)",
        },
    ),
    "neural_mlp": MLFamilyDefinition(
        family_id="neural_mlp",
        display_name="Neural & Representation Family",
        clinical_rationale="Learns hierarchical latent representations of multi-scale radiomic descriptors.",
        algorithms={
            "mlp_deep": "Deep MLP (3 Layers: 256-128-64)",
            "mlp_shallow": "Wide Shallow MLP (1 Layer: 128)",
        },
    ),
    "meta_stacking": MLFamilyDefinition(
        family_id="meta_stacking",
        display_name="Meta-Learning & Stacking Ensemble Family",
        clinical_rationale="Combines diverse inductive biases across distinct families; minimizes individual variance.",
        algorithms={
            "soft_voting_ensemble": "Soft Voting Meta-Ensemble (Top-5)",
            "stacking_classifier": "Stacking Classifier (Meta-Logistic Regression)",
        },
    ),
}


def build_taxonomy_model(model_key: str, scaler_type: str = "robust", seed: int = 42) -> Pipeline:
    """Builds a scikit-learn pipeline for any model in the clinical taxonomy."""
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()

    if model_key == "svm_rbf":
        clf = SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    elif model_key == "svm_linear":
        base_svc = SVC(C=1.0, kernel="linear", random_state=seed)
        clf = CalibratedClassifierCV(estimator=base_svc, cv=3)
    elif model_key == "svm_poly":
        base_poly = SVC(C=5.0, kernel="poly", degree=3, random_state=seed)
        clf = CalibratedClassifierCV(estimator=base_poly, cv=3)
    elif model_key == "random_forest":
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
    elif model_key == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
    elif model_key == "balanced_rf":
        clf = RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        )
    elif model_key == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
    elif model_key == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=seed)
    elif model_key == "adaboost":
        clf = AdaBoostClassifier(n_estimators=80, learning_rate=0.1, random_state=seed)
    elif model_key == "logistic_regression_l2":
        clf = LogisticRegression(C=2.0, penalty="l2", max_iter=500, random_state=seed)
    elif model_key == "logistic_regression_elasticnet":
        clf = LogisticRegression(
            C=1.5, penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=600, random_state=seed
        )
    elif model_key == "gaussian_nb":
        clf = GaussianNB()
    elif model_key == "mlp_deep":
        clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=350, alpha=0.001, random_state=seed)
    elif model_key == "mlp_shallow":
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, alpha=0.001, random_state=seed)
    elif model_key == "soft_voting_ensemble":
        e1 = SVC(C=10.0, kernel="rbf", probability=True, random_state=seed)
        e2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        e3 = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        e4 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=seed)
        e5 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, random_state=seed)
        clf = VotingClassifier(
            estimators=[("svm", e1), ("rf", e2), ("et", e3), ("hgb", e4), ("mlp", e5)],
            voting="soft",
        )
    elif model_key == "stacking_classifier":
        base_estimators = [
            ("svm", SVC(C=10.0, kernel="rbf", probability=True, random_state=seed)),
            ("et", ExtraTreesClassifier(n_estimators=80, max_depth=8, random_state=seed, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=80, random_state=seed)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=seed)),
        ]
        clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=300),
            cv=3,
        )
    else:
        raise ValueError(f"Unknown taxonomy model key: {model_key}")

    return Pipeline([("scaler", scaler), ("classifier", clf)])


class MLFamilyTaxonomyBenchmarkEngine:
    """Evaluates algorithms grouped strictly by their clinical machine learning families."""

    def __init__(self, class_names: List[str], scaler_type: str = "robust", seed: int = 42):
        self.class_names = class_names
        self.scaler_type = scaler_type
        self.seed = seed

    def evaluate_model(
        self,
        model_key: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Trains a model and computes clinical metrics and execution timing."""
        pipe = build_taxonomy_model(model_key, scaler_type=self.scaler_type, seed=self.seed)

        t_tr_start = time.perf_counter()
        pipe.fit(X_train, y_train)
        t_tr = time.perf_counter() - t_tr_start

        t_inf_start = time.perf_counter()
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
        t_inf = time.perf_counter() - t_inf_start

        lat_ms = (t_inf / len(y_test)) * 1000.0

        metrics = evaluate_clinical_dr_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names,
            referral_threshold=2,
        )

        metrics["model_key"] = model_key
        metrics["training_time_sec"] = round(t_tr, 3)
        metrics["inference_latency_ms"] = round(lat_ms, 3)
        return metrics

    def run_family_benchmark(
        self,
        family_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Benchmarks all constituent models within a designated family."""
        if family_id not in ML_TAXONOMY:
            raise ValueError(f"Unknown family: {family_id}")

        family_def = ML_TAXONOMY[family_id]
        model_results = {}
        for m_key, m_name in family_def.algorithms.items():
            res = self.evaluate_model(m_key, X_train, y_train, X_test, y_test)
            res["display_name"] = m_name
            model_results[m_key] = res

        # Family Aggregation
        qwks = [m["quadratic_weighted_kappa"] for m in model_results.values()]
        accs = [m["accuracy"] for m in model_results.values()]
        f1s = [m["macro_f1"] for m in model_results.values()]
        ref_sens = [m["binary_referral"]["referral_sensitivity"] for m in model_results.values()]
        specs = [m["macro_specificity"] for m in model_results.values()]
        latencies = [m["inference_latency_ms"] for m in model_results.values()]

        # Identify Best in Family (by QWK)
        best_key = max(model_results.keys(), key=lambda k: model_results[k]["quadratic_weighted_kappa"])

        family_summary = {
            "family_id": family_id,
            "display_name": family_def.display_name,
            "clinical_rationale": family_def.clinical_rationale,
            "models": model_results,
            "best_in_family_key": best_key,
            "best_in_family_name": family_def.algorithms[best_key],
            "best_in_family_qwk": model_results[best_key]["quadratic_weighted_kappa"],
            "best_in_family_accuracy": model_results[best_key]["accuracy"],
            "best_in_family_referral_sens": model_results[best_key]["binary_referral"]["referral_sensitivity"],
            "aggregated": {
                "mean_qwk": round(float(np.mean(qwks)), 4),
                "max_qwk": round(float(np.max(qwks)), 4),
                "mean_accuracy": round(float(np.mean(accs)), 4),
                "mean_macro_f1": round(float(np.mean(f1s)), 4),
                "mean_referral_sensitivity": round(float(np.mean(ref_sens)), 4),
                "mean_specificity": round(float(np.mean(specs)), 4),
                "mean_latency_ms": round(float(np.mean(latencies)), 3),
            },
        }

        return family_summary

    def run_all_families_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Runs the complete benchmark across all 6 machine learning families."""
        all_results = {}
        for f_id in ML_TAXONOMY:
            all_results[f_id] = self.run_family_benchmark(f_id, X_train, y_train, X_test, y_test)
        return all_results
