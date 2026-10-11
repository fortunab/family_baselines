"""Test runner using Python standard library (no external pytest dependency required)."""

import sys
import tempfile
import traceback
from pathlib import Path

from tests.test_advanced_skin import (
    test_abcd_extractor,
    test_advanced_metrics,
    test_cam_explainer,
    test_clinical_report_generator,
    test_dca_dermoscopy,
    test_fitzpatrick_fairness,
    test_multimodal_fusion,
)
from tests.test_federated_pipeline import (
    test_client_local_training,
    test_clinical_evaluator,
    test_config_loaders,
    test_dataset_generation,
    test_dermoscopy_synthesis,
    test_dirichlet_partitioner,
    test_model_forward_and_proximal,
    test_server_aggregation_and_evaluation,
)


def run_all_tests():
    print("=" * 75)
    print("     FEDERATED DERMOSCOPY PIPELINE: TEST SUITE EXECUTION     ")
    print("=" * 75)

    tests = [
        ("Configuration Loaders (TOML & INI)", lambda tmp: test_config_loaders(tmp)),
        ("Dermoscopy Lesion Synthesis", lambda tmp: test_dermoscopy_synthesis()),
        ("HAM10000 Dataset Generation", lambda tmp: test_dataset_generation(tmp)),
        ("Non-IID Dirichlet Partitioner", lambda tmp: test_dirichlet_partitioner(tmp)),
        ("ResNet Model & FedProx Regularization", lambda tmp: test_model_forward_and_proximal()),
        ("Client Local Training Step", lambda tmp: test_client_local_training(tmp)),
        ("Server Aggregation & Global Evaluation", lambda tmp: test_server_aggregation_and_evaluation(tmp)),
        ("Clinical Dermatology Evaluator", lambda tmp: test_clinical_evaluator()),
        ("Advanced Clinical Metrics (Wilson CIs, Brier, pAUC)", lambda tmp: test_advanced_metrics()),
        ("Fitzpatrick Skin Tone Fairness & Disparity", lambda tmp: test_fitzpatrick_fairness()),
        ("Decision Curve Analysis (DCA Net Benefit)", lambda tmp: test_dca_dermoscopy()),
        ("Algorithmic ABCD Biomarkers & TDS Score", lambda tmp: test_abcd_extractor()),
        ("Multimodal Metadata Feature Fusion", lambda tmp: test_multimodal_fusion()),
        ("Grad-CAM++ Spatial Explainability Heatmaps", lambda tmp: test_cam_explainer()),
        ("Multi-Task Structured Clinical Dermatology Report", lambda tmp: test_clinical_report_generator()),
    ]

    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for name, test_func in tests:
            try:
                print(f"[RUNNING] {name}...", end=" ")
                test_func(tmp_path)
                print("PASSED")
                passed += 1
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                failed += 1

    print("=" * 75)
    print(f"RESULTS: {passed} PASSED | {failed} FAILED")
    print("=" * 75)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
