"""Test runner for Federated Diabetic Retinopathy Suite (Python standard library)."""

import sys
import tempfile
import traceback
from pathlib import Path

from tests.test_federated_dr_pipeline import (
    test_client_local_training,
    test_clinical_dr_evaluator,
    test_config_loaders,
    test_dirichlet_partitioner,
    test_dr_dataset_generation,
    test_fundus_synthesis,
    test_model_forward_and_proximal,
    test_server_aggregation_and_evaluation,
)
from tests.test_advanced_dr import (
    test_advanced_metrics,
    test_dca_calculation,
    test_iqa_gatekeeper,
    test_vessel_biomarkers,
    test_cam_explainer,
    test_clinical_report_generator,
)


def run_all_tests():
    print("=" * 75)
    print("    FEDERATED DIABETIC RETINOPATHY PIPELINE: TEST SUITE EXECUTION    ")
    print("=" * 75)

    tests = [
        ("Configuration Loaders (TOML & INI)", lambda tmp: test_config_loaders(tmp)),
        ("Retinal Fundus Image Synthesis", lambda tmp: test_fundus_synthesis()),
        ("5-Grade DR Dataset Generation", lambda tmp: test_dr_dataset_generation(tmp)),
        ("Non-IID Dirichlet Partitioner", lambda tmp: test_dirichlet_partitioner(tmp)),
        ("RetinaResNet & FedProx Regularization", lambda tmp: test_model_forward_and_proximal()),
        ("Client Local Training Step", lambda tmp: test_client_local_training(tmp)),
        ("Server Aggregation & Global Evaluation", lambda tmp: test_server_aggregation_and_evaluation(tmp)),
        ("Clinical Ophthalmology Evaluator (QWK)", lambda tmp: test_clinical_dr_evaluator()),
        ("Advanced Clinical Metrics (Wilson CIs, Brier, Kappa Trinity)", lambda tmp: test_advanced_metrics()),
        ("Decision Curve Analysis (DCA Net Benefit)", lambda tmp: test_dca_calculation()),
        ("Fundus IQA Gatekeeper (Optical Quality Scoring)", lambda tmp: test_iqa_gatekeeper()),
        ("Retinal Vessel Tree Biomarkers (Density & Tortuosity)", lambda tmp: test_vessel_biomarkers()),
        ("Grad-CAM++ Spatial Explainability Heatmaps", lambda tmp: test_cam_explainer()),
        ("Multi-Task Clinical Ophthalmology Report Generator", lambda tmp: test_clinical_report_generator()),
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
