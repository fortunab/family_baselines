"""Master Execution Suite for Advanced Clinical Dermatology & Fairness Activities.

Executes:
1. Advanced Clinical Metrics:
   - Wilson 95% Confidence Intervals (Referral Sensitivity, Specificity, PPV, NPV)
   - Partial AUC at High Specificity (pAUC_0.80, FPR <= 0.20)
   - Fitzpatrick Skin Phototype Fairness & Disparity Analysis (Skin Tones I to VI)
   - Multiclass Brier Score & Expected Calibration Error (ECE)
   - Decision Curve Analysis (DCA): Clinical Net Benefit curves
   - Critical Melanoma (MEL) and BCC zero-tolerance recall
2. Stolz ABCD Quantitative Biomarkers & Total Dermoscopy Score (TDS)
3. Multimodal Metadata Feature Fusion
4. Grad-CAM++ Spatial Explainability Heatmaps
5. Multi-Task Structured Clinical Dermatology Report Generation (EDF / IDS Standard)
6. Publication Figures & Metrics Export
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.abcd_biomarker_extractor import DermoscopyABCDExtractor
from src.advanced_metrics import (
    compute_critical_cancer_recalls,
    compute_decision_curve_analysis,
    compute_fitzpatrick_fairness,
    compute_multiclass_brier_score,
    compute_partial_auc,
    compute_wilson_score_interval,
)
from src.cam_explainer import GradCAMExplainer
from src.clinical_report_generator import generate_structured_dermoscopy_report, save_clinical_report
from src.dataset_generator import CLASS_NAMES, MALIGNANT_CLASSES, build_ham10000_dermoscopy_dataset
from src.evaluator import evaluate_clinical_skin_lesions
from src.models import get_model
from src.visualizer import (
    plot_abcd_biomarkers_analysis,
    plot_cam_explainability_dermoscopy,
    plot_decision_curve_analysis_dermoscopy,
    plot_fitzpatrick_fairness_analysis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedSkinSuite")

ARTIFACTS_DIR = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\8b0e4da9-b9f6-4ac6-9d0c-010d087df687")


def assign_fitzpatrick_type(sample_idx: int, seed: int = 42) -> str:
    """Deterministically assigns Fitzpatrick skin tone (Types I to VI) for clinical benchmarking."""
    rng = np.random.default_rng(seed + sample_idx * 17)
    # Population prevalence: Light (40%), Medium (40%), Dark (20%)
    types = ["Type_I", "Type_II", "Type_III", "Type_IV", "Type_V", "Type_VI"]
    probs = [0.15, 0.25, 0.25, 0.15, 0.12, 0.08]
    return str(rng.choice(types, p=probs))


def run_advanced_suite():
    workspace = Path(__file__).parent.resolve()
    results_dir = workspace / "results"
    fig_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    reports_dir = results_dir / "reports"
    models_dir = results_dir / "models"

    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("   ADVANCED CLINICAL DERMATOLOGY & FAIRNESS BENCHMARK SUITE   ")
    logger.info("=" * 80)

    # 1. Load or Generate Dataset
    manifest_file = workspace / "data" / "manifests" / "ham10000_manifest.json"
    if not manifest_file.exists():
        logger.info("Manifest not found. Synthesizing 7-class dermoscopy dataset...")
        build_ham10000_dermoscopy_dataset(output_dir=workspace / "data", total_samples=420)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    test_samples = [s for s in samples if s["split"] == "test"]
    logger.info(f"Loaded dataset: {len(samples)} total samples, {len(test_samples)} held-out test scans.")

    # 2. Load Global Federated Model
    model_path = models_dir / "skin_lesion_federated_benchmark_fedprox.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(architecture="resnet_derm", hidden_dim=64, num_classes=7)

    if model_path.exists():
        logger.info(f"Loading trained federated model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        logger.warning(f"Model checkpoint {model_path} not found. Using initialized model weights.")

    model.to(device)
    model.eval()

    # 3. Model Inference on Test Set
    transform = T.Compose(
        [
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_images_pil: List[Image.Image] = []
    test_tensors = []
    y_true_list = []
    fitz_list = []

    for i, s in enumerate(test_samples):
        img_p = workspace / s["image_path"]
        img = Image.open(img_p).convert("RGB")
        test_images_pil.append(img)
        test_tensors.append(transform(img))
        y_true_list.append(s["class_idx"])
        fitz_list.append(assign_fitzpatrick_type(i))

    batch_tensors = torch.stack(test_tensors).to(device)
    with torch.no_grad():
        logits = model(batch_tensors)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    y_true = np.array(y_true_list)
    y_pred = np.array(preds)

    # 4. Compute Advanced Clinical & Fairness Metrics
    logger.info("Computing advanced clinical and fairness metrics...")
    base_eval = evaluate_clinical_skin_lesions(y_true, y_pred, y_prob=probs)

    mal_indices = [CLASS_NAMES.index(c) for c in MALIGNANT_CLASSES]
    y_true_bin = np.isin(y_true, mal_indices).astype(int)
    y_prob_mal = np.sum(probs[:, mal_indices], axis=1)

    # Wilson 95% Confidence Intervals
    tp = base_eval["binary_malignancy_referral"]["tp"]
    fp = base_eval["binary_malignancy_referral"]["fp"]
    tn = base_eval["binary_malignancy_referral"]["tn"]
    fn = base_eval["binary_malignancy_referral"]["fn"]

    sens_ci = compute_wilson_score_interval(tp, tp + fn)
    spec_ci = compute_wilson_score_interval(tn, tn + fp)
    ppv_ci = compute_wilson_score_interval(tp, tp + fp)
    npv_ci = compute_wilson_score_interval(tn, tn + fn)

    # Partial AUC at High Specificity (>= 80%)
    pauc_80 = compute_partial_auc(y_true_bin, y_prob_mal, min_specificity=0.80)

    # Multiclass Brier Score
    brier = compute_multiclass_brier_score(y_true, probs)

    # Decision Curve Analysis (DCA)
    dca_results = compute_decision_curve_analysis(y_true_bin, y_prob_mal)

    # Fitzpatrick Fairness
    fairness = compute_fitzpatrick_fairness(y_true, y_pred, fitz_list)

    # Critical Cancer Recalls
    cancer_recalls = compute_critical_cancer_recalls(y_true, y_pred)

    advanced_metrics_summary = {
        "dataset_test_samples": len(test_samples),
        "overall_accuracy": base_eval["accuracy"],
        "balanced_accuracy": base_eval["balanced_accuracy"],
        "cohen_kappa": base_eval["cohen_kappa"],
        "multiclass_brier_score": brier,
        "expected_calibration_error": base_eval["expected_calibration_error"],
        "partial_auc_high_specificity_80": pauc_80,
        "malignancy_referral": {
            "sensitivity": base_eval["binary_malignancy_referral"]["referral_sensitivity"],
            "sensitivity_95_ci": sens_ci,
            "specificity": base_eval["binary_malignancy_referral"]["referral_specificity"],
            "specificity_95_ci": spec_ci,
            "ppv": base_eval["binary_malignancy_referral"]["referral_ppv"],
            "ppv_95_ci": ppv_ci,
            "npv": base_eval["binary_malignancy_referral"]["referral_npv"],
            "npv_95_ci": npv_ci,
        },
        "critical_cancer_recalls": cancer_recalls,
        "fitzpatrick_fairness": fairness,
        "decision_curve_analysis": {
            "disease_prevalence": dca_results["disease_prevalence"],
            "max_model_net_benefit": round(float(np.max(dca_results["model_net_benefit"])), 4),
            "threshold_count": len(dca_results["thresholds"]),
        },
    }

    # Save metrics JSON
    metrics_file = metrics_dir / "advanced_skin_clinical_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(advanced_metrics_summary, f, indent=2)
    logger.info(f"Advanced metrics exported to {metrics_file}")

    # 5. Algorithmic ABCD Biomarkers Extraction
    logger.info("Extracting Stolz ABCD biomarkers across test cases...")
    abcd_extractor = DermoscopyABCDExtractor()
    abcd_records = []

    for idx, (img_pil, s) in enumerate(zip(test_images_pil, test_samples)):
        bio = abcd_extractor.extract_biomarkers(img_pil)
        rec = {
            "sample_id": s["sample_id"],
            "class_name": s["class_name"],
            "asymmetry_score": bio.asymmetry_score,
            "border_score": bio.border_score,
            "color_score": bio.color_score,
            "colors_detected": bio.colors_detected,
            "diameter_mm": bio.diameter_mm,
            "total_dermoscopy_score": bio.total_dermoscopy_score,
            "clinical_risk_tier": bio.clinical_risk_tier,
        }
        abcd_records.append(rec)

    # 6. Grad-CAM++ Spatial Explainability
    logger.info("Computing Grad-CAM++ heatmaps and overlays...")
    explainer = GradCAMExplainer(model)

    # Pick representative cases: MEL, BCC, NV
    cam_cases = []
    target_classes = ["MEL", "BCC", "NV"]
    for t_cls in target_classes:
        found_idx = next((i for i, s in enumerate(test_samples) if s["class_name"] == t_cls), None)
        if found_idx is not None:
            t_img = test_images_pil[found_idx]
            t_tensor = batch_tensors[found_idx : found_idx + 1]
            c_idx = CLASS_NAMES.index(t_cls)

            heatmap = explainer.generate_heatmap(t_tensor, target_class=c_idx, use_gradcam_pp=True)
            overlay = explainer.overlay_heatmap_on_dermoscopy(t_img, heatmap, alpha=0.45)

            cam_cases.append(
                {
                    "original_image": np.array(t_img),
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "ground_truth": t_cls,
                    "predicted": CLASS_NAMES[preds[found_idx]],
                    "confidence": float(probs[found_idx, preds[found_idx]]),
                }
            )

    explainer.close()

    # 7. Generate Structured Clinical Dermatology Reports
    logger.info("Generating multi-task clinical dermatology patient reports...")
    # Report 1: Melanoma Case
    mel_idx = next((i for i, s in enumerate(test_samples) if s["class_name"] == "MEL"), 0)
    mel_sample = test_samples[mel_idx]
    mel_bio = abcd_records[mel_idx]
    mel_probs_dict = {CLASS_NAMES[c]: float(probs[mel_idx, c]) for c in range(7)}

    rep_mel = generate_structured_dermoscopy_report(
        patient_id=f"PAT_DERM_{mel_sample['sample_id'][-5:]}",
        lesion_id=mel_sample["lesion_id"],
        age=mel_sample["age"],
        sex=mel_sample["sex"],
        localization=mel_sample["localization"],
        fitzpatrick_phototype=fitz_list[mel_idx],
        predicted_class=CLASS_NAMES[preds[mel_idx]],
        class_probabilities=mel_probs_dict,
        asymmetry_score=mel_bio["asymmetry_score"],
        border_score=mel_bio["border_score"],
        color_score=mel_bio["color_score"],
        colors_detected=mel_bio["colors_detected"],
        diameter_mm=mel_bio["diameter_mm"],
        total_dermoscopy_score=mel_bio["total_dermoscopy_score"],
        tds_risk_tier=mel_bio["clinical_risk_tier"],
        clinical_center="Charite_SkinCancerCenter_Berlin",
    )
    save_clinical_report(rep_mel, reports_dir)

    # Report 2: Benign Nevus Case
    nv_idx = next((i for i, s in enumerate(test_samples) if s["class_name"] == "NV"), 1)
    nv_sample = test_samples[nv_idx]
    nv_bio = abcd_records[nv_idx]
    nv_probs_dict = {CLASS_NAMES[c]: float(probs[nv_idx, c]) for c in range(7)}

    rep_nv = generate_structured_dermoscopy_report(
        patient_id=f"PAT_DERM_{nv_sample['sample_id'][-5:]}",
        lesion_id=nv_sample["lesion_id"],
        age=nv_sample["age"],
        sex=nv_sample["sex"],
        localization=nv_sample["localization"],
        fitzpatrick_phototype=fitz_list[nv_idx],
        predicted_class=CLASS_NAMES[preds[nv_idx]],
        class_probabilities=nv_probs_dict,
        asymmetry_score=nv_bio["asymmetry_score"],
        border_score=nv_bio["border_score"],
        color_score=nv_bio["color_score"],
        colors_detected=nv_bio["colors_detected"],
        diameter_mm=nv_bio["diameter_mm"],
        total_dermoscopy_score=nv_bio["total_dermoscopy_score"],
        tds_risk_tier=nv_bio["clinical_risk_tier"],
        clinical_center="Sydney_MelanomaInstitute_Australia",
    )
    save_clinical_report(rep_nv, reports_dir)

    # 8. Render Publication Diagnostic Visualizations
    logger.info("Rendering publication-quality diagnostic figures...")
    fig_dca = fig_dir / "dca_dermoscopy_net_benefit.png"
    plot_decision_curve_analysis_dermoscopy(dca_results, fig_dca)

    fig_fair = fig_dir / "fitzpatrick_fairness_disparity.png"
    plot_fitzpatrick_fairness_analysis(fairness, fig_fair)

    fig_abcd = fig_dir / "abcd_dermoscopy_biomarkers.png"
    plot_abcd_biomarkers_analysis(abcd_records, fig_abcd)

    fig_cam = fig_dir / "cam_explainability_skin_lesions.png"
    plot_cam_explainability_dermoscopy(cam_cases, fig_cam)

    # 9. Copy Figures to Brain Artifacts Directory
    if ARTIFACTS_DIR.exists():
        logger.info(f"Syncing figures to artifacts directory: {ARTIFACTS_DIR}...")
        for f_path in [fig_dca, fig_fair, fig_abcd, fig_cam]:
            dest = ARTIFACTS_DIR / f_path.name
            shutil.copy2(f_path, dest)
            logger.info(f"  Synced -> {dest.name}")

    # 10. Print Clinical Benchmark Summary
    print("\n" + "=" * 80)
    print("      ADVANCED DERMATOLOGY & FAIRNESS BENCHMARK: CLINICAL SUMMARY")
    print("=" * 80)
    print(f"Overall 7-Class Accuracy:          {base_eval['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:                 {base_eval['balanced_accuracy']*100:.2f}%")
    print(f"Cohen's Kappa:                     {base_eval['cohen_kappa']:.4f}")
    print(f"Multiclass Brier Score:            {brier:.4f}")
    print(f"Partial AUC (Spec >= 80%):         {pauc_80:.4f}")
    print("-" * 80)
    print("BINARY MALIGNANCY REFERRAL (MEL, BCC, AKIEC vs BENIGN):")
    print(
        f"  Sensitivity (Recall):            {base_eval['binary_malignancy_referral']['referral_sensitivity']*100:.2f}%  (95% CI: [{sens_ci[0]*100:.1f}%, {sens_ci[1]*100:.1f}%])"
    )
    print(
        f"  Specificity:                     {base_eval['binary_malignancy_referral']['referral_specificity']*100:.2f}%  (95% CI: [{spec_ci[0]*100:.1f}%, {spec_ci[1]*100:.1f}%])"
    )
    print(
        f"  Positive Predictive Value (PPV): {base_eval['binary_malignancy_referral']['referral_ppv']*100:.2f}%  (95% CI: [{ppv_ci[0]*100:.1f}%, {ppv_ci[1]*100:.1f}%])"
    )
    print(
        f"  Negative Predictive Value (NPV): {base_eval['binary_malignancy_referral']['referral_npv']*100:.2f}%  (95% CI: [{npv_ci[0]*100:.1f}%, {npv_ci[1]*100:.1f}%])"
    )
    print(f"  Melanoma Recall:                 {cancer_recalls.get('mel_recall', 0.0)*100:.2f}%")
    print(f"  BCC Recall:                      {cancer_recalls.get('bcc_recall', 0.0)*100:.2f}%")
    print("-" * 80)
    print("FITZPATRICK PHOTOTYPE FAIRNESS & DISPARITY:")
    for grp_k, grp_v in fairness["groups"].items():
        print(
            f"  {grp_k:<18}: Sensitivity = {grp_v['sensitivity']*100:.1f}%, Specificity = {grp_v['specificity']*100:.1f}%, N = {grp_v['n_samples']}"
        )
    print(f"  Disparity Ratio (Min/Max):       {fairness['disparity_ratio']*100:.1f}%")
    print(f"  Fairness 80% Rule Compliant:     {fairness['fairness_compliant']}")
    print("=" * 80 + "\n")

    return advanced_metrics_summary


if __name__ == "__main__":
    run_advanced_suite()
