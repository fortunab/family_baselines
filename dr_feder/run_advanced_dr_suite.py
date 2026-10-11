"""Master Runner for Advanced Clinical Diabetic Retinopathy Activities & Metrics.

Executes:
1. Optical Image Quality Assurance (IQA Gatekeeper) on retinal fundus scans
2. Advanced clinical metric evaluations:
   - Quadratic vs Linear vs Unweighted Kappa trinity
   - Wilson 95% Confidence Intervals (Sensitivity, Specificity, PPV, NPV)
   - Multiclass Brier Score
   - Decision Curve Analysis (DCA) Clinical Net Benefit curves
   - Critical Grade 4 (PDR) Safety Recall
3. Vascular morphometric biomarker extraction (Vessel Density & Tortuosity Index)
4. Grad-CAM++ spatial explainability on microvascular hotspots
5. Multi-task structured clinical ophthalmology report generation
6. Publication diagnostic figure generation & artifact copying
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.advanced_metrics import compute_comprehensive_advanced_dr_metrics
from src.cam_explainer import GradCAMExplainer
from src.clinical_report_generator import (
    generate_structured_clinical_report,
    save_clinical_report,
)
from src.config_loader import load_config
from src.dataset_generator import (
    CLASS_NAMES,
    build_diabetic_retinopathy_dataset,
)
from src.iqa_gatekeeper import FundusIQAGatekeeper
from src.models import get_model
from src.vessel_biomarker_extractor import RetinalVesselExtractor
from src.visualizer import (
    plot_cam_explainability_cases,
    plot_decision_curve_analysis,
    plot_iqa_quality_distribution,
    plot_vessel_biomarkers_analysis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedDRSuite")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Advanced Clinical DR Suite")
    parser.add_argument("--config", type=str, default="configs/federated_config.toml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_root = Path(args.output_dir)
    fig_dir = out_root / "figures"
    metrics_dir = out_root / "metrics"
    reports_dir = out_root / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading
    logger.info("Loading 5-grade Diabetic Retinopathy dataset...")
    samples, _ = build_diabetic_retinopathy_dataset(
        output_dir="data",
        total_samples=cfg.dataset.total_samples,
        image_size=cfg.dataset.image_size,
        test_split_ratio=cfg.dataset.test_split_ratio,
        seed=cfg.experiment.seed,
    )
    test_samples = [s for s in samples if s.split == "test"]
    logger.info(f"Loaded {len(test_samples)} held-out clinical test scans.")

    # 2. Activity 1: Image Quality Assurance (IQA Gatekeeper)
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVITY 1: OPTICAL IMAGE QUALITY ASSURANCE (IQA GATEKEEPER)")
    logger.info("=" * 80)
    gatekeeper = FundusIQAGatekeeper()
    iqa_results = []
    for s in test_samples:
        res = gatekeeper.evaluate_image(s.image_path)
        iqa_results.append(res)

    gradable_count = sum(1 for r in iqa_results if r.is_gradable)
    logger.info(
        f"IQA Evaluation finished: {gradable_count}/{len(iqa_results)} scans diagnostically gradable "
        f"({gradable_count/len(iqa_results)*100:.1f}%)."
    )

    fig_iqa = fig_dir / "iqa_quality_distribution.png"
    plot_iqa_quality_distribution(iqa_results, fig_iqa)
    logger.info(f"Saved IQA distribution figure to {fig_iqa}")

    # 3. Model Loading & Inference
    device = torch.device(cfg.experiment.device)
    model = get_model(
        architecture=cfg.model.architecture,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)

    # Check for checkpoint
    ckpt_path = args.checkpoint or f"results/models/{cfg.experiment.name}_{cfg.experiment.algorithm}.pt"
    if Path(ckpt_path).exists():
        logger.info(f"Loading trained global model checkpoint from {ckpt_path}...")
        ckpt_data = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}. Using initialized model.")

    model.eval()

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_imgs_tensors = []
    y_true_list = []
    for s in test_samples:
        img_pil = Image.open(s.image_path).convert("RGB")
        test_imgs_tensors.append(eval_transform(img_pil))
        y_true_list.append(s.grade)

    batch_tensors = torch.stack(test_imgs_tensors).to(device)
    with torch.no_grad():
        logits = model(batch_tensors)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = np.array(y_true_list)

    # 4. Activity 2: Advanced Clinical Ophthalmology Metrics
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVITY 2: ADVANCED CLINICAL METRICS (QWK TRINITY, WILSON CIs, DCA)")
    logger.info("=" * 80)
    adv_metrics = compute_comprehensive_advanced_dr_metrics(
        y_true=y_true,
        y_pred=preds,
        y_prob=probs,
        referral_threshold_grade=cfg.dataset.referral_threshold_grade,
    )

    fig_dca = fig_dir / "dca_clinical_net_benefit.png"
    plot_decision_curve_analysis(adv_metrics["decision_curve_analysis"], fig_dca)
    logger.info(f"Saved Decision Curve Analysis figure to {fig_dca}")

    # 5. Activity 3: Vascular Biomarker Extraction
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVITY 3: VASCULAR SEGMENTATION & HEMODYNAMIC BIOMARKERS")
    logger.info("=" * 80)
    vessel_extractor = RetinalVesselExtractor()
    biomarkers_by_grade: Dict[int, List[Dict[str, float]]] = {g: [] for g in range(5)}
    vessel_results_list = []

    for s in test_samples:
        bio = vessel_extractor.extract_biomarkers(s.image_path)
        biomarkers_by_grade[s.grade].append(bio.to_dict())
        vessel_results_list.append(bio)

    fig_vessel = fig_dir / "vessel_biomarkers_analysis.png"
    plot_vessel_biomarkers_analysis(biomarkers_by_grade, fig_vessel)
    logger.info(f"Saved vessel biomarker figure to {fig_vessel}")

    # 6. Activity 4: Grad-CAM++ Explainability Suite
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVITY 4: GRAD-CAM++ SPATIAL EXPLAINABILITY HEATMAPS")
    logger.info("=" * 80)
    explainer = GradCAMExplainer(model)
    cam_cases = []

    # Pick 1 representative test case from each grade (0 to 4)
    for g in range(5):
        idx = next((i for i, s in enumerate(test_samples) if s.grade == g), 0)
        s_case = test_samples[idx]

        t_img = test_imgs_tensors[idx].unsqueeze(0)
        cam_map = explainer.generate_heatmap(t_img, target_class=int(preds[idx]), use_gradcam_pp=True)

        v_mask, _, _ = vessel_extractor.segment_vasculature(s_case.image_path)
        overlay = explainer.overlay_heatmap_on_fundus(s_case.image_path, cam_map, alpha=0.5)

        cam_cases.append({
            "grade": g,
            "grade_name": CLASS_NAMES[g],
            "ground_truth": s_case.grade,
            "predicted_grade": int(preds[idx]),
            "original_image": np.array(Image.open(s_case.image_path).convert("RGB")),
            "vessel_mask": v_mask,
            "cam_overlay": overlay,
            "sample_obj": s_case,
            "iqa_res": iqa_results[idx],
            "bio_res": vessel_results_list[idx],
            "probs": {CLASS_NAMES[k]: round(float(probs[idx, k]), 4) for k in range(5)},
        })

    fig_cam = fig_dir / "cam_explainability_dr_cases.png"
    plot_cam_explainability_cases(cam_cases, fig_cam)
    logger.info(f"Saved Grad-CAM++ explainability figure to {fig_cam}")
    explainer.close()

    # 7. Activity 5: Multi-Task Clinical Ophthalmology Report Generation
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVITY 5: MULTI-TASK STRUCTURED CLINICAL REPORT GENERATION")
    logger.info("=" * 80)
    for c_idx, c_info in enumerate(cam_cases[:2], 1):
        s_obj = c_info["sample_obj"]
        iqa_r = c_info["iqa_res"]
        bio_r = c_info["bio_res"]

        report = generate_structured_clinical_report(
            patient_id=s_obj.patient_id,
            examination_id=s_obj.sample_id,
            eye=s_obj.eye,
            camera_hardware=s_obj.camera_type,
            clinical_center="TeleScreening_EyePACS_California" if c_idx == 1 else "Hospital_Topcon_Paris_Messidor",
            iqa_gradable=iqa_r.is_gradable,
            iqa_quality_tier=iqa_r.quality_tier,
            iqa_quality_score=iqa_r.quality_score,
            predicted_grade=c_info["predicted_grade"],
            grade_probabilities=c_info["probs"],
            vessel_density_percent=bio_r.vessel_density_percent,
            vessel_tortuosity_index=bio_r.tortuosity_index,
        )

        rep_path = reports_dir / f"clinical_patient_report_{c_idx:02d}"
        save_clinical_report(report, rep_path)
        logger.info(f"Generated patient report: {rep_path.with_suffix('.md')}")

    # 8. Copy Figures to Brain Artifact Directory
    brain_artifact_dir = Path("C:/Users/Lenovo/.gemini/antigravity/brain/8b0e4da9-b9f6-4ac6-9d0c-010d087df687")
    if brain_artifact_dir.exists():
        for fig_f in [fig_iqa, fig_dca, fig_vessel, fig_cam]:
            dest = brain_artifact_dir / fig_f.name
            shutil.copy2(fig_f, dest)
            logger.info(f"Copied figure to artifact directory: {dest.name}")

    # 9. Export Comprehensive Metrics JSON
    export_data = {
        "benchmark_title": "Advanced Clinical Diabetic Retinopathy Activities & Metrics",
        "total_test_scans": len(test_samples),
        "iqa_gatekeeper_summary": {
            "gradable_count": gradable_count,
            "gradable_percent": round(gradable_count / len(iqa_results) * 100, 2),
            "mean_quality_score": round(float(np.mean([r.quality_score for r in iqa_results])), 2),
        },
        "advanced_metrics": adv_metrics,
        "vessel_biomarkers_summary": {
            f"grade_{g}": {
                "mean_density_pct": round(float(np.mean([x["vessel_density_percent"] for x in biomarkers_by_grade[g]])), 2),
                "mean_tortuosity": round(float(np.mean([x["tortuosity_index"] for x in biomarkers_by_grade[g]])), 3),
            }
            for g in range(5)
        },
    }
    json_path = metrics_dir / "advanced_dr_clinical_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    logger.info(f"Saved advanced metrics JSON to {json_path}")

    # 10. Formatted Console Presentation
    print("\n" + "=" * 105)
    print("        ADVANCED CLINICAL DIABETIC RETINOPATHY METRICS & ACTIVITIES SUMMARY        ")
    print("=" * 105)
    kappas = adv_metrics["kappas"]
    ref_ci = adv_metrics["referral_metrics_with_ci"]
    pdr_saf = adv_metrics["critical_pdr_safety"]

    print("1. KAPPA TRINITY & RELIABILITY AGREEMENT:")
    print(f"   • Unweighted Cohen's Kappa:      {kappas['unweighted_kappa']:.4f}")
    print(f"   • Linear Weighted Kappa:         {kappas['linear_weighted_kappa']:.4f}")
    print(f"   • Quadratic Weighted Kappa (QWK): {kappas['quadratic_weighted_kappa']:.4f}  (Gold-Standard Clinical Target: >= 0.80)")
    print(f"   • Multiclass Brier Score:        {adv_metrics['brier_score']:.4f}  (Probability Error)")

    print("\n2. PATIENT SAFETY & REFERRAL METRICS (GRADE >= 2) WITH WILSON 95% CIs:")
    print(f"   • Referral Sensitivity:          {ref_ci['sensitivity']*100:.2f}%  [95% CI: {ref_ci['sensitivity_95_ci'][0]*100:.1f}% - {ref_ci['sensitivity_95_ci'][1]*100:.1f}%]")
    print(f"   • Referral Specificity:          {ref_ci['specificity']*100:.2f}%  [95% CI: {ref_ci['specificity_95_ci'][0]*100:.1f}% - {ref_ci['specificity_95_ci'][1]*100:.1f}%]")
    print(f"   • Positive Predictive Value:     [95% CI: {ref_ci['ppv_95_ci'][0]*100:.1f}% - {ref_ci['ppv_95_ci'][1]*100:.1f}%]")
    print(f"   • Negative Predictive Value:     [95% CI: {ref_ci['npv_95_ci'][0]*100:.1f}% - {ref_ci['npv_95_ci'][1]*100:.1f}%]")

    print("\n3. CRITICAL GRADE 4 (PDR) EMERGENCIES:")
    print(f"   • PDR Referral Sensitivity:      {pdr_saf['pdr_referral_recall']*100:.2f}% (Zero-Tolerance Referral Policy)")
    print(f"   • PDR Exact Grade Recall:        {pdr_saf['pdr_exact_recall']*100:.2f}%")

    print("\n4. OPTICAL QUALITY ASSURANCE (IQA GATEKEEPER):")
    print(f"   • Gradable Scans:                {gradable_count}/{len(iqa_results)} ({gradable_count/len(iqa_results)*100:.1f}%)")
    print(f"   • Mean Image Quality Score:      {np.mean([r.quality_score for r in iqa_results]):.2f} / 100")

    print("\n5. VASCULAR HEMODYNAMIC BIOMARKERS:")
    for g in range(5):
        d_m = np.mean([x["vessel_density_percent"] for x in biomarkers_by_grade[g]])
        t_m = np.mean([x["tortuosity_index"] for x in biomarkers_by_grade[g]])
        print(f"   • Grade {g} ({CLASS_NAMES[g]:<16}): Density = {d_m:.2f}% | Tortuosity Index = {t_m:.3f}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
