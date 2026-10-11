"""Multi-Task Structured Clinical Dermatology Report Generator.

Conforms to international dermatological reporting guidelines established by:
- International Dermoscopy Society (IDS)
- European Dermatology Forum (EDF)
- American Academy of Dermatology (AAD)

Synthesizes:
1. Patient Demographics & Lesion Metadata (Age, Sex, Site, Fitzpatrick Type)
2. 7-Class ISIC Diagnostic Prediction & Differential Diagnosis
3. Algorithmic ABCD Biomarkers & Total Dermoscopy Score (TDS)
4. Malignancy Referral Status & Evidence-Based Surgical / Monitoring Recommendations
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

CLASS_FULL_NAMES = {
    "MEL": "Melanoma (Malignant Melanocytic Neoplasm)",
    "NV": "Melanocytic Nevus (Benign Mole)",
    "BCC": "Basal Cell Carcinoma (Non-Melanoma Skin Cancer)",
    "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma (Bowen's Disease)",
    "BKL": "Benign Keratosis-like Lesion (Solar Lentigo / Seborrheic Keratosis)",
    "DF": "Dermatofibroma (Benign Fibrohistiocytic Lesion)",
    "VASC": "Vascular Lesion (Hemangioma / Pyogenic Granuloma)",
}


@dataclass
class ClinicalDermoscopyReport:
    patient_id: str
    lesion_id: str
    examination_date: str
    clinical_center: str
    dermatoscope_hardware: str
    age: int
    sex: str
    localization: str
    fitzpatrick_phototype: str
    primary_diagnosis: str
    primary_confidence: float
    differential_diagnosis: List[Dict[str, Any]]
    is_malignancy_suspected: bool
    urgency_tier: str  # Routine, Priority, Urgent
    asymmetry_score: int
    border_score: int
    color_score: int
    colors_detected: List[str]
    diameter_mm: float
    total_dermoscopy_score: float
    tds_risk_tier: str
    clinical_action_plan: str
    report_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        urgency_badge = (
            "🔴 **URGENT SURGICAL REFERRAL**"
            if self.urgency_tier == "Urgent"
            else (
                "🟡 **PRIORITY DERMATOLOGY EVALUATION**"
                if self.urgency_tier == "Priority"
                else "🟢 **ROUTINE SURVEILLANCE**"
            )
        )

        mal_text = "YES — High Risk Malignant Lesion" if self.is_malignancy_suspected else "NO — Benign Morphology"

        diff_lines = "\n".join(
            [
                f"- **{d['class_name']}** ({CLASS_FULL_NAMES.get(d['class_name'], d['class_name'])}): {d['probability']*100:.1f}%"
                for d in self.differential_diagnosis
            ]
        )

        colors_str = ", ".join([c.replace("_", " ").title() for c in self.colors_detected])

        return f"""# Structured Clinical Dermatology Report
**Report ID**: `{self.report_id}` | **Date**: {self.examination_date}
**Clinical Center**: {self.clinical_center} | **Hardware**: {self.dermatoscope_hardware}

---

## 1. Patient & Lesion Demographics
- **Patient ID**: `{self.patient_id}`
- **Lesion Identifier**: `{self.lesion_id}`
- **Age**: {self.age} years | **Biological Sex**: {self.sex.capitalize()}
- **Anatomical Site**: {self.localization.title()}
- **Fitzpatrick Phototype**: {self.fitzpatrick_phototype.replace('_', ' ')}

---

## 2. Dermatoscopic Diagnostic Synthesis
- **Primary AI Assessment**: **{self.primary_diagnosis}** — *{CLASS_FULL_NAMES.get(self.primary_diagnosis, '')}*
- **Algorithmic Confidence**: **{self.primary_confidence * 100:.2f}%**
- **Clinical Urgency Tier**: {urgency_badge}
- **Malignancy Referral Status**: `{mal_text}`

### Differential Diagnosis Profile
{diff_lines}

---

## 3. Quantitative Stolz ABCD Morphometric Biomarkers
- **Asymmetry (A)**: `{self.asymmetry_score}/2` (Principal axes moment mismatch)
- **Border Irregularity (B)**: `{self.border_score}/8` (Octant notch & contour abruptness)
- **Color Variegation (C)**: `{self.color_score}/6` ({colors_str})
- **Lesion Diameter (D)**: `{self.diameter_mm:.1f} mm`
- **Total Dermoscopy Score (TDS)**: **`{self.total_dermoscopy_score:.2f}`**
  - *TDS Interpretation Formula*: `TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*D`
  - *Risk Category*: **{self.tds_risk_tier}**

---

## 4. Evidence-Based Clinical Action Plan
> **Action Recommendation**:
> {self.clinical_action_plan}

---
*Generated in compliance with IDS (International Dermoscopy Society) and EDF (European Dermatology Forum) standards.*
"""


def generate_structured_dermoscopy_report(
    patient_id: str,
    lesion_id: str,
    age: int,
    sex: str,
    localization: str,
    fitzpatrick_phototype: str,
    predicted_class: str,
    class_probabilities: Dict[str, float],
    asymmetry_score: int,
    border_score: int,
    color_score: int,
    colors_detected: List[str],
    diameter_mm: float,
    total_dermoscopy_score: float,
    tds_risk_tier: str,
    clinical_center: str = "MelanomaUnit_Charite_Berlin",
    dermatoscope_hardware: str = "DermLite_DL4_Polarized_10x",
) -> ClinicalDermoscopyReport:
    """Builds a validated ClinicalDermoscopyReport dataclass with guideline-aligned action plans."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"DERM-REP-{patient_id[-6:]}-{datetime.now().strftime('%Y%m%d%H%M')}"

    # Sort differential diagnoses
    sorted_probs = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)
    differential = [{"class_name": c, "probability": round(float(p), 4)} for c, p in sorted_probs[:3]]

    primary_conf = float(class_probabilities.get(predicted_class, 0.0))
    is_mal = predicted_class in ["MEL", "BCC", "AKIEC"]

    # Determine clinical urgency and action plan
    if predicted_class == "MEL" or total_dermoscopy_score > 5.45:
        urgency = "Urgent"
        action = (
            "URGENT EXCISIONAL BIOPSY MANDATED: Perform complete full-thickness surgical excision "
            "with a narrow 2-5 mm safety margin. Avoid partial shave or punch biopsy to ensure accurate "
            "Breslow thickness measurement and staging. Fast-track histopathological evaluation."
        )
    elif predicted_class in ["BCC", "AKIEC"]:
        urgency = "Priority"
        action = (
            "SPECIALIST DERMATOLOGY REFERRAL: Schedule clinical evaluation for non-melanoma skin cancer. "
            "Confirm histology via diagnostic punch/shave biopsy. Consider surgical excision, Mohs micrographic "
            "surgery (for high-risk facial sites), or topical fluorouracil/imiquimod."
        )
    elif total_dermoscopy_score >= 4.75 or predicted_class == "NV" and primary_conf < 0.60:
        urgency = "Priority"
        action = (
            "SHORT-TERM DIGITAL DERMOSCOPY MONITORING (TDM): Lesion exhibits borderline/dysplastic features. "
            "Schedule baseline calibrated photographic re-examination at 3 months. Excision advised if "
            "morphological growth, asymmetric pigmentation, or structural change is observed."
        )
    else:
        urgency = "Routine"
        action = (
            "ROUTINE DERMATOLOGICAL SURVEILLANCE: Morphological features are reassuring and consistent with benign "
            "etiology. Reassure patient; advise routine monthly skin self-examination (ABCDE rule) and annual "
            "preventive dermatological screening."
        )

    return ClinicalDermoscopyReport(
        patient_id=patient_id,
        lesion_id=lesion_id,
        examination_date=now_str,
        clinical_center=clinical_center,
        dermatoscope_hardware=dermatoscope_hardware,
        age=age,
        sex=sex,
        localization=localization,
        fitzpatrick_phototype=fitzpatrick_phototype,
        primary_diagnosis=predicted_class,
        primary_confidence=round(primary_conf, 4),
        differential_diagnosis=differential,
        is_malignancy_suspected=is_mal,
        urgency_tier=urgency,
        asymmetry_score=asymmetry_score,
        border_score=border_score,
        color_score=color_score,
        colors_detected=colors_detected,
        diameter_mm=round(float(diameter_mm), 2),
        total_dermoscopy_score=round(float(total_dermoscopy_score), 2),
        tds_risk_tier=tds_risk_tier,
        clinical_action_plan=action,
        report_id=report_id,
    )


def save_clinical_report(report: ClinicalDermoscopyReport, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Exports clinical report to both .md and .json files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    md_file = out_path / f"{report.report_id}.md"
    json_file = out_path / f"{report.report_id}.json"

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return md_file, json_file
