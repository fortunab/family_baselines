# Dermatology Dual-Database: ISIC Archive + HAM10000 Benchmark

This guide documents the dermatology dual-database construction, metadata harmonization, and lesion-level split management for the **ISIC Archive** and **HAM10000** collections.

---

## 1. Clinical Context & Diagnostic Classes

The benchmark supports the 7 standard clinical dermoscopy categories:
1. `MEL`: Melanoma (Malignant melanocytic neoplasm, high referral priority)
2. `NV`: Melanocytic Nevus (Benign melanocytic proliferation)
3. `BCC`: Basal Cell Carcinoma (Common malignant keratinocytic cancer)
4. `AKIEC`: Actinic Keratosis / Intraepithelial Carcinoma (Pre-malignant squamous lesion)
5. `BKL`: Benign Keratosis-like Lesions (Seborrheic keratosis, solar lentigo, lichen planus-like keratosis)
6. `DF`: Dermatofibroma (Benign fibrohistiocytic tumor)
7. `VASC`: Vascular Lesions (Angiomas, pyogenic granulomas, hemorrhage)

---

## 2. Key Harmonization Principles

1. **Lesion ID Isolation**: In HAM10000, 10,015 images represent 7,470 unique lesions. Multiple images of the same lesion (e.g. cross-polarized vs contact non-polarized) must **never** be split across train and test sets.
2. **Standardized Diagnostic Labels**: Mapping disparate local clinical diagnostic nomenclature from the Medical University of Vienna (ViDIR) and the Cliff Rosendahl Queensland clinic into unified ICD-11 / ISIC taxonomies.
3. **Artifact Management**: Removing boundary vignetting, skin markings, and air bubbles through normalized ROI pre-filtering.

---

## 3. Running the Dermatology Benchmark

```bash
python run_isic_ham_benchmark.py
```
Exports results to `results/metrics/isic_ham_ml_benchmark_summary.json`.
