# PolypGen2.0 Hugging Face Foundation Object Detection Benchmark (COCO Format)

This repository provides reproducible implementations of **4 performant Foundation Object Detectors** adapted for the official **COCO-style [`halyusuf/PolypGen2.0`](https://huggingface.co/datasets/halyusuf/PolypGen2.0)** dataset on Hugging Face.

---

## 🏆 The 4 Foundation Object Detectors

| Model # | Foundation Architecture | Hugging Face ID | Paradigm & Capabilities | Dedicated Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Microsoft Florence-2** | `microsoft/Florence-2-large` / `base` | **Unified Prompt-Guided `<OD>`**: Pre-trained on 5.4B region annotations with SOTA spatial comprehension. | [README_FLORENCE2.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline/README_FLORENCE2.md) |
| **2** | **Google OWLv2** | `google/owlv2-base-patch16-ensemble` | **Open-Vocabulary Vision Transformer**: Dual-encoder ViT querying text embeddings ("polyp", "lesion"). | [README_OWLV2.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline/README_OWLV2.md) |
| **3** | **Grounding DINO** | `IDEA-Research/grounding-dino-base` | **Cross-Modal Detection Transformer**: SOTA cross-attention feature enhancement for mucosal lesions. | [README_GROUNDING_DINO.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline/README_GROUNDING_DINO.md) |
| **4** | **Google DeepMind PaliGemma** | `google/paligemma-3b-pt-224` / `448` | **Autoregressive Spatial Token Grounding**: Sequence-to-sequence coordinate tokenization (`<loc...>`). | [README_PALIGEMMA.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline/README_PALIGEMMA.md) |

---

## 🔬 Dataset Schema (`halyusuf/PolypGen2.0`)

```json
{
  "image_id": 142,
  "image": "<PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512>",
  "height": 512,
  "width": 512,
  "label": "polyp",
  "objects": {
    "bbox": [[120, 85, 260, 210]],     // [x, y, w, h] COCO standard
    "category": [0],                     // 0 = Polyp
    "area": [29400]
  },
  "tags": {
    "CenterID": "C1",                    // Multi-center clinical origin (C1 to C6)
    "PolypCount": 1,
    "filename": "C1_00142.png"
  }
}
```

---

## 📊 Evaluation Metrics & Protocol
- **COCO Primary Metric**: $\text{mAP}@[50:95]$ (averaged across IoU thresholds from 0.50 to 0.95 in 0.05 increments).
- **PASCAL VOC Metric**: $\text{mAP}@50$.
- **Strict Localization**: $\text{mAP}@75$.
- **Bounding Box Overlap**: Mean Intersection-over-Union ($\text{mIoU}$).
- **Detection Reliability**: Precision @ 50, Recall @ 50, F1-Score @ 50.
- **Multi-Center Breakdown**: Evaluates out-of-distribution (OOD) generalizability across centers $C1, C2, C3, C4, C5, C6$.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
# Navigate to the project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_hf_detection_baseline

# 1. Evaluate Model 1: Microsoft Florence-2
python main_detection.py --model-name florence2

# 2. Evaluate Model 2: Google OWLv2
python main_detection.py --model-name owlv2

# 3. Evaluate Model 3: Grounding DINO
python main_detection.py --model-name grounding_dino

# 4. Evaluate Model 4: Google DeepMind PaliGemma
python main_detection.py --model-name paligemma

# Or run all 4 sequentially:
python main_detection.py --model-name all

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_detectors.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
# Navigate to the project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline

# Activate virtual environment
source venv/bin/activate

# 1. Florence-2
python3 main_detection.py --model-name florence2

# 2. OWLv2
python3 main_detection.py --model-name owlv2

# 3. Grounding DINO
python3 main_detection.py --model-name grounding_dino

# 4. PaliGemma
python3 main_detection.py --model-name paligemma

# Compare all 4 models
python3 compare_detectors.py
```
