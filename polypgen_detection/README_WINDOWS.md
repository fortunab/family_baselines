# Running PolypGen2.0 Hugging Face Detection on Windows (PowerShell & CMD)

This guide walks you through running all 4 Foundation Object Detectors on the **Hugging Face `halyusuf/PolypGen2.0` dataset** on **Windows 10/11**.

---

## 1. Environment & Setup

Open **PowerShell** or **Command Prompt** and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_hf_detection_baseline

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running Individual Foundation Detectors

### Model 1: Microsoft Florence-2 (Prompt-Guided `<OD>`)
```powershell
python main_detection.py --model-name florence2
```

### Model 2: Google OWLv2 (Open-Vocabulary ViT Detector)
```powershell
python main_detection.py --model-name owlv2
```

### Model 3: Grounding DINO (Cross-Modal Detection Transformer)
```powershell
python main_detection.py --model-name grounding_dino
```

### Model 4: Google DeepMind PaliGemma (Spatial Token Grounding)
```powershell
python main_detection.py --model-name paligemma
```

---

## 3. Running All Detectors & Generating Comparison Leaderboard

```powershell
# Sequentially evaluate all 4 models:
python main_detection.py --model-name all

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_detectors.py
```

All results, JSON metrics, and diagnostic visual bounding-box overlays (`.png`) are saved in the `results/` folder.
